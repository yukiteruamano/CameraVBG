import argparse
import os
import threading
import time

import cv2
import numpy as np

# Detección de capacidades
# MediaPipe (Motor Principal)
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mppython
    from mediapipe.tasks.python import vision as mpvision

    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("! ERROR: MediaPipe no instalado. Instálalo con: pip install mediapipe")

# PyVirtualCam (Salida a Zoom/Teams/OBS)
try:
    import pyvirtualcam

    HAS_VIRTUAL_CAM = True
except ImportError:
    HAS_VIRTUAL_CAM = False

# Guided Filter (Mejora de bordes)
try:
    _ = cv2.ximgproc.guidedFilter
    HAS_GUIDED_FILTER = True
except AttributeError:
    HAS_GUIDED_FILTER = False


# --- CLASES Y FUNCIONES ---
class WebcamStream:
    """Captura de video en hilo separado para maximizar FPS"""

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


class MediaPipeEngine:
    """Wrapper para el motor estándar de MediaPipe"""

    def __init__(self, model_path="models/selfie_segmenter.tflite"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encuentra el modelo: {model_path}")

        base_options = mppython.BaseOptions(model_asset_path=model_path)
        options = mpvision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True,
            running_mode=mpvision.RunningMode.IMAGE,
        )
        self.segmenter = mpvision.ImageSegmenter.create_from_options(options)

    def process(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        segmentation_result = self.segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask
        return category_mask.numpy_view().astype(np.float32)

    def close(self):
        self.segmenter.close()


def color_transfer(source, target, intensity=0.3):
    """Armonización de color entre el sujeto y el fondo"""
    # source y target entran como uint8 BGR
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # meanStdDev devuelve float64 (double)
    (l_mean_src, l_std_src) = cv2.meanStdDev(source_lab)
    (l_mean_tar, l_std_tar) = cv2.meanStdDev(target_lab)

    l, a, b = cv2.split(source_lab) # Estos son float32

    def scale_channel(ch, m_src, s_src, m_tar, s_tar):
        s_src = max(s_src, 1e-5)
        new_ch = (ch - m_src) * (s_tar / s_src) + m_tar
        return np.clip(new_ch, 0, 255)

    # CORRECCIÓN: Forzamos .astype(np.float32) porque scale_channel devolvía float64
    a_new = scale_channel(a, l_mean_src[1], l_std_src[1], l_mean_tar[1], l_std_tar[1]).astype(np.float32)
    b_new = scale_channel(b, l_mean_src[2], l_std_src[2], l_mean_tar[2], l_std_tar[2]).astype(np.float32)

    # Ahora ambos inputs son float32
    a_final = cv2.addWeighted(a_new, intensity, a, 1 - intensity, 0)
    b_final = cv2.addWeighted(b_new, intensity, b, 1 - intensity, 0)

    merged = cv2.merge([l, a_final, b_final])
    return cv2.cvtColor(merged.astype("uint8"), cv2.COLOR_LAB2BGR)


def sigmoid(x, slope, shift):
    return 1 / (1 + np.exp(-slope * (x - shift)))


def apply_light_wrap(foreground, background, mask, radius=20):
    """Efecto de envoltura de luz en los bordes"""
    mask_inv = 1.0 - mask
    k = radius if radius % 2 == 1 else radius + 1
    mask_inv_blurred = cv2.GaussianBlur(mask_inv, (k, k), 0)
    wrap_map = mask * mask_inv_blurred

    if len(wrap_map.shape) == 2:
        wrap_map = wrap_map[:, :, np.newaxis]

    bg_blur = cv2.GaussianBlur(background, (k, k), 0)
    fg_float = foreground.astype(np.float32)
    bg_float = bg_blur.astype(np.float32)

    wrapped = fg_float * (1.0 - wrap_map) + bg_float * wrap_map
    return wrapped.astype(np.uint8)


# --- CLI ---
parser = argparse.ArgumentParser(description="VBG: MediaPipe Edition")
parser.add_argument("--cam", type=int, default=0, help="Index cámara")
parser.add_argument("--bg_img", type=str, default=None, help="Ruta imagen de fondo")
parser.add_argument("--virtual", action="store_true", help="Activar Virtual Cam")
parser.add_argument("--model_path", type=str, default="models/selfie_segmenter.tflite")

# Calidad y Refinamiento
parser.add_argument(
    "--scale", type=float, default=0.5, help="Escala inferencia (0.1 a 1.0)"
)
parser.add_argument(
    "--temporal", type=float, default=0.8, help="Suavizado temporal (0.0 a 1.0)"
)
parser.add_argument("--mask_contrast", type=float, default=12.0)
parser.add_argument("--morph", type=int, default=5, help="Operaciones morfológicas")
parser.add_argument("--blur", type=int, default=7, help="Desenfoque de máscara")
parser.add_argument("--guided", action="store_true", help="Usar Guided Filter")
parser.add_argument("--light_wrap", type=int, default=20, help="Intensidad Light Wrap")
parser.add_argument(
    "--harmonize", type=float, default=0.0, help="Intensidad Armonización (0 a 1)"
)

args = parser.parse_args()


def main():
    if not HAS_MEDIAPIPE:
        return

    try:
        engine = MediaPipeEngine(args.model_path)
    except Exception as e:
        print(f"ERROR Iniciando MediaPipe: {e}")
        return

    print("Iniciando cámara...")
    vs = WebcamStream(src=args.cam).start()
    time.sleep(1.0)

    bg_image_source = cv2.imread(args.bg_img) if args.bg_img else None
    previous_mask_stable = None
    morph_kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.morph, args.morph))
        if args.morph > 0
        else None
    )

    # Virtual Cam Init
    cam_out = None
    if args.virtual and HAS_VIRTUAL_CAM:
        frame_test = vs.read()
        if frame_test is not None:
            h, w, _ = frame_test.shape
            cam_out = pyvirtualcam.Camera(
                width=w, height=h, fps=30, fmt=pyvirtualcam.PixelFormat.BGR
            )
            print("--> VIRTUAL CAM ACTIVA")

    use_guided = args.guided and HAS_GUIDED_FILTER
    print("SYSTEM ONLINE. Backend: MEDIAPIPE")

    while True:
        frame = vs.read()
        if frame is None:
            break
        h_orig, w_orig, _ = frame.shape

        # 1. Armonización
        frame_proc = (
            color_transfer(
                frame, cv2.resize(bg_image_source, (200, 200)), args.harmonize
            )
            if (args.harmonize > 0 and bg_image_source is not None)
            else frame
        )

        # 2. Inferencia
        work_w, work_h = (
            max(160, int(w_orig * args.scale)),
            max(90, int(h_orig * args.scale)),
        )
        frame_infer = cv2.resize(frame, (work_w, work_h))
        raw_mask = engine.process(frame_infer)

        # 3. Estabilidad Temporal
        if previous_mask_stable is None or previous_mask_stable.shape != raw_mask.shape:
            previous_mask_stable = raw_mask

        mask_stable = (raw_mask * (1 - args.temporal)) + (
            previous_mask_stable * args.temporal
        )
        previous_mask_stable = mask_stable

        # 4. Refinado y Upscaling
        mask_sigmoid = sigmoid(mask_stable, args.mask_contrast, 0.5)

        if morph_kernel is not None:
            _, mb = cv2.threshold(mask_sigmoid, 0.5, 1.0, cv2.THRESH_BINARY)
            mask_morph = cv2.morphologyEx(
                (mb * 255).astype(np.uint8), cv2.MORPH_CLOSE, morph_kernel
            )
            mask_base = (
                cv2.morphologyEx(mask_morph, cv2.MORPH_OPEN, morph_kernel).astype(
                    np.float32
                )
                / 255.0
            )
        else:
            mask_base = mask_sigmoid

        # Usamos INTER_LANCZOS4 para el Upscaling pero con GPU podría usarse FSR
        # Por el momento se queda así, ya que Intel apesta para hwaccel de IA en Linux
        mask_hd = cv2.resize(
            mask_base, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4
        )

        if use_guided:
            mask_final = cv2.ximgproc.guidedFilter(
                guide=frame_proc, src=mask_hd, radius=args.blur, eps=1e-6
            )
        else:
            k = args.blur if args.blur % 2 == 1 else args.blur + 1
            mask_final = (
                cv2.GaussianBlur(mask_hd, (k, k), 0) if args.blur > 0 else mask_hd
            )

        # 5. Composición Final
        mask_final = np.clip(mask_final, 0, 1)
        mask_3d = np.stack((mask_final,) * 3, axis=-1)
        bg_final = (
            cv2.resize(bg_image_source, (w_orig, h_orig))
            if bg_image_source is not None
            else np.zeros_like(frame)
        )

        frame_to_compose = (
            apply_light_wrap(frame_proc, bg_final, mask_final, args.light_wrap)
            if (args.light_wrap > 0 and bg_image_source is not None)
            else frame_proc
        )

        final_image = cv2.multiply(
            frame_to_compose, mask_3d, dtype=cv2.CV_8U
        ) + cv2.multiply(bg_final, 1.0 - mask_3d, dtype=cv2.CV_8U)

        # Salida
        cv2.imshow("VBG MediaPipe", final_image)
        if cam_out:
            cam_out.send(final_image)
            cam_out.sleep_until_next_frame()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vs.stop()
    if cam_out:
        cam_out.close()
    engine.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

