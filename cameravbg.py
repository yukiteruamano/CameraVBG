import argparse
import os
import queue
import threading
import time

import cv2
import numpy as np
from numba import float32, njit, prange, uint8, vectorize

# --- CONFIGURACIÓN E IMPORTACIONES ---

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mppython
    from mediapipe.tasks.python import vision as mpvision

    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("! ERROR: MediaPipe no instalado.")

try:
    import pyvirtualcam

    HAS_VIRTUAL_CAM = True
except ImportError:
    HAS_VIRTUAL_CAM = False

try:
    _ = cv2.ximgproc.guidedFilter
    HAS_GUIDED_FILTER = True
except AttributeError:
    HAS_GUIDED_FILTER = False

# --- CLASES ---


class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()


class MediaPipeEngine:
    def __init__(self, model_path="models/selfie_segmenter.tflite"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

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
        return segmentation_result.category_mask.numpy_view().astype(np.float32)

    def close(self):
        self.segmenter.close()


class FrameManager:
    """Gestor de frames optimizado"""

    def __init__(self, frame_skip=0, skip_non_essential=False):
        self.frame_skip = frame_skip
        self.skip_non_essential = skip_non_essential
        self.frame_counter = 0

    def update(self):
        self.frame_counter += 1

    def should_run_inference(self):
        if self.frame_skip <= 0:
            return True
        return (self.frame_counter % (self.frame_skip + 1)) == 0

    def should_run_effects(self):
        if not self.skip_non_essential:
            return True
        return self.should_run_inference()


class ProcessingPipeline:
    def __init__(self, num_workers=2):
        self.task_queue = queue.Queue(maxsize=num_workers * 2)
        self.result_queue = queue.Queue(maxsize=num_workers * 2)
        self.workers = []
        self.shutdown_flag = False
        self.num_workers = num_workers
        self.task_counter = 0
        self._setup_workers()

    def _setup_workers(self):
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f"PipelineWorker-{i}",
            )
            worker.start()
            self.workers.append(worker)

    def _worker_loop(self, worker_id):
        while not self.shutdown_flag:
            try:
                task_func, task_args, task_id = self.task_queue.get(timeout=0.1)
                try:
                    result = task_func(*task_args)
                    self.result_queue.put((task_id, result))
                except Exception as e:
                    print(f"Worker {worker_id} error: {e}")
                    self.result_queue.put((task_id, None))
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} critical error: {e}")
                break

    def submit_task(self, task_func, task_args, task_id=None):
        if task_id is None:
            task_id = f"task_{time.time()}_{self.task_counter}"
            self.task_counter += 1

        try:
            self.task_queue.put((task_func, task_args, task_id), timeout=0.01)
            return task_id
        except queue.Full:
            return self._execute_in_main_thread(task_func, task_args, task_id)

    def _execute_in_main_thread(self, task_func, task_args, task_id):
        try:
            result = task_func(*task_args)
            self.result_queue.put((task_id, result))
            return task_id
        except Exception as e:
            print(f"Main thread execution error: {e}")
            self.result_queue.put((task_id, None))
            return task_id

    def collect_result(self, task_id, timeout=0.1):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result_task_id, result = self.result_queue.get_nowait()
                if result_task_id == task_id:
                    return result
                else:
                    self.result_queue.put((result_task_id, result))
            except queue.Empty:
                time.sleep(0.001)
        return None

    def shutdown(self):
        self.shutdown_flag = True
        for worker in self.workers:
            worker.join(timeout=1.0)

    def get_worker_status(self):
        return {
            "active_workers": len([w for w in self.workers if w.is_alive()]),
            "task_queue_size": self.task_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
        }


# --- FUNCIONES AUXILIARES ---


def safe_collect_result(pipeline, task_id, default_value):
    """Recoge resultados de manera segura evitando errores con arrays"""
    if pipeline is None:
        return default_value

    result = pipeline.collect_result(task_id)
    if result is None:
        return default_value

    # Verificar que el resultado sea válido
    if hasattr(result, "shape") and len(result.shape) > 0 and result.size > 0:
        return result
    else:
        return default_value


def color_transfer(source, target_stats, intensity=0.3):
    """
    source: Frame de la cámara (uint8 BGR)
    target_stats: Tupla (mean, std) pre-calculada del fondo en espacio LAB
    """
    if target_stats is None:
        return source

    (l_mean_tar, l_std_tar) = target_stats

    # Convertimos solo el frame actual a LAB
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    (l_mean_src, l_std_src) = cv2.meanStdDev(source_lab)

    # Asegurarse de que las dimensiones sean correctas
    l_mean_src = l_mean_src.flatten()
    l_std_src = l_std_src.flatten()
    l_mean_tar = l_mean_tar.flatten()
    l_std_tar = l_std_tar.flatten()

    l, a, b = cv2.split(source_lab)

    def scale_channel(ch, m_src, s_src, m_tar, s_tar):
        # Evitamos división por cero
        s_src = max(s_src, 1e-5)
        new_ch = (ch - m_src) * (s_tar / s_src) + m_tar
        return np.clip(new_ch, 0, 255)

    # Aplicamos el escalado solo a los canales A y B (el color)
    a_new = scale_channel(a, l_mean_src[1], l_std_src[1], l_mean_tar[1], l_std_tar[1])
    b_new = scale_channel(b, l_mean_src[2], l_std_src[2], l_mean_tar[2], l_std_tar[2])

    # Mezclamos con el original según la intensidad
    a_final = cv2.addWeighted(a_new.astype(np.float32), intensity, a, 1 - intensity, 0)
    b_final = cv2.addWeighted(b_new.astype(np.float32), intensity, b, 1 - intensity, 0)

    merged = cv2.merge([l, a_final, b_final])
    return cv2.cvtColor(merged.astype("uint8"), cv2.COLOR_LAB2BGR)


@vectorize([float32(float32, float32, float32)], target="parallel")
def sigmoid(x, slope, shift):
    return 1.0 / (1.0 + np.exp(-slope * (x - shift)))


def wait_for_camera_ready(vs, max_attempts=10):
    time.sleep(0.5)
    for attempt in range(max_attempts):
        frame = vs.read()
        if frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            if h > 0 and w > 0:
                print(f"Cámara lista: {w}x{h}")
                return True
        time.sleep(0.2)
    return False


def apply_light_wrap(foreground, bg_blurred, mask, radius=20):
    mask_inv = 1.0 - mask
    k = radius if radius % 2 == 1 else radius + 1

    mask_inv_blurred = cv2.GaussianBlur(mask_inv, (k, k), 0)
    wrap_map = mask * mask_inv_blurred

    if len(wrap_map.shape) == 2:
        wrap_map = wrap_map[:, :, np.newaxis]

    return (
        foreground.astype(np.float32) * (1.0 - wrap_map)
        + bg_blurred.astype(np.float32) * wrap_map
    ).astype(np.uint8)


def guided_filter_task(frame_proc, mask_hd, blur_radius):
    try:
        if blur_radius > 0:
            return cv2.ximgproc.guidedFilter(
                guide=frame_proc, src=mask_hd, radius=blur_radius, eps=1e-6
            )
        return mask_hd
    except Exception as e:
        print(f"Guided filter error: {e}")
        return mask_hd


def light_wrap_task(frame_proc, bg_blurred, mask_final, radius):
    try:
        if radius > 0 and bg_blurred is not None:
            return apply_light_wrap(frame_proc, bg_blurred, mask_final, radius)
        return frame_proc
    except Exception as e:
        print(f"Light wrap error: {e}")
        return frame_proc


def morph_operations_task(mask_sigmoid, morph_kernel):
    try:
        if morph_kernel is not None:
            mask_binary = (mask_sigmoid > 0.5).astype(np.uint8) * 255
            mask_morph = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, morph_kernel)
            return (
                cv2.morphologyEx(mask_morph, cv2.MORPH_OPEN, morph_kernel).astype(
                    np.float32
                )
                / 255.0
            )
        return mask_sigmoid
    except Exception as e:
        print(f"Morph operations error: {e}")
        return mask_sigmoid


@njit(parallel=True, fastmath=True, cache=True)
def optimized_blend(fg, bg, mask):
    h, w, c = fg.shape
    out = np.empty((h, w, c), dtype=np.uint8)
    inv_mask = 1.0 - mask
    for i in prange(h):
        for j in range(w):
            m = mask[i, j]
            inv_m = inv_mask[i, j]
            for k in range(c):
                out[i, j, k] = uint8(fg[i, j, k] * m + bg[i, j, k] * inv_m)
    return out


# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument("--cam", type=int, default=0)
parser.add_argument("--bg_img", type=str, default=None)
parser.add_argument("--virtual", action="store_true")
parser.add_argument("--model_path", type=str, default="models/selfie_segmenter.tflite")
parser.add_argument("--scale", type=float, default=0.5)
parser.add_argument("--temporal", type=float, default=0.8)
parser.add_argument("--mask_contrast", type=float, default=12.0)
parser.add_argument("--morph", type=int, default=5)
parser.add_argument("--blur", type=int, default=7)
parser.add_argument("--guided", action="store_true")
parser.add_argument("--light_wrap", type=int, default=20)
parser.add_argument("--harmonize", type=float, default=0.0)
parser.add_argument(
    "--frame_skip", type=int, default=0, help="Saltar inferencia cada N frames (Ej: 2)"
)
parser.add_argument("--skip_non_essential", action="store_true")
parser.add_argument(
    "--multi_thread",
    type=int,
    default=0,
    help="Multi-hilo (0=desactivado, 2=2 workers, 4=4 workers)",
)

args = parser.parse_args()


def main():
    if not HAS_MEDIAPIPE:
        return

    try:
        engine = MediaPipeEngine(args.model_path)
    except Exception as e:
        print(f"ERROR MP: {e}")
        return

    print("Iniciando cámara...")
    vs = WebcamStream(src=args.cam).start()
    if not wait_for_camera_ready(vs):
        print("Fallo al iniciar cámara")
        return

    frame_ref = vs.read()
    h_orig, w_orig = frame_ref.shape[:2]

    cam_out = None
    if args.virtual:
        if HAS_VIRTUAL_CAM:
            try:
                cam_out = pyvirtualcam.Camera(
                    width=w_orig,
                    height=h_orig,
                    fps=30,
                    fmt=pyvirtualcam.PixelFormat.BGR,
                )
                print("--> VIRTUAL CAM ACTIVA (BGR)")
            except Exception as e:
                print(f"Error al inicializar cámara virtual: {e}")
                cam_out = None
        else:
            print("Advertencia: PyVirtualCam no está instalado")
            cam_out = None

    bg_image_ready = None
    bg_blurred_light_wrap = None
    bg_stats = None

    if args.bg_img:
        bg_raw = cv2.imread(args.bg_img)
        if bg_raw is not None:
            bg_image_ready = cv2.resize(bg_raw, (w_orig, h_orig))
            k_wrap = (
                args.light_wrap if args.light_wrap % 2 == 1 else args.light_wrap + 1
            )
            if k_wrap > 0:
                bg_blurred_light_wrap = cv2.GaussianBlur(
                    bg_image_ready, (k_wrap, k_wrap), 0
                )
            if args.harmonize > 0:
                bg_small = cv2.resize(bg_image_ready, (200, 200))
                bg_lab = cv2.cvtColor(bg_small, cv2.COLOR_BGR2LAB).astype("float32")
                bg_stats = cv2.meanStdDev(bg_lab)

    if bg_image_ready is None:
        bg_image_ready = np.zeros_like(frame_ref)

    # Configurar VSync para Qt (multiplataforma)
    try:
        import os

        # Configuración multiplataforma para VSync
        os.environ["QT_OPENGL"] = "angle"
        os.environ["QT_OPENGL_SYNC"] = "1"
        print("Configuración VSync multiplataforma aplicada")
    except Exception as e:
        print(f"Advertencia: No se pudo configurar VSync: {e}")

    print("Calentando motores SIMD...")
    dummy_f = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_m = np.zeros((100, 100), dtype=np.float32)
    _ = sigmoid(dummy_m, 12.0, 0.5)
    _ = optimized_blend(dummy_f, dummy_f, dummy_m)
    print("Sistemas listos.")

    previous_mask_stable = None
    last_raw_mask = None

    morph_kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.morph, args.morph))
        if args.morph > 0
        else None
    )
    use_guided = args.guided and HAS_GUIDED_FILTER

    fm = FrameManager(args.frame_skip, args.skip_non_essential)

    pipeline = None
    if args.multi_thread > 0:
        pipeline = ProcessingPipeline(num_workers=args.multi_thread)
        print(f"MULTI-THREADING ACTIVADO: {args.multi_thread} workers")
    else:
        print("Procesamiento single-threaded")

    fps_counter = {
        "frame_count": 0,
        "last_time": time.time(),
        "current_fps": 0,
        "fps_history": [],
    }

    print(f"SYSTEM ONLINE. FrameSkip: {args.frame_skip}")

    k_wrap = args.light_wrap if args.light_wrap % 2 == 1 else args.light_wrap + 1

    while True:
        frame = vs.read()
        if frame is None:
            break

        fm.update()
        do_effects = fm.should_run_effects()
        do_inference = fm.should_run_inference()

        if args.harmonize > 0 and do_effects and bg_stats:
            if pipeline:
                task_id = pipeline.submit_task(
                    color_transfer, (frame, bg_stats, args.harmonize)
                )
                frame_proc = safe_collect_result(pipeline, task_id, frame)
            else:
                frame_proc = color_transfer(frame, bg_stats, args.harmonize)
        else:
            frame_proc = frame

        work_w, work_h = (
            max(160, int(w_orig * args.scale)),
            max(90, int(h_orig * args.scale)),
        )

        if do_inference or last_raw_mask is None:
            frame_infer = cv2.resize(frame, (work_w, work_h))
            raw_mask = engine.process(frame_infer)
            last_raw_mask = raw_mask
        else:
            raw_mask = last_raw_mask

        if previous_mask_stable is None:
            previous_mask_stable = raw_mask

        mask_stable = (raw_mask * (1 - args.temporal)) + (
            previous_mask_stable * args.temporal
        )
        previous_mask_stable = mask_stable

        mask_sigmoid = sigmoid(
            mask_stable.astype(np.float32),
            np.float32(args.mask_contrast),
            np.float32(0.5),
        )

        if pipeline and morph_kernel is not None and do_effects:
            morph_task = pipeline.submit_task(
                morph_operations_task, (mask_sigmoid, morph_kernel)
            )
            mask_base = safe_collect_result(pipeline, morph_task, mask_sigmoid)
        elif morph_kernel is not None:
            mask_binary = (mask_sigmoid > 0.5).astype(np.uint8) * 255
            mask_morph = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, morph_kernel)
            mask_base = (
                cv2.morphologyEx(mask_morph, cv2.MORPH_OPEN, morph_kernel).astype(
                    np.float32
                )
                / 255.0
            )
        else:
            mask_base = mask_sigmoid

        mask_hd = cv2.resize(
            mask_base, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
        )

        if pipeline and use_guided and do_effects:
            guided_task = pipeline.submit_task(
                guided_filter_task, (frame_proc, mask_hd, args.blur)
            )
            mask_final = safe_collect_result(pipeline, guided_task, mask_hd)
        elif use_guided and do_effects:
            mask_final = cv2.ximgproc.guidedFilter(
                guide=frame_proc, src=mask_hd, radius=args.blur, eps=1e-6
            )
        else:
            k = args.blur if args.blur % 2 == 1 else args.blur + 1
            mask_final = (
                cv2.GaussianBlur(mask_hd, (k, k), 0) if args.blur > 0 else mask_hd
            )

        mask_final = np.clip(mask_final, 0, 1)

        if (
            pipeline
            and args.light_wrap > 0
            and do_effects
            and bg_blurred_light_wrap is not None
        ):
            light_wrap_task_id = pipeline.submit_task(
                light_wrap_task,
                (frame_proc, bg_blurred_light_wrap, mask_final, args.light_wrap),
            )
            frame_to_compose = safe_collect_result(
                pipeline, light_wrap_task_id, frame_proc
            )
        elif args.light_wrap > 0 and do_effects and bg_blurred_light_wrap is not None:
            frame_to_compose = apply_light_wrap(
                frame_proc, bg_blurred_light_wrap, mask_final, args.light_wrap
            )
        else:
            frame_to_compose = frame_proc

        final_image = optimized_blend(frame_to_compose, bg_image_ready, mask_final)

        fps_counter["frame_count"] += 1
        current_time = time.time()
        elapsed = current_time - fps_counter["last_time"]

        if elapsed >= 1.0:
            fps_counter["current_fps"] = fps_counter["frame_count"] / elapsed
            fps_counter["fps_history"].append(fps_counter["current_fps"])
            if len(fps_counter["fps_history"]) > 10:
                fps_counter["fps_history"].pop(0)
            fps_counter["frame_count"] = 0
            fps_counter["last_time"] = current_time

            if len(fps_counter["fps_history"]) % 5 == 0:
                avg_fps = sum(fps_counter["fps_history"]) / len(
                    fps_counter["fps_history"]
                )
                print(f"FPS: {fps_counter['current_fps']:.1f} (Avg: {avg_fps:.1f})")

        cv2.imshow("VBG", final_image)
        if cam_out:
            cam_out.send(final_image)
            cam_out.sleep_until_next_frame()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vs.stop()
    if cam_out:
        cam_out.close()
    engine.close()

    if pipeline:
        pipeline.shutdown()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
