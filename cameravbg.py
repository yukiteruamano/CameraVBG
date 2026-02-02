import argparse
import os
import queue
import threading
import time

import cv2
import numpy as np

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
        """Devuelve True si se debe ejecutar la IA en este frame"""
        if self.frame_skip <= 0:
            return True
        # Si frame_skip es 2: Ejecuta en 0, 2, 4... (Salta 1)
        # Si frame_skip es 3: Ejecuta en 0, 3, 6... (Salta 2)
        return (self.frame_counter % self.frame_skip) == 0

    def should_run_effects(self):
        """Devuelve True si se deben ejecutar efectos pesados (LightWrap, etc)"""
        if not self.skip_non_essential:
            return True
        # Si estamos saltando inferencia, también saltamos efectos pesados
        # para maximizar FPS en los frames intermedios
        return self.should_run_inference()


class ProcessingPipeline:
    """Pipeline de procesamiento multi-hilo para optimización de rendimiento"""

    def __init__(self, num_workers=2):
        self.task_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.workers = []
        self.shutdown_flag = False
        self.num_workers = num_workers
        self._setup_workers()

    def _setup_workers(self):
        """Inicializa los workers del thread pool"""
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
        """Bucle principal de los workers"""
        while not self.shutdown_flag:
            try:
                task_func, task_args, task_id = self.task_queue.get(timeout=0.1)

                # Ejecutar la tarea
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
        """Envía una tarea al pipeline"""
        if task_id is None:
            task_id = f"task_{time.time()}_{len(self.workers)}"

        try:
            self.task_queue.put((task_func, task_args, task_id), timeout=0.01)
            return task_id
        except queue.Full:
            # Si la cola está llena, ejecutamos en el hilo principal 
            # para evitar bloqueos
            return self._execute_in_main_thread(task_func, task_args, task_id)

    def _execute_in_main_thread(self, task_func, task_args, task_id):
        """Ejecuta tarea en hilo principal si el pipeline está saturado"""
        try:
            result = task_func(*task_args)
            self.result_queue.put((task_id, result))
            return task_id
        except Exception as e:
            print(f"Main thread execution error: {e}")
            self.result_queue.put((task_id, None))
            return task_id

    def collect_result(self, task_id, timeout=0.5):
        """Recoge el resultado de una tarea"""
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if not self.result_queue.empty():
                    result_task_id, result = self.result_queue.get()
                    if result_task_id == task_id:
                        return result
                    else:
                        # Si no es el resultado que buscamos, lo volvemos a poner
                        # en la cola
                        self.result_queue.put((result_task_id, result))
                time.sleep(0.01)
            return None
        except Exception as e:
            print(f"Error collecting result: {e}")
            return None

    def shutdown(self):
        """Apaga el pipeline gracefulmente"""
        self.shutdown_flag = True
        # Esperar a que los workers terminen
        for worker in self.workers:
            worker.join(timeout=1.0)

    def get_worker_status(self):
        """Información de depuración sobre los workers"""
        return {
            "active_workers": len([w for w in self.workers if w.is_alive()]),
            "task_queue_size": self.task_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
        }


# --- FUNCIONES AUXILIARES ---


def color_transfer(source, target_stats, intensity=0.3):
    """
    source: Frame de la cámara (uint8 BGR)
    target_stats: Tupla (mean, std) pre-calculada del fondo en espacio LAB
    """
    (l_mean_tar, l_std_tar) = target_stats

    # Convertimos solo el frame actual a LAB
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    (l_mean_src, l_std_src) = cv2.meanStdDev(source_lab)

    l, a, b = cv2.split(source_lab)

    def scale_channel(ch, m_src, s_src, m_tar, s_tar):
        # Evitamos división por cero
        s_src = max(s_src, 1e-5)
        new_ch = (ch - m_src) * (s_tar / s_src) + m_tar
        return np.clip(new_ch, 0, 255)

    # Aplicamos el escalado solo a los canales A y B (el color)
    # El canal L (luminosidad) lo dejamos intacto o lo escalamos suavemente si prefieres
    a_new = scale_channel(a, l_mean_src[1], l_std_src[1], l_mean_tar[1], l_std_tar[1])
    b_new = scale_channel(b, l_mean_src[2], l_std_src[2], l_mean_tar[2], l_std_tar[2])

    # Mezclamos con el original según la intensidad
    a_final = cv2.addWeighted(a_new.astype(np.float32), intensity, a, 1 - intensity, 0)
    b_final = cv2.addWeighted(b_new.astype(np.float32), intensity, b, 1 - intensity, 0)

    merged = cv2.merge([l, a_final, b_final])
    return cv2.cvtColor(merged.astype("uint8"), cv2.COLOR_LAB2BGR)


def sigmoid(x, slope, shift):
    return 1 / (1 + np.exp(-slope * (x - shift)))


def wait_for_camera_ready(vs, max_attempts=10):
    time.sleep(0.5)
    attempt = 0

    for attempt in range(max_attempts):
        frame = vs.read()
        if frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            if h > 0 and w > 0:
                print(f"Cámara lista: {w}x{h}")
                return True
        time.sleep(0.2)
        attempt = attempt + 1
    return False


def apply_light_wrap(foreground, bg_blurred, mask, radius=20):
    mask_inv = 1.0 - mask
    k = radius if radius % 2 == 1 else radius + 1

    # El desenfoque de la máscara SÍ debe ser interno porque la máscara cambia
    mask_inv_blurred = cv2.GaussianBlur(mask_inv, (k, k), 0)
    wrap_map = mask * mask_inv_blurred

    if len(wrap_map.shape) == 2:
        wrap_map = wrap_map[:, :, np.newaxis]

    # Usamos bg_blurred que ya viene procesado de afuera
    return (
        foreground.astype(np.float32) * (1.0 - wrap_map)
        + bg_blurred.astype(np.float32) * wrap_map
    ).astype(np.uint8)


def guided_filter_task(frame_proc, mask_hd, blur_radius):
    """Tarea paralela para guided filter con manejo de errores"""
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
    """Tarea paralela para light wrap con manejo de errores"""
    try:
        if radius > 0 and bg_blurred is not None:
            return apply_light_wrap(frame_proc, bg_blurred, mask_final, radius)
        return frame_proc
    except Exception as e:
        print(f"Light wrap error: {e}")
        return frame_proc


def morph_operations_task(mask_sigmoid, morph_kernel):
    """Tarea paralela para operaciones morfológicas con manejo de errores"""
    try:
        if morph_kernel is not None:
            _, mb = cv2.threshold(mask_sigmoid, 0.5, 1.0, cv2.THRESH_BINARY)
            mask_morph = cv2.morphologyEx(
                (mb * 255).astype(np.uint8), cv2.MORPH_CLOSE, morph_kernel
            )
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
    help="Procesamiento multi-hilo (0=desactivado, 2=2 workers, 4=4 workers)",
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

    # Obtener dimensiones reales
    frame_ref = vs.read()
    h_orig, w_orig = frame_ref.shape[:2]

    # VCam
    cam_out = None
    if args.virtual and HAS_VIRTUAL_CAM:
        cam_out = pyvirtualcam.Camera(
            width=w_orig, height=h_orig, fps=30, fmt=pyvirtualcam.PixelFormat.BGR
        )
        print("--> VIRTUAL CAM ACTIVA")

    # --- PREPARACIÓN DE FONDO OPTIMIZADA ---
    bg_image_ready = None
    bg_blurred_light_wrap = None
    bg_stats = None

    if args.bg_img:
        bg_raw = cv2.imread(args.bg_img)
        if bg_raw is not None:
            bg_image_ready = cv2.resize(bg_raw, (w_orig, h_orig))

            # 1. Pre-calcular desenfoque para Light Wrap
            k_wrap = (
                args.light_wrap if args.light_wrap % 2 == 1 else args.light_wrap + 1
            )
            if k_wrap > 0:
                bg_blurred_light_wrap = cv2.GaussianBlur(
                    bg_image_ready, (k_wrap, k_wrap), 0
                )

            # 2. Pre-calcular estadísticas para Harmonize (Color Transfer)
            if args.harmonize > 0:
                bg_small = cv2.resize(bg_image_ready, (200, 200))
                bg_lab = cv2.cvtColor(bg_small, cv2.COLOR_BGR2LAB).astype("float32")
                bg_stats = cv2.meanStdDev(bg_lab)  # Guarda (mean, std)

    if bg_image_ready is None:
        bg_image_ready = np.zeros_like(frame_ref)

    # Variables de estado
    previous_mask_stable = None
    last_raw_mask = None  # Para frame skipping

    morph_kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.morph, args.morph))
        if args.morph > 0
        else None
    )
    use_guided = args.guided and HAS_GUIDED_FILTER

    # Manager
    fm = FrameManager(args.frame_skip, args.skip_non_essential)

    # Inicializar pipeline multi-hilo si está activado
    pipeline = None
    if args.multi_thread > 0:
        pipeline = ProcessingPipeline(num_workers=args.multi_thread)
        print(f"MULTI-THREADING ACTIVADO: {args.multi_thread} workers")
    else:
        print("Procesamiento single-threaded")

    # Contador de FPS
    fps_counter = {
        "frame_count": 0,
        "last_time": time.time(),
        "current_fps": 0,
        "fps_history": [],
    }

    print(f"SYSTEM ONLINE. FrameSkip: {args.frame_skip}")

    while True:
        frame = vs.read()
        if frame is None:
            break

        fm.update()
        do_effects = fm.should_run_effects()
        do_inference = fm.should_run_inference()

        # 1. Armonización (Solo si toca efectos o si es frame completo)
        # Nota: La armonización depende del frame actual, si la cámara se mueve mucho
        # y saltamos esto, se verá raro. Pero para rendimiento extremo, se salta.
        if args.harmonize > 0 and do_effects and bg_stats:
            if pipeline:
                # Procesamiento paralelo
                task_id = pipeline.submit_task(
                    color_transfer, (frame, bg_stats, args.harmonize)
                )
                result = pipeline.collect_result(task_id)
                frame_proc = result if result is not None else frame
            else:
                # Procesamiento single-threaded
                frame_proc = color_transfer(frame, bg_stats, args.harmonize)
        else:
            frame_proc = frame

        # 2. Inferencia (CONTROLADA POR FRAME SKIP)
        work_w, work_h = (
            max(160, int(w_orig * args.scale)),
            max(90, int(h_orig * args.scale)),
        )

        if do_inference or last_raw_mask is None:
            # Solo ejecutamos la IA si toca
            frame_infer = cv2.resize(frame, (work_w, work_h))
            raw_mask = engine.process(frame_infer)
            last_raw_mask = raw_mask  # Guardamos para el siguiente skip
        else:
            # Reutilizamos la máscara del frame anterior
            raw_mask = last_raw_mask

        # 3. Estabilidad Temporal
        if previous_mask_stable is None:
            previous_mask_stable = raw_mask

        # Incluso si saltamos inferencia, aplicamos el suavizado temporal
        # para que la transición no sea tan brusca si el objeto se movió
        mask_stable = (raw_mask * (1 - args.temporal)) + (
            previous_mask_stable * args.temporal
        )
        previous_mask_stable = mask_stable

        # 4. Refinado (con procesamiento paralelo)
        mask_sigmoid = sigmoid(mask_stable, args.mask_contrast, 0.5)

        if pipeline and morph_kernel is not None and do_effects:
            # Procesamiento paralelo de operaciones morfológicas
            morph_task = pipeline.submit_task(
                morph_operations_task, (mask_sigmoid, morph_kernel)
            )
            morph_result = pipeline.collect_result(morph_task)
            mask_base = morph_result if morph_result is not None else mask_sigmoid
        elif morph_kernel is not None:
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

        # OPTIMIZACIÓN: Upscaling LINEAR (Mucho más rápido que LANCZOS4)
        mask_hd = cv2.resize(
            mask_base, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
        )

        if pipeline and use_guided and do_effects:  # Guided filter paralelo
            guided_task = pipeline.submit_task(
                guided_filter_task, (frame_proc, mask_hd, args.blur)
            )
            mask_final = pipeline.collect_result(guided_task) or mask_hd
        elif use_guided and do_effects:  # Guided filter single-threaded
            mask_final = cv2.ximgproc.guidedFilter(
                guide=frame_proc, src=mask_hd, radius=args.blur, eps=1e-6
            )
        else:
            k = args.blur if args.blur % 2 == 1 else args.blur + 1
            mask_final = (
                cv2.GaussianBlur(mask_hd, (k, k), 0) if args.blur > 0 else mask_hd
            )

        # 5. Composición
        mask_final = np.clip(mask_final, 0, 1)
        mask_3d = np.stack((mask_final,) * 3, axis=-1)

        # Light Wrap controlado por el manager (con procesamiento paralelo)
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
            frame_to_compose = pipeline.collect_result(light_wrap_task_id) or frame_proc
        elif args.light_wrap > 0 and do_effects and bg_blurred_light_wrap is not None:
            frame_to_compose = apply_light_wrap(
                frame_proc, bg_blurred_light_wrap, mask_final, args.light_wrap
            )
        else:
            frame_to_compose = frame_proc

        # Usamos bg_image_ready que YA tiene el tamaño correcto (sin resize en bucle)
        #final_image = cv2.multiply(
        #    frame_to_compose, mask_3d, dtype=cv2.CV_8U
        #) + cv2.multiply(bg_image_ready, 1.0 - mask_3d, dtype=cv2.CV_8U)
        
        final_image = (frame_to_compose * mask_3d + bg_image_ready * 
                       (1.0 - mask_3d)).astype(np.uint8)

        # Actualizar contador de FPS
        fps_counter["frame_count"] += 1
        current_time = time.time()
        elapsed = current_time - fps_counter["last_time"]

        if elapsed >= 1.0:  # Actualizar FPS cada segundo
            fps_counter["current_fps"] = fps_counter["frame_count"] / elapsed
            fps_counter["fps_history"].append(fps_counter["current_fps"])
            if (
                len(fps_counter["fps_history"]) > 10
            ):  # Mantener solo los últimos 10 valores
                fps_counter["fps_history"].pop(0)
            fps_counter["frame_count"] = 0
            fps_counter["last_time"] = current_time

            # Mostrar FPS en consola (cada 5 segundos para no saturar)
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

    # Apagar pipeline multi-hilo si está activo
    if pipeline:
        pipeline.shutdown()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
