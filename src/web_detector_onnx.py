"""
Web Detector ONNX - Deteccao de Cartas TCG via Web
Streaming MJPEG com Flask + ONNX Runtime

Funcionalidades:
  - Seletor de camera na interface web
  - Deteccao de cartas usando modelo ONNX
  - Streaming em tempo real com bounding boxes
  - Estatisticas de FPS e tempo de inferencia

Uso:
    python src/web_detector_onnx.py
    # Acesse http://localhost:5000 no navegador
"""

import cv2
import numpy as np
import os
import sys
import signal
import time
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from flask import Flask, Response, render_template, jsonify, request

# Tenta importar ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("AVISO: onnxruntime nao instalado. Execute: pip install onnxruntime")

# Diretorio base do projeto
BASE_DIR = Path(__file__).parent.parent


@dataclass
class DetectorConfig:
    """Configuracoes do detector ONNX"""
    model_path: str = "data/best_card_detect.onnx"
    model_input_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    class_names: Dict[int, str] = field(default_factory=lambda: {0: "carta_tcg"})

    # Camera
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    # Streaming
    jpeg_quality: int = 85
    frame_skip: int = 1

    # Display
    bbox_color: Tuple[int, int, int] = (0, 255, 0)
    bbox_thickness: int = 2
    font_scale: float = 0.6

    # Web
    host: str = "0.0.0.0"
    port: int = 5000


class CardDetectorONNX:
    """Detector de cartas usando ONNX Runtime"""

    def __init__(self, model_path: str, config: DetectorConfig):
        self.config = config
        self.class_names = config.class_names
        self.session = None
        self.input_name = None
        self.input_shape = None

        if not ONNX_AVAILABLE:
            print("ERRO: ONNX Runtime nao disponivel")
            return

        if not os.path.exists(model_path):
            print(f"ERRO: Modelo nao encontrado: {model_path}")
            return

        # Configura ONNX Runtime
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1

        # Carrega modelo
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        print(f"Modelo ONNX carregado: {model_path}")
        print(f"Input: {self.input_name}, shape: {self.input_shape}")

    def is_ready(self) -> bool:
        return self.session is not None

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Pre-processa frame para inferencia"""
        h, w = frame.shape[:2]
        target = self.config.model_input_size

        # Calcula escala mantendo aspect ratio
        scale = min(target / h, target / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Redimensiona
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Adiciona padding (letterbox)
        padded = np.full((target, target, 3), 114, dtype=np.uint8)
        pad_h = (target - new_h) // 2
        pad_w = (target - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Normaliza e converte para NCHW
        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)

        info = {
            'scale': scale,
            'pad_w': pad_w,
            'pad_h': pad_h,
            'orig_w': w,
            'orig_h': h
        }

        return blob, info

    def postprocess(self, output: np.ndarray, info: dict) -> List[dict]:
        """Pos-processa output do modelo"""
        predictions = output[0]

        # Ajusta formato se necessario [1, 5, 8400] -> [8400, 5]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        detections = []
        conf_thresh = self.config.confidence_threshold

        for pred in predictions:
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]

            max_score = np.max(class_scores)
            if max_score < conf_thresh:
                continue

            class_id = np.argmax(class_scores)

            # Converte para coordenadas originais
            scale = info['scale']
            pad_w = info['pad_w']
            pad_h = info['pad_h']

            x1 = int((x_center - width / 2 - pad_w) / scale)
            y1 = int((y_center - height / 2 - pad_h) / scale)
            x2 = int((x_center + width / 2 - pad_w) / scale)
            y2 = int((y_center + height / 2 - pad_h) / scale)

            # Clipa as dimensoes originais
            x1 = max(0, min(x1, info['orig_w']))
            y1 = max(0, min(y1, info['orig_h']))
            x2 = max(0, min(x2, info['orig_w']))
            y2 = max(0, min(y2, info['orig_h']))

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(max_score),
                'class_id': int(class_id),
                'class_name': self.class_names.get(class_id, f"class_{class_id}")
            })

        # Aplica NMS
        if detections:
            detections = self._nms(detections)

        return detections

    def _nms(self, detections: List[dict]) -> List[dict]:
        """Non-Maximum Suppression"""
        if not detections:
            return []

        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [
                d for d in detections
                if self._iou(best['bbox'], d['bbox']) < self.config.iou_threshold
            ]

        return keep

    def _iou(self, box1: List[int], box2: List[int]) -> float:
        """Calcula Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter + 1e-6)

    def detect(self, frame: np.ndarray) -> Tuple[List[dict], float]:
        """Executa deteccao no frame"""
        if not self.is_ready():
            return [], 0.0

        start = time.perf_counter()

        # Converte BGR para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pre-processa
        blob, info = self.preprocess(rgb_frame)

        # Inferencia
        outputs = self.session.run(None, {self.input_name: blob})

        # Pos-processa
        detections = self.postprocess(outputs[0], info)

        inference_time = (time.perf_counter() - start) * 1000

        return detections, inference_time


class CameraManager:
    """Gerencia multiplas cameras e permite trocar entre elas"""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.cap = None
        self.current_camera_id = 0
        self.lock = threading.Lock()
        self.available_cameras = []

    def list_cameras(self) -> List[Dict]:
        """Lista todas as cameras disponiveis"""
        cameras = []

        # Testa cameras de 0 a 9
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Tenta obter informacoes
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append({
                    'id': i,
                    'name': f"Camera {i}",
                    'resolution': f"{w}x{h}",
                    'active': i == self.current_camera_id
                })
                cap.release()

        self.available_cameras = cameras
        return cameras

    def select_camera(self, camera_id: int) -> bool:
        """Seleciona uma camera pelo ID"""
        with self.lock:
            # Libera camera atual
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            # Abre nova camera
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print(f"Erro ao abrir camera {camera_id}")
                return False

            # Configura camera
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Descarta primeiros frames
            for _ in range(5):
                self.cap.read()

            self.current_camera_id = camera_id

            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera {camera_id} selecionada: {actual_w}x{actual_h}")

            return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Le um frame da camera atual"""
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return False, None
            return self.cap.read()

    def release(self):
        """Libera recursos"""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None


class CameraThread:
    """Thread para captura continua de frames"""

    def __init__(self, camera_manager: CameraManager):
        self.camera_manager = camera_manager
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        """Inicia a thread de captura"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        """Loop de captura em thread separada"""
        while self.running:
            ret, frame = self.camera_manager.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> Optional[np.ndarray]:
        """Retorna o ultimo frame capturado (thread-safe)"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Para a thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class DetectorThread:
    """Thread para deteccao ONNX em paralelo"""

    def __init__(self, detector: CardDetectorONNX):
        self.detector = detector

        self.detections = []
        self.inference_time = 0.0
        self.fps = 0.0
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

        self.input_frame = None
        self.input_lock = threading.Lock()
        self.new_frame_event = threading.Event()

    def start(self):
        """Inicia a thread de deteccao"""
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        return self

    def submit_frame(self, frame: np.ndarray):
        """Envia um frame para processamento"""
        with self.input_lock:
            self.input_frame = frame
        self.new_frame_event.set()

    def _detection_loop(self):
        """Loop de deteccao em thread separada"""
        fps_counter = 0
        fps_start = time.time()

        while self.running:
            # Espera novo frame
            if not self.new_frame_event.wait(timeout=0.1):
                continue
            self.new_frame_event.clear()

            # Pega o frame
            with self.input_lock:
                frame = self.input_frame
                self.input_frame = None

            if frame is None:
                continue

            # Executa deteccao
            detections, inference_time = self.detector.detect(frame)

            # Atualiza resultado
            with self.lock:
                self.detections = detections
                self.inference_time = inference_time

            # Calcula FPS
            fps_counter += 1
            if fps_counter >= 10:
                elapsed = time.time() - fps_start
                if elapsed > 0:
                    with self.lock:
                        self.fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

    def get_results(self) -> Tuple[List[dict], float, float]:
        """Retorna deteccoes, tempo de inferencia e FPS"""
        with self.lock:
            return list(self.detections), self.inference_time, self.fps

    def stop(self):
        """Para a thread"""
        self.running = False
        self.new_frame_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


def draw_detections(frame: np.ndarray, detections: List[dict], config: DetectorConfig) -> np.ndarray:
    """Desenha deteccoes no frame"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_name = det['class_name']

        # Desenha bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.bbox_color, config.bbox_thickness)

        # Label
        label = f"{class_name}: {conf:.2f}"

        # Background do texto
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, config.font_scale, 2
        )

        cv2.rectangle(
            frame,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 5, y1),
            config.bbox_color,
            -1
        )

        cv2.putText(
            frame, label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.font_scale,
            (0, 0, 0),
            2
        )

    return frame


def draw_stats(frame: np.ndarray, fps: float, inference_ms: float, num_detections: int) -> np.ndarray:
    """Desenha estatisticas no frame"""
    stats = [
        f"FPS: {fps:.1f}",
        f"Inference: {inference_ms:.1f}ms",
        f"Detections: {num_detections}"
    ]

    y = 25
    for stat in stats:
        cv2.putText(
            frame, stat,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        y += 25

    return frame


# --- Configuracao global ---
config = DetectorConfig()

# Resolve caminho do modelo
model_path = BASE_DIR / config.model_path
if not model_path.exists():
    print(f"AVISO: Modelo nao encontrado em {model_path}")
    model_path = None

# Inicializa componentes
camera_manager = CameraManager(config)
detector = CardDetectorONNX(str(model_path), config) if model_path else None
detector_thread = None
camera_thread = None

# Estatisticas globais
stats = {
    'fps': 0.0,
    'inference_time': 0.0,
    'detections': 0,
    'camera_id': 0
}
stats_lock = threading.Lock()


def init_threads():
    """Inicializa threads de captura e deteccao"""
    global camera_thread, detector_thread

    # Inicia com camera 0
    if camera_manager.select_camera(0):
        camera_thread = CameraThread(camera_manager).start()

        if detector and detector.is_ready():
            detector_thread = DetectorThread(detector).start()


def generate_frames():
    """Gerador de frames para streaming MJPEG"""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
    frame_skip = config.frame_skip
    frame_count = 0

    # Controle de taxa
    target_fps = 30
    frame_time = 1.0 / target_fps
    last_frame_time = time.time()

    # FPS do streaming
    stream_fps = 0.0
    stream_fps_counter = 0
    stream_fps_start = time.time()

    while True:
        # Controle de taxa
        now = time.time()
        elapsed = now - last_frame_time
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)
        last_frame_time = time.time()

        # Verifica se camera esta ativa
        if camera_thread is None:
            # Frame preto com mensagem
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "Selecione uma camera", (480, 360),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            frame = camera_thread.read()
            if frame is None:
                continue

            frame_count += 1

            # Envia para deteccao
            if detector_thread and frame_count % frame_skip == 0:
                detector_thread.submit_frame(frame.copy())

            # Pega resultados
            detections, inference_time, detect_fps = [], 0.0, 0.0
            if detector_thread:
                detections, inference_time, detect_fps = detector_thread.get_results()

            # Calcula FPS do streaming
            stream_fps_counter += 1
            if stream_fps_counter >= 30:
                stream_elapsed = time.time() - stream_fps_start
                if stream_elapsed > 0:
                    stream_fps = stream_fps_counter / stream_elapsed
                stream_fps_counter = 0
                stream_fps_start = time.time()

            # Desenha no frame
            frame = draw_detections(frame, detections, config)
            frame = draw_stats(frame, stream_fps, inference_time, len(detections))

            # Atualiza estatisticas globais
            with stats_lock:
                stats['fps'] = stream_fps
                stats['inference_time'] = inference_time
                stats['detections'] = len(detections)

        # Codifica e envia
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# --- Flask App ---

app = Flask(__name__)


@app.route('/')
def index():
    """Pagina principal"""
    return render_template('index_onnx.html')


@app.route('/video_feed')
def video_feed():
    """Endpoint de streaming MJPEG"""
    response = Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@app.route('/cameras')
def get_cameras():
    """Retorna lista de cameras disponiveis"""
    cameras = camera_manager.list_cameras()
    return jsonify(cameras)


@app.route('/camera/<int:camera_id>', methods=['POST'])
def set_camera(camera_id: int):
    """Troca a camera ativa"""
    global camera_thread

    # Para thread atual
    if camera_thread:
        camera_thread.stop()
        camera_thread = None

    # Seleciona nova camera
    success = camera_manager.select_camera(camera_id)
    if success:
        camera_thread = CameraThread(camera_manager).start()
        with stats_lock:
            stats['camera_id'] = camera_id

    return jsonify({'success': success, 'camera_id': camera_id})


@app.route('/stats')
def get_stats():
    """Retorna estatisticas atuais"""
    with stats_lock:
        return jsonify(stats)


def cleanup():
    """Libera recursos"""
    if detector_thread:
        detector_thread.stop()
    if camera_thread:
        camera_thread.stop()
    camera_manager.release()
    print("Recursos liberados")


def main():
    print("=" * 60)
    print("WEB DETECTOR ONNX - Deteccao de Cartas TCG")
    print("=" * 60)

    if not ONNX_AVAILABLE:
        print("\nERRO: ONNX Runtime nao instalado!")
        print("Execute: pip install onnxruntime")
        return 1

    if detector is None or not detector.is_ready():
        print("\nAVISO: Detector ONNX nao inicializado")
        print(f"Verifique se o modelo existe em: {config.model_path}")
    else:
        print(f"Modelo: {config.model_path}")

    print(f"Confianca minima: {config.confidence_threshold}")
    print(f"Resolucao camera: {config.camera_width}x{config.camera_height}")
    print("=" * 60)

    # Inicializa threads
    init_threads()

    # Configura signal handler
    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))

    print(f"\nServidor iniciado em http://{config.host}:{config.port}")
    print("Acesse de qualquer dispositivo na rede local")
    print("Pressione Ctrl+C para encerrar\n")

    try:
        app.run(host=config.host, port=config.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario")
    finally:
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
