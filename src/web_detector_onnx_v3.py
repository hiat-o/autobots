"""
Web Detector ONNX v3 - Deteccao de Cartas TCG com Camera CSI (Raspberry Pi 5)
Streaming MJPEG com Flask + ONNX Runtime + Picamera2

Funcionalidades:
  - Suporte a camera CSI IMX219 via Picamera2 (Raspberry Pi)
  - Fallback para webcam USB via OpenCV
  - Deteccao de cartas usando modelo ONNX
  - Stream em 640x480 para FPS alto
  - Captura em alta resolucao (ate 3280x2464)
  - Botao para capturar ROI da carta detectada

Uso:
    python src/web_detector_onnx_v3.py
    # Acesse http://<ip-raspberry>:5000 no navegador
"""

import cv2
import numpy as np
import os
import sys
import signal
import time
import threading
import platform
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, request, send_from_directory

# Detecta se esta rodando em Raspberry Pi
IS_RASPBERRY_PI = platform.machine() in ('aarch64', 'armv7l') and os.path.exists('/proc/device-tree/model')

# Tenta importar Picamera2 (apenas Raspberry Pi)
PICAMERA2_AVAILABLE = False
if IS_RASPBERRY_PI:
    try:
        from picamera2 import Picamera2
        from libcamera import controls
        PICAMERA2_AVAILABLE = True
        print("Picamera2 disponivel - usando camera CSI")
    except ImportError:
        print("AVISO: Picamera2 nao instalado. Execute: sudo apt install python3-picamera2")

# Tenta importar ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("AVISO: onnxruntime nao instalado. Execute: pip install onnxruntime")

# Diretorio base do projeto
BASE_DIR = Path(__file__).parent.parent

# Diretorio para salvar capturas
CAPTURES_DIR = BASE_DIR / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)


# ============================================================================
# PRE-PROCESSAMENTO DE IMAGEM
# ============================================================================

def denoise_image(image: np.ndarray, d: int = 7, sigma_color: int = 50, sigma_space: int = 50) -> np.ndarray:
    """Remove ruido da imagem usando Bilateral Filter."""
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def sharpen_image(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Aplica sharpening usando Unsharp Mask."""
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return sharpened


def enhance_contrast_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Melhora contraste usando CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def preprocess_card_image(image: np.ndarray,
                          denoise: bool = True,
                          sharpen: bool = True,
                          enhance_contrast: bool = True,
                          denoise_d: int = 7,
                          sharpen_amount: float = 0.3) -> np.ndarray:
    """Pipeline de pre-processamento para imagem de carta TCG."""
    result = image.copy()

    if denoise:
        result = denoise_image(result, d=denoise_d)

    if enhance_contrast:
        result = enhance_contrast_clahe(result, clip_limit=2.0)

    if sharpen:
        result = sharpen_image(result, amount=sharpen_amount)

    return result


@dataclass
class DetectorConfig:
    """Configuracoes do detector ONNX"""
    model_path: str = "data/best_card_detect.onnx"
    model_input_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    class_names: Dict[int, str] = field(default_factory=lambda: {0: "carta_tcg"})

    # Camera CSI - Captura em alta resolucao
    csi_capture_width: int = 1920
    csi_capture_height: int = 1080
    csi_max_width: int = 3280  # Resolucao maxima IMX219
    csi_max_height: int = 2464

    # Camera USB - Fallback
    usb_capture_width: int = 1280
    usb_capture_height: int = 720
    camera_fps: int = 30

    # Stream - Resolucao reduzida para FPS alto
    stream_width: int = 640
    stream_height: int = 480

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

        scale = min(target / h, target / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((target, target, 3), 114, dtype=np.uint8)
        pad_h = (target - new_h) // 2
        pad_w = (target - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

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

            scale = info['scale']
            pad_w = info['pad_w']
            pad_h = info['pad_h']

            x1 = int((x_center - width / 2 - pad_w) / scale)
            y1 = int((y_center - height / 2 - pad_h) / scale)
            x2 = int((x_center + width / 2 - pad_w) / scale)
            y2 = int((y_center + height / 2 - pad_h) / scale)

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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blob, info = self.preprocess(rgb_frame)
        outputs = self.session.run(None, {self.input_name: blob})
        detections = self.postprocess(outputs[0], info)

        inference_time = (time.perf_counter() - start) * 1000

        return detections, inference_time


# ============================================================================
# CAMERA CSI (PICAMERA2) - Raspberry Pi
# ============================================================================

class PiCameraManager:
    """Gerencia camera CSI via Picamera2 no Raspberry Pi"""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.picam2 = None
        self.lock = threading.Lock()
        self.is_running = False

        # Configuracoes atuais da camera
        self.current_config = {
            'exposure_time': 33000,  # microsegundos (~30fps)
            'analogue_gain': 8.0,
            'brightness': 0.0,
            'contrast': 1.0,
            'saturation': 1.0,
            'sharpness': 1.0
        }

    def start(self) -> bool:
        """Inicia a camera CSI"""
        if not PICAMERA2_AVAILABLE:
            print("Picamera2 nao disponivel")
            return False

        try:
            self.picam2 = Picamera2()

            # Configura para stream (resolucao menor) + captura (resolucao alta)
            # Usa BGR888 para compatibilidade direta com OpenCV (sem conversao)
            preview_config = self.picam2.create_preview_configuration(
                main={"size": (self.config.csi_capture_width, self.config.csi_capture_height), "format": "BGR888"},
                lores={"size": (self.config.stream_width, self.config.stream_height), "format": "BGR888"},
                buffer_count=4
            )
            self.picam2.configure(preview_config)

            # Configura controles manuais
            self.picam2.set_controls({
                "ExposureTime": self.current_config['exposure_time'],
                "AnalogueGain": self.current_config['analogue_gain'],
                "Brightness": self.current_config['brightness'],
                "Contrast": self.current_config['contrast'],
                "Saturation": self.current_config['saturation'],
                "Sharpness": self.current_config['sharpness'],
                "AeEnable": False,  # Desabilita auto-exposure
                "AwbEnable": True,  # Mantem auto white balance
            })

            self.picam2.start()
            self.is_running = True

            # Aguarda estabilizar
            time.sleep(0.5)

            print(f"Camera CSI iniciada: {self.config.csi_capture_width}x{self.config.csi_capture_height}")
            return True

        except Exception as e:
            print(f"Erro ao iniciar camera CSI: {e}")
            return False

    def read_stream(self) -> Optional[np.ndarray]:
        """Le frame em resolucao de stream (lores)"""
        if not self.is_running or self.picam2 is None:
            return None

        try:
            with self.lock:
                # Captura frame de baixa resolucao (ja em BGR888)
                return self.picam2.capture_array("lores")
        except Exception as e:
            print(f"Erro ao ler frame stream: {e}")
            return None

    def read_hd(self) -> Optional[np.ndarray]:
        """Le frame em alta resolucao (main)"""
        if not self.is_running or self.picam2 is None:
            return None

        try:
            with self.lock:
                # Captura frame HD (ja em BGR888)
                return self.picam2.capture_array("main")
        except Exception as e:
            print(f"Erro ao ler frame HD: {e}")
            return None

    def capture_full_resolution(self) -> Optional[np.ndarray]:
        """Captura em resolucao maxima (3280x2464)"""
        if not self.is_running or self.picam2 is None:
            return None

        try:
            with self.lock:
                # Para o stream atual
                self.picam2.stop()

                # Reconfigura para resolucao maxima (BGR888 para OpenCV)
                still_config = self.picam2.create_still_configuration(
                    main={"size": (self.config.csi_max_width, self.config.csi_max_height), "format": "BGR888"}
                )
                self.picam2.configure(still_config)
                self.picam2.start()

                # Captura
                time.sleep(0.2)  # Aguarda estabilizar
                frame = self.picam2.capture_array("main")

                # Restaura configuracao de stream
                self.picam2.stop()
                preview_config = self.picam2.create_preview_configuration(
                    main={"size": (self.config.csi_capture_width, self.config.csi_capture_height), "format": "BGR888"},
                    lores={"size": (self.config.stream_width, self.config.stream_height), "format": "BGR888"},
                    buffer_count=4
                )
                self.picam2.configure(preview_config)
                self.picam2.start()

                return frame  # Ja em BGR888

        except Exception as e:
            print(f"Erro ao capturar full resolution: {e}")
            return None

    def set_exposure(self, exposure_us: int):
        """Define tempo de exposicao em microsegundos"""
        if self.picam2 and self.is_running:
            self.current_config['exposure_time'] = exposure_us
            self.picam2.set_controls({"ExposureTime": exposure_us})

    def set_gain(self, gain: float):
        """Define ganho analogico"""
        if self.picam2 and self.is_running:
            self.current_config['analogue_gain'] = gain
            self.picam2.set_controls({"AnalogueGain": gain})

    def set_brightness(self, brightness: float):
        """Define brilho (-1.0 a 1.0)"""
        if self.picam2 and self.is_running:
            self.current_config['brightness'] = brightness
            self.picam2.set_controls({"Brightness": brightness})

    def set_contrast(self, contrast: float):
        """Define contraste (0.0 a 2.0)"""
        if self.picam2 and self.is_running:
            self.current_config['contrast'] = contrast
            self.picam2.set_controls({"Contrast": contrast})

    def set_sharpness(self, sharpness: float):
        """Define nitidez (0.0 a 2.0)"""
        if self.picam2 and self.is_running:
            self.current_config['sharpness'] = sharpness
            self.picam2.set_controls({"Sharpness": sharpness})

    def get_config(self) -> dict:
        """Retorna configuracao atual"""
        return self.current_config.copy()

    def stop(self):
        """Para a camera"""
        if self.picam2 and self.is_running:
            self.picam2.stop()
            self.picam2.close()
            self.is_running = False
            print("Camera CSI parada")


# ============================================================================
# CAMERA USB (OPENCV) - Fallback
# ============================================================================

class USBCameraManager:
    """Gerencia webcam USB via OpenCV (fallback)"""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.cap = None
        self.current_camera_id = 0
        self.lock = threading.Lock()

    def list_cameras(self) -> List[Dict]:
        """Lista cameras USB disponiveis"""
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append({
                    'id': i,
                    'name': f"USB Camera {i}",
                    'resolution': f"{w}x{h}",
                    'type': 'usb',
                    'active': i == self.current_camera_id
                })
                cap.release()
        return cameras

    def start(self, camera_id: int = 0) -> bool:
        """Inicia camera USB"""
        with self.lock:
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print(f"Erro ao abrir camera USB {camera_id}")
                return False

            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.usb_capture_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.usb_capture_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            for _ in range(5):
                self.cap.read()

            self.current_camera_id = camera_id
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera USB {camera_id} iniciada: {actual_w}x{actual_h}")
            return True

    def read_stream(self) -> Optional[np.ndarray]:
        """Le frame para stream"""
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return None
            ret, frame = self.cap.read()
            if ret:
                return cv2.resize(frame, (self.config.stream_width, self.config.stream_height))
            return None

    def read_hd(self) -> Optional[np.ndarray]:
        """Le frame em HD"""
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return None
            ret, frame = self.cap.read()
            return frame if ret else None

    def stop(self):
        """Para a camera"""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None


# ============================================================================
# CAMERA MANAGER UNIFICADO
# ============================================================================

class UnifiedCameraManager:
    """Gerencia camera CSI ou USB de forma unificada"""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.pi_camera = None
        self.usb_camera = None
        self.active_camera = None  # 'csi' ou 'usb'
        self.lock = threading.Lock()

        # Inicializa gerenciadores
        if PICAMERA2_AVAILABLE:
            self.pi_camera = PiCameraManager(config)
        self.usb_camera = USBCameraManager(config)

    def list_cameras(self) -> List[Dict]:
        """Lista todas as cameras disponiveis"""
        cameras = []

        # Camera CSI (se disponivel)
        if PICAMERA2_AVAILABLE:
            cameras.append({
                'id': 'csi',
                'name': 'Camera CSI (IMX219)',
                'resolution': f"{self.config.csi_capture_width}x{self.config.csi_capture_height}",
                'type': 'csi',
                'active': self.active_camera == 'csi'
            })

        # Cameras USB
        usb_cameras = self.usb_camera.list_cameras()
        for cam in usb_cameras:
            cam['active'] = self.active_camera == f"usb_{cam['id']}"
        cameras.extend(usb_cameras)

        return cameras

    def select_camera(self, camera_id) -> bool:
        """Seleciona camera por ID"""
        with self.lock:
            # Para camera atual
            self.stop_current()

            if camera_id == 'csi' and self.pi_camera:
                if self.pi_camera.start():
                    self.active_camera = 'csi'
                    return True
            elif isinstance(camera_id, int) or camera_id.startswith('usb_'):
                usb_id = int(camera_id.replace('usb_', '')) if isinstance(camera_id, str) else camera_id
                if self.usb_camera.start(usb_id):
                    self.active_camera = f"usb_{usb_id}"
                    return True

            return False

    def stop_current(self):
        """Para camera atual"""
        if self.active_camera == 'csi' and self.pi_camera:
            self.pi_camera.stop()
        elif self.active_camera and self.active_camera.startswith('usb'):
            self.usb_camera.stop()
        self.active_camera = None

    def read_stream(self) -> Optional[np.ndarray]:
        """Le frame para stream"""
        if self.active_camera == 'csi' and self.pi_camera:
            return self.pi_camera.read_stream()
        elif self.active_camera and self.active_camera.startswith('usb'):
            return self.usb_camera.read_stream()
        return None

    def read_hd(self) -> Optional[np.ndarray]:
        """Le frame HD"""
        if self.active_camera == 'csi' and self.pi_camera:
            return self.pi_camera.read_hd()
        elif self.active_camera and self.active_camera.startswith('usb'):
            return self.usb_camera.read_hd()
        return None

    def capture_max_resolution(self) -> Optional[np.ndarray]:
        """Captura em resolucao maxima (apenas CSI)"""
        if self.active_camera == 'csi' and self.pi_camera:
            return self.pi_camera.capture_full_resolution()
        return self.read_hd()

    def get_camera_config(self) -> dict:
        """Retorna configuracao da camera atual"""
        if self.active_camera == 'csi' and self.pi_camera:
            config = self.pi_camera.get_config()
            return {
                'type': 'csi',
                'exposure_time': {'value': config['exposure_time'], 'min': 1000, 'max': 100000, 'unit': 'us'},
                'analogue_gain': {'value': config['analogue_gain'], 'min': 1.0, 'max': 16.0},
                'brightness': {'value': config['brightness'], 'min': -1.0, 'max': 1.0},
                'contrast': {'value': config['contrast'], 'min': 0.0, 'max': 2.0},
                'sharpness': {'value': config['sharpness'], 'min': 0.0, 'max': 2.0}
            }
        return {'type': 'usb'}

    def set_camera_config(self, config: dict) -> bool:
        """Aplica configuracao na camera"""
        if self.active_camera == 'csi' and self.pi_camera:
            try:
                if 'exposure_time' in config:
                    self.pi_camera.set_exposure(int(config['exposure_time']))
                if 'analogue_gain' in config:
                    self.pi_camera.set_gain(float(config['analogue_gain']))
                if 'brightness' in config:
                    self.pi_camera.set_brightness(float(config['brightness']))
                if 'contrast' in config:
                    self.pi_camera.set_contrast(float(config['contrast']))
                if 'sharpness' in config:
                    self.pi_camera.set_sharpness(float(config['sharpness']))
                return True
            except Exception as e:
                print(f"Erro ao configurar camera: {e}")
        return False

    def stop(self):
        """Para todas as cameras"""
        self.stop_current()


# ============================================================================
# THREADS DE CAPTURA E DETECCAO
# ============================================================================

class CaptureThread:
    """Thread para captura continua"""

    def __init__(self, camera_manager: UnifiedCameraManager, config: DetectorConfig):
        self.camera_manager = camera_manager
        self.config = config

        self.stream_frame = None
        self.hd_frame = None
        self.stream_lock = threading.Lock()
        self.hd_lock = threading.Lock()

        self.running = False
        self.thread = None

    def start(self):
        """Inicia thread de captura"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        """Loop de captura"""
        while self.running:
            stream_frame = self.camera_manager.read_stream()
            if stream_frame is not None:
                with self.stream_lock:
                    self.stream_frame = stream_frame.copy()

            hd_frame = self.camera_manager.read_hd()
            if hd_frame is not None:
                with self.hd_lock:
                    self.hd_frame = hd_frame.copy()

            time.sleep(0.001)

    def read_stream(self) -> Optional[np.ndarray]:
        """Le frame de stream"""
        with self.stream_lock:
            return self.stream_frame.copy() if self.stream_frame is not None else None

    def read_hd(self) -> Optional[np.ndarray]:
        """Le frame HD"""
        with self.hd_lock:
            return self.hd_frame.copy() if self.hd_frame is not None else None

    def stop(self):
        """Para thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


class DetectorThread:
    """Thread para deteccao ONNX"""

    def __init__(self, detector: CardDetectorONNX, config: DetectorConfig):
        self.detector = detector
        self.config = config

        self.detections = []
        self.detections_hd = []
        self.inference_time = 0.0
        self.fps = 0.0
        self.lock = threading.Lock()

        self.running = False
        self.thread = None

        self.input_frame = None
        self.input_lock = threading.Lock()
        self.new_frame_event = threading.Event()

    def start(self):
        """Inicia thread"""
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        return self

    def submit_frame(self, frame: np.ndarray):
        """Envia frame para processamento"""
        with self.input_lock:
            self.input_frame = frame
        self.new_frame_event.set()

    def _scale_detections_to_hd(self, detections: List[dict], stream_w: int, stream_h: int, hd_w: int, hd_h: int) -> List[dict]:
        """Escala coordenadas para HD"""
        scale_x = hd_w / stream_w
        scale_y = hd_h / stream_h

        hd_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            hd_det = det.copy()
            hd_det['bbox'] = [
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y)
            ]
            hd_detections.append(hd_det)

        return hd_detections

    def _detection_loop(self):
        """Loop de deteccao"""
        fps_counter = 0
        fps_start = time.time()

        while self.running:
            if not self.new_frame_event.wait(timeout=0.1):
                continue
            self.new_frame_event.clear()

            with self.input_lock:
                frame = self.input_frame
                self.input_frame = None

            if frame is None:
                continue

            detections, inference_time = self.detector.detect(frame)

            # Escala para HD (assumindo proporcao do config)
            detections_hd = self._scale_detections_to_hd(
                detections,
                self.config.stream_width, self.config.stream_height,
                self.config.csi_capture_width, self.config.csi_capture_height
            )

            with self.lock:
                self.detections = detections
                self.detections_hd = detections_hd
                self.inference_time = inference_time

            fps_counter += 1
            if fps_counter >= 10:
                elapsed = time.time() - fps_start
                if elapsed > 0:
                    with self.lock:
                        self.fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

    def get_results(self) -> Tuple[List[dict], List[dict], float, float]:
        """Retorna deteccoes"""
        with self.lock:
            return list(self.detections), list(self.detections_hd), self.inference_time, self.fps

    def stop(self):
        """Para thread"""
        self.running = False
        self.new_frame_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)


# ============================================================================
# FUNCOES DE DESENHO
# ============================================================================

def draw_detections(frame: np.ndarray, detections: List[dict], config: DetectorConfig) -> np.ndarray:
    """Desenha deteccoes no frame"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_name = det['class_name']

        cv2.rectangle(frame, (x1, y1), (x2, y2), config.bbox_color, config.bbox_thickness)

        label = f"{class_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.font_scale, 2)

        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), config.bbox_color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, config.font_scale, (0, 0, 0), 2)

    return frame


def draw_stats(frame: np.ndarray, fps: float, inference_ms: float, num_detections: int, camera_type: str) -> np.ndarray:
    """Desenha estatisticas no frame"""
    stats = [
        f"FPS: {fps:.1f}",
        f"Inference: {inference_ms:.1f}ms",
        f"Detections: {num_detections}",
        f"Camera: {camera_type.upper()}"
    ]

    y = 25
    for stat in stats:
        cv2.putText(frame, stat, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y += 22

    return frame


# ============================================================================
# APLICACAO FLASK
# ============================================================================

# Configuracao global
config = DetectorConfig()

# Resolve caminho do modelo
model_path = BASE_DIR / config.model_path
if not model_path.exists():
    print(f"AVISO: Modelo nao encontrado em {model_path}")
    model_path = None

# Inicializa componentes
camera_manager = UnifiedCameraManager(config)
detector = CardDetectorONNX(str(model_path), config) if model_path else None
detector_thread = None
capture_thread = None

# Estatisticas
stats = {
    'fps': 0.0,
    'inference_time': 0.0,
    'detections': 0,
    'camera_type': 'none',
    'has_detection': False
}
stats_lock = threading.Lock()


def init_camera():
    """Inicializa camera (prioriza CSI)"""
    global capture_thread, detector_thread

    # Tenta CSI primeiro, depois USB 0
    cameras = camera_manager.list_cameras()
    if cameras:
        first_camera = cameras[0]['id']
        if camera_manager.select_camera(first_camera):
            capture_thread = CaptureThread(camera_manager, config).start()

            if detector and detector.is_ready():
                detector_thread = DetectorThread(detector, config).start()

            with stats_lock:
                stats['camera_type'] = cameras[0]['type']


def capture_roi() -> dict:
    """Captura ROI da carta"""
    if capture_thread is None:
        return {'success': False, 'error': 'Camera nao inicializada'}

    if detector is None or not detector.is_ready():
        return {'success': False, 'error': 'Detector nao inicializado'}

    # Tenta captura em resolucao maxima (apenas CSI)
    hd_frame = camera_manager.capture_max_resolution()
    if hd_frame is None:
        hd_frame = capture_thread.read_hd()

    if hd_frame is None:
        return {'success': False, 'error': 'Frame nao disponivel'}

    # Detecta no frame HD
    detections_hd, inference_time = detector.detect(hd_frame)

    if not detections_hd:
        return {'success': False, 'error': 'Nenhuma carta detectada'}

    # Melhor deteccao
    best_detection = max(detections_hd, key=lambda x: x['confidence'])
    x1, y1, x2, y2 = best_detection['bbox']

    real_h, real_w = hd_frame.shape[:2]

    # Aspect ratio carta TCG
    CARD_RATIO = 88.0 / 63.0
    det_w = x2 - x1
    det_h = y2 - y1

    if det_w > 0:
        current_ratio = det_h / det_w
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if abs(current_ratio - CARD_RATIO) > 0.1:
            if current_ratio < CARD_RATIO:
                new_h = int(det_w * CARD_RATIO)
                y1 = max(0, cy - new_h // 2)
                y2 = min(real_h, cy + new_h // 2)
            else:
                new_w = int(det_h / CARD_RATIO)
                x1 = max(0, cx - new_w // 2)
                x2 = min(real_w, cx + new_w // 2)

    # Margem
    margin_x = int((x2 - x1) * 0.03)
    margin_y = int((y2 - y1) * 0.03)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(real_w, x2 + margin_x)
    y2 = min(real_h, y2 + margin_y)

    roi = hd_frame[y1:y2, x1:x2]

    if roi.size == 0:
        return {'success': False, 'error': 'ROI invalido'}

    # Pre-processamento
    roi_processed = preprocess_card_image(roi)

    # Salva
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"card_{timestamp}.png"
    filepath = CAPTURES_DIR / filename

    cv2.imwrite(str(filepath), roi_processed)

    print(f"Captura salva: {filepath} ({roi.shape[1]}x{roi.shape[0]})")

    return {
        'success': True,
        'filename': filename,
        'resolution': f"{roi.shape[1]}x{roi.shape[0]}",
        'confidence': best_detection['confidence'],
        'inference_ms': inference_time,
        'path': f"/captures/{filename}"
    }


def generate_frames():
    """Gerador de frames MJPEG"""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
    frame_count = 0

    target_fps = 30
    frame_time = 1.0 / target_fps
    last_frame_time = time.time()

    stream_fps = 0.0
    stream_fps_counter = 0
    stream_fps_start = time.time()

    while True:
        now = time.time()
        elapsed = now - last_frame_time
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)
        last_frame_time = time.time()

        if capture_thread is None:
            frame = np.zeros((config.stream_height, config.stream_width, 3), dtype=np.uint8)
            cv2.putText(frame, "Selecione uma camera", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            frame = capture_thread.read_stream()
            if frame is None:
                continue

            frame_count += 1

            if detector_thread and frame_count % config.frame_skip == 0:
                detector_thread.submit_frame(frame.copy())

            detections, _, inference_time, detect_fps = [], [], 0.0, 0.0
            has_detection = False
            if detector_thread:
                detections, _, inference_time, detect_fps = detector_thread.get_results()
                has_detection = len(detections) > 0

            stream_fps_counter += 1
            if stream_fps_counter >= 30:
                stream_elapsed = time.time() - stream_fps_start
                if stream_elapsed > 0:
                    stream_fps = stream_fps_counter / stream_elapsed
                stream_fps_counter = 0
                stream_fps_start = time.time()

            camera_type = camera_manager.active_camera or 'none'
            frame = draw_detections(frame, detections, config)
            frame = draw_stats(frame, stream_fps, inference_time, len(detections), camera_type)

            with stats_lock:
                stats['fps'] = stream_fps
                stats['inference_time'] = inference_time
                stats['detections'] = len(detections)
                stats['has_detection'] = has_detection
                stats['camera_type'] = camera_type

        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# Flask App
app = Flask(__name__)


@app.route('/')
def index():
    """Pagina principal"""
    return render_template('index_onnx_v3.html')


@app.route('/video_feed')
def video_feed():
    """Streaming MJPEG"""
    response = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/cameras')
def get_cameras():
    """Lista cameras"""
    cameras = camera_manager.list_cameras()
    return jsonify(cameras)


@app.route('/camera/<camera_id>', methods=['POST'])
def set_camera(camera_id):
    """Troca camera"""
    global capture_thread

    if capture_thread:
        capture_thread.stop()
        capture_thread = None

    # Converte ID se for numero
    try:
        camera_id = int(camera_id)
    except ValueError:
        pass  # Mantem como string ('csi')

    success = camera_manager.select_camera(camera_id)
    if success:
        capture_thread = CaptureThread(camera_manager, config).start()

    return jsonify({'success': success, 'camera_id': str(camera_id)})


@app.route('/stats')
def get_stats():
    """Estatisticas"""
    with stats_lock:
        return jsonify(stats)


@app.route('/capture', methods=['POST'])
def capture():
    """Captura ROI"""
    result = capture_roi()
    return jsonify(result)


@app.route('/captures/<filename>')
def serve_capture(filename):
    """Serve captura"""
    return send_from_directory(str(CAPTURES_DIR), filename)


@app.route('/captures')
def list_captures():
    """Lista capturas"""
    files = sorted(list(CAPTURES_DIR.glob("*.jpg")) + list(CAPTURES_DIR.glob("*.png")), reverse=True)
    captures = [{'filename': f.name, 'path': f"/captures/{f.name}", 'size': f.stat().st_size} for f in files[:50]]
    return jsonify(captures)


@app.route('/camera/config', methods=['GET'])
def get_camera_config():
    """Config da camera"""
    return jsonify(camera_manager.get_camera_config())


@app.route('/camera/config', methods=['POST'])
def set_camera_config():
    """Aplica config"""
    data = request.get_json() or {}
    success = camera_manager.set_camera_config(data)
    return jsonify({'success': success, 'config': camera_manager.get_camera_config()})


def cleanup():
    """Libera recursos"""
    if detector_thread:
        detector_thread.stop()
    if capture_thread:
        capture_thread.stop()
    camera_manager.stop()
    print("Recursos liberados")


def main():
    print("=" * 60)
    print("WEB DETECTOR ONNX v3 - Camera CSI (Raspberry Pi 5)")
    print("=" * 60)

    if IS_RASPBERRY_PI:
        print(f"Plataforma: Raspberry Pi ({platform.machine()})")
        print(f"Picamera2: {'Disponivel' if PICAMERA2_AVAILABLE else 'NAO disponivel'}")
    else:
        print(f"Plataforma: {platform.system()} ({platform.machine()})")
        print("Modo: Webcam USB apenas")

    if not ONNX_AVAILABLE:
        print("\nERRO: ONNX Runtime nao instalado!")
        return 1

    if detector is None or not detector.is_ready():
        print(f"\nAVISO: Detector nao inicializado - {config.model_path}")
    else:
        print(f"Modelo: {config.model_path}")

    print(f"Stream: {config.stream_width}x{config.stream_height}")
    print(f"Capturas em: {CAPTURES_DIR}")
    print("=" * 60)

    init_camera()

    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))

    print(f"\nServidor: http://{config.host}:{config.port}")
    print("Pressione Ctrl+C para encerrar\n")

    try:
        app.run(host=config.host, port=config.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nInterrompido")
    finally:
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
