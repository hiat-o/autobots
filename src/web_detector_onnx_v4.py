"""
Web Detector ONNX v4 - Deteccao de Cartas TCG (Raspberry Pi 5)
Streaming MJPEG com Flask + ONNX Runtime + Picamera2

Versao simplificada - apenas controles nativos do Picamera2.
Sem efeitos por software (gamma, tint, color_brightness).
Sem analise de exposicao/qualidade.
Sem pre-processamento na captura.

Funcionalidades:
  - Suporte a camera CSI via Picamera2 (Raspberry Pi)
  - Fallback para webcam USB via OpenCV
  - Deteccao de cartas usando modelo ONNX
  - Troca de resolucao e camera ao vivo
  - Captura ROI da carta detectada em HD

Uso:
    python src/web_detector_onnx_v4.py
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

# Diretorios
BASE_DIR = Path(__file__).parent.parent
CAPTURES_DIR = BASE_DIR / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)


@dataclass
class DetectorConfig:
    """Configuracoes do detector"""
    model_path: str = "data/best_card_detect.onnx"
    model_input_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    class_names: Dict[int, str] = field(default_factory=lambda: {0: "carta_tcg"})

    # Camera USB - Fallback
    usb_capture_width: int = 1280
    usb_capture_height: int = 720
    camera_fps: int = 30

    # Stream
    stream_width: int = 640
    stream_height: int = 480
    jpeg_quality: int = 85
    frame_skip: int = 1

    # Display
    bbox_color: Tuple[int, int, int] = (0, 255, 0)
    bbox_thickness: int = 2
    font_scale: float = 0.6

    # Web
    host: str = "0.0.0.0"
    port: int = 5000


# ============================================================================
# DETECTOR ONNX
# ============================================================================

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

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            model_path, sess_options, providers=['CPUExecutionProvider']
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        print(f"Modelo ONNX carregado: {model_path}")
        print(f"Input: {self.input_name}, shape: {self.input_shape}")

    def is_ready(self) -> bool:
        return self.session is not None

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
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

        info = {'scale': scale, 'pad_w': pad_w, 'pad_h': pad_h, 'orig_w': w, 'orig_h': h}
        return blob, info

    def postprocess(self, output: np.ndarray, info: dict) -> List[dict]:
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
            pad_w, pad_h = info['pad_w'], info['pad_h']

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
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    def detect(self, frame: np.ndarray) -> Tuple[List[dict], float]:
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
# CAMERA CSI (PICAMERA2)
# ============================================================================

class PiCameraManager:
    """Gerencia camera CSI via Picamera2 - controles nativos apenas"""

    SENSOR_RESOLUTIONS = {
        'imx219': [
            # Paisagem
            {'label': '3280x2464 Paisagem (Max 8MP)',  'main': (3280, 2464), 'lores': (820, 616)},
            {'label': '1920x1080 Paisagem (FHD)',      'main': (1920, 1080), 'lores': (640, 360)},
            {'label': '1640x1232 Paisagem',            'main': (1640, 1232), 'lores': (820, 616)},
            {'label': '1280x720 Paisagem (HD 720p)',   'main': (1280, 720),  'lores': (640, 360)},
            {'label': '640x480 Paisagem (Fast)',       'main': (640, 480),   'lores': (640, 480)},
            # Retrato
            {'label': '2464x3280 Retrato (Max 8MP)',   'main': (2464, 3280), 'lores': (616, 820)},
            {'label': '1080x1920 Retrato (FHD)',       'main': (1080, 1920), 'lores': (360, 640)},
            {'label': '1232x1640 Retrato',             'main': (1232, 1640), 'lores': (616, 820)},
            {'label': '720x1280 Retrato (HD 720p)',    'main': (720, 1280),  'lores': (360, 640)},
            {'label': '480x640 Retrato (Fast)',        'main': (480, 640),   'lores': (480, 640)},
        ],
        'ov5647': [
            # Paisagem
            {'label': '2592x1944 Paisagem (Max 5MP)',  'main': (2592, 1944), 'lores': (648, 486)},
            {'label': '1920x1080 Paisagem (FHD)',      'main': (1920, 1080), 'lores': (640, 360)},
            {'label': '1296x972 Paisagem',             'main': (1296, 972),  'lores': (648, 486)},
            {'label': '1280x720 Paisagem (HD 720p)',   'main': (1280, 720),  'lores': (640, 360)},
            {'label': '640x480 Paisagem (Fast)',       'main': (640, 480),   'lores': (640, 480)},
            # Retrato
            {'label': '1944x2592 Retrato (Max 5MP)',   'main': (1944, 2592), 'lores': (486, 648)},
            {'label': '1080x1920 Retrato (FHD)',       'main': (1080, 1920), 'lores': (360, 640)},
            {'label': '972x1296 Retrato',              'main': (972, 1296),  'lores': (486, 648)},
            {'label': '720x1280 Retrato (HD 720p)',    'main': (720, 1280),  'lores': (360, 640)},
            {'label': '480x640 Retrato (Fast)',        'main': (480, 640),   'lores': (480, 640)},
        ],
    }

    def __init__(self, config: DetectorConfig, camera_index: int = 0, camera_model: str = 'unknown'):
        self.config = config
        self.camera_index = camera_index
        self.camera_model = camera_model.lower()
        self.picam2 = None
        self.lock = threading.Lock()
        self.is_running = False

        self.current_resolution_index = self._find_default_resolution_index()

        # Controles nativos Picamera2 - valores padrao recomendados
        self.current_config = {
            'ae_enable': True,
            'awb_enable': True,
            'brightness': 0.0,      # -1.0 a 1.0
            'contrast': 1.0,        # 0.0 a 32.0
            'saturation': 1.0,      # 0.0 a 32.0
            'sharpness': 1.0,       # 0.0 a 16.0
            'horizontal_flip': False,
            'vertical_flip': False,
        }

    def _find_default_resolution_index(self) -> int:
        resolutions = self.SENSOR_RESOLUTIONS.get(self.camera_model, [])
        for i, res in enumerate(resolutions):
            if res['main'] == (1920, 1080):
                return i
        return 0

    def get_resolutions(self) -> List[dict]:
        resolutions = self.SENSOR_RESOLUTIONS.get(self.camera_model, [])
        return [
            {'index': i, 'label': res['label'],
             'main': f"{res['main'][0]}x{res['main'][1]}",
             'active': i == self.current_resolution_index}
            for i, res in enumerate(resolutions)
        ]

    def _get_current_preset(self):
        resolutions = self.SENSOR_RESOLUTIONS.get(self.camera_model, [])
        if resolutions and self.current_resolution_index < len(resolutions):
            return resolutions[self.current_resolution_index]
        return {'main': (1920, 1080), 'lores': (640, 480)}

    def set_resolution(self, resolution_index: int) -> bool:
        resolutions = self.SENSOR_RESOLUTIONS.get(self.camera_model, [])
        if resolution_index < 0 or resolution_index >= len(resolutions):
            return False

        self.current_resolution_index = resolution_index
        preset = resolutions[resolution_index]

        if not self.is_running or self.picam2 is None:
            return True

        try:
            with self.lock:
                self.picam2.stop()
                preview_config = self.picam2.create_preview_configuration(
                    main={"size": preset['main'], "format": "RGB888"},
                    lores={"size": preset['lores'], "format": "RGB888"},
                    buffer_count=4
                )
                self.picam2.configure(preview_config)
                self._apply_controls()
                self.picam2.start()
                time.sleep(0.3)
            print(f"Resolucao alterada: {preset['main'][0]}x{preset['main'][1]} ({preset['label']})")
            return True
        except Exception as e:
            print(f"Erro ao trocar resolucao: {e}")
            return False

    def _apply_controls(self):
        """Aplica controles nativos do Picamera2"""
        ctrl = {
            "AeEnable": self.current_config['ae_enable'],
            "AwbEnable": self.current_config['awb_enable'],
            "Brightness": self.current_config['brightness'],
            "Contrast": self.current_config['contrast'],
            "Saturation": self.current_config['saturation'],
            "Sharpness": self.current_config['sharpness'],
        }
        self.picam2.set_controls(ctrl)

    def start(self) -> bool:
        if not PICAMERA2_AVAILABLE:
            return False

        try:
            self.picam2 = Picamera2(self.camera_index)
            preset = self._get_current_preset()

            preview_config = self.picam2.create_preview_configuration(
                main={"size": preset['main'], "format": "RGB888"},
                lores={"size": preset['lores'], "format": "RGB888"},
                buffer_count=4
            )
            self.picam2.configure(preview_config)
            self._apply_controls()
            self.picam2.start()
            self.is_running = True
            time.sleep(0.5)
            print(f"Camera CSI {self.camera_index} ({self.camera_model}) iniciada: {preset['main'][0]}x{preset['main'][1]}")
            return True
        except Exception as e:
            print(f"Erro ao iniciar camera CSI: {e}")
            return False

    def read_stream(self) -> Optional[np.ndarray]:
        if not self.is_running or self.picam2 is None:
            return None
        try:
            with self.lock:
                frame = self.picam2.capture_array("lores")
                return self._apply_flips(frame)
        except Exception as e:
            print(f"Erro ao ler frame stream: {e}")
            return None

    def read_hd(self) -> Optional[np.ndarray]:
        if not self.is_running or self.picam2 is None:
            return None
        try:
            with self.lock:
                frame = self.picam2.capture_array("main")
                return self._apply_flips(frame)
        except Exception as e:
            print(f"Erro ao ler frame HD: {e}")
            return None

    def _apply_flips(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return None
        h_flip = self.current_config['horizontal_flip']
        v_flip = self.current_config['vertical_flip']
        if h_flip and v_flip:
            return cv2.flip(frame, -1)
        elif h_flip:
            return cv2.flip(frame, 1)
        elif v_flip:
            return cv2.flip(frame, 0)
        return frame

    def set_config(self, cfg: dict):
        """Aplica configuracao recebida do frontend"""
        if 'ae_enable' in cfg:
            self.current_config['ae_enable'] = bool(cfg['ae_enable'])
        if 'awb_enable' in cfg:
            self.current_config['awb_enable'] = bool(cfg['awb_enable'])
        if 'brightness' in cfg:
            self.current_config['brightness'] = max(-1.0, min(1.0, float(cfg['brightness'])))
        if 'contrast' in cfg:
            self.current_config['contrast'] = max(0.0, min(32.0, float(cfg['contrast'])))
        if 'saturation' in cfg:
            self.current_config['saturation'] = max(0.0, min(32.0, float(cfg['saturation'])))
        if 'sharpness' in cfg:
            self.current_config['sharpness'] = max(0.0, min(16.0, float(cfg['sharpness'])))
        if 'horizontal_flip' in cfg:
            self.current_config['horizontal_flip'] = bool(cfg['horizontal_flip'])
        if 'vertical_flip' in cfg:
            self.current_config['vertical_flip'] = bool(cfg['vertical_flip'])

        if self.picam2 and self.is_running:
            self._apply_controls()

    def get_config(self) -> dict:
        return self.current_config.copy()

    def stop(self):
        if self.picam2 and self.is_running:
            self.picam2.stop()
            self.picam2.close()
            self.is_running = False
            print("Camera CSI parada")


# ============================================================================
# CAMERA USB (OPENCV)
# ============================================================================

class USBCameraManager:
    """Gerencia webcam USB via OpenCV (fallback)"""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.cap = None
        self.current_camera_id = 0
        self.lock = threading.Lock()

    def list_cameras(self) -> List[Dict]:
        if PICAMERA2_AVAILABLE:
            return []
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append({
                    'id': i, 'name': f"USB Camera {i}",
                    'resolution': f"{w}x{h}", 'type': 'usb',
                    'active': i == self.current_camera_id
                })
                cap.release()
        return cameras

    def start(self, camera_id: int = 0) -> bool:
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
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return None
            ret, frame = self.cap.read()
            if ret:
                return cv2.resize(frame, (self.config.stream_width, self.config.stream_height))
            return None

    def read_hd(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return None
            ret, frame = self.cap.read()
            return frame if ret else None

    def stop(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None


# ============================================================================
# CAMERA MANAGER UNIFICADO
# ============================================================================

class UnifiedCameraManager:
    """Gerencia cameras CSI e USB de forma unificada"""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.pi_camera = None
        self.usb_camera = USBCameraManager(config)
        self.active_camera = None
        self.lock = threading.Lock()

        self.csi_cameras = []
        if PICAMERA2_AVAILABLE:
            try:
                for cam_info in Picamera2.global_camera_info():
                    self.csi_cameras.append({
                        'index': cam_info['Num'],
                        'model': cam_info.get('Model', 'unknown'),
                    })
                print(f"Cameras CSI detectadas: {len(self.csi_cameras)}")
                for cam in self.csi_cameras:
                    print(f"  CSI {cam['index']}: {cam['model']}")
            except Exception as e:
                print(f"Erro ao detectar cameras CSI: {e}")

    def list_cameras(self) -> List[Dict]:
        cameras = []
        for cam in self.csi_cameras:
            idx = cam['index']
            model = cam['model']
            cameras.append({
                'id': f'csi_{idx}',
                'name': f'Camera CSI {idx} ({model.upper()})',
                'type': 'csi', 'sensor': model,
                'active': self.active_camera == f'csi_{idx}'
            })
        usb_cameras = self.usb_camera.list_cameras()
        for cam in usb_cameras:
            cam['active'] = self.active_camera == f"usb_{cam['id']}"
        cameras.extend(usb_cameras)
        return cameras

    def select_camera(self, camera_id) -> bool:
        with self.lock:
            self.stop_current()
            camera_id_str = str(camera_id)
            if camera_id_str == 'csi':
                camera_id_str = 'csi_0'

            if camera_id_str.startswith('csi_'):
                csi_index = int(camera_id_str.replace('csi_', ''))
                cam_info = next((c for c in self.csi_cameras if c['index'] == csi_index), None)
                if cam_info is None:
                    return False
                self.pi_camera = PiCameraManager(self.config, csi_index, cam_info['model'])
                if self.pi_camera.start():
                    self.active_camera = f'csi_{csi_index}'
                    return True
                self.pi_camera = None
                return False

            if isinstance(camera_id, int) or camera_id_str.startswith('usb_'):
                usb_id = int(camera_id_str.replace('usb_', '')) if camera_id_str.startswith('usb_') else int(camera_id)
                if self.usb_camera.start(usb_id):
                    self.active_camera = f"usb_{usb_id}"
                    return True
            return False

    def stop_current(self):
        if self.pi_camera and self.active_camera and self.active_camera.startswith('csi'):
            self.pi_camera.stop()
            self.pi_camera = None
        elif self.active_camera and self.active_camera.startswith('usb'):
            self.usb_camera.stop()
        self.active_camera = None

    def read_stream(self) -> Optional[np.ndarray]:
        if self.pi_camera and self.active_camera and self.active_camera.startswith('csi'):
            return self.pi_camera.read_stream()
        if self.active_camera and self.active_camera.startswith('usb'):
            return self.usb_camera.read_stream()
        return None

    def read_hd(self) -> Optional[np.ndarray]:
        if self.pi_camera and self.active_camera and self.active_camera.startswith('csi'):
            return self.pi_camera.read_hd()
        if self.active_camera and self.active_camera.startswith('usb'):
            return self.usb_camera.read_hd()
        return None

    def get_camera_config(self) -> dict:
        if self.pi_camera and self.active_camera and self.active_camera.startswith('csi'):
            cfg = self.pi_camera.get_config()
            return {
                'type': 'csi',
                'camera_model': self.pi_camera.camera_model,
                'ae_enable': cfg['ae_enable'],
                'awb_enable': cfg['awb_enable'],
                'brightness': {'value': cfg['brightness'], 'min': -1.0, 'max': 1.0},
                'contrast': {'value': cfg['contrast'], 'min': 0.0, 'max': 32.0},
                'saturation': {'value': cfg['saturation'], 'min': 0.0, 'max': 32.0},
                'sharpness': {'value': cfg['sharpness'], 'min': 0.0, 'max': 16.0},
                'horizontal_flip': cfg['horizontal_flip'],
                'vertical_flip': cfg['vertical_flip'],
            }
        return {'type': 'usb'}

    def set_camera_config(self, cfg: dict) -> bool:
        if self.pi_camera and self.active_camera and self.active_camera.startswith('csi'):
            try:
                self.pi_camera.set_config(cfg)
                return True
            except Exception as e:
                print(f"Erro ao configurar camera: {e}")
        return False

    def get_resolutions(self) -> List[dict]:
        if self.pi_camera and self.active_camera and self.active_camera.startswith('csi'):
            return self.pi_camera.get_resolutions()
        return []

    def set_resolution(self, resolution_index: int) -> bool:
        if self.pi_camera and self.active_camera and self.active_camera.startswith('csi'):
            return self.pi_camera.set_resolution(resolution_index)
        return False

    def stop(self):
        self.stop_current()


# ============================================================================
# THREADS DE CAPTURA E DETECCAO
# ============================================================================

class CaptureThread:
    """Thread para captura continua de stream"""

    def __init__(self, camera_manager: UnifiedCameraManager):
        self.camera_manager = camera_manager
        self.stream_frame = None
        self.stream_lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        frame_interval = 1.0 / 30
        while self.running:
            t0 = time.perf_counter()
            frame = self.camera_manager.read_stream()
            if frame is not None:
                with self.stream_lock:
                    self.stream_frame = frame
            elapsed = time.perf_counter() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read_stream(self) -> Optional[np.ndarray]:
        with self.stream_lock:
            return self.stream_frame.copy() if self.stream_frame is not None else None

    def read_hd(self) -> Optional[np.ndarray]:
        return self.camera_manager.read_hd()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


class DetectorThread:
    """Thread para deteccao ONNX"""

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
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        return self

    def submit_frame(self, frame: np.ndarray):
        with self.input_lock:
            self.input_frame = frame
        self.new_frame_event.set()

    def _detection_loop(self):
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

            with self.lock:
                self.detections = detections
                self.inference_time = inference_time

            fps_counter += 1
            if fps_counter >= 10:
                elapsed = time.time() - fps_start
                if elapsed > 0:
                    with self.lock:
                        self.fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

    def get_results(self) -> Tuple[List[dict], float, float]:
        with self.lock:
            return list(self.detections), self.inference_time, self.fps

    def stop(self):
        self.running = False
        self.new_frame_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)


# ============================================================================
# FUNCOES DE DESENHO
# ============================================================================

def draw_detections(frame: np.ndarray, detections: List[dict], config: DetectorConfig) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        label = f"{det['class_name']}: {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), config.bbox_color, config.bbox_thickness)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.font_scale, 2)
        cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 5, y1), config.bbox_color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, config.font_scale, (0, 0, 0), 2)
    return frame


def draw_stats(frame: np.ndarray, fps: float, inference_ms: float, num_det: int, cam_type: str) -> np.ndarray:
    stats = [f"FPS: {fps:.1f}", f"Inference: {inference_ms:.1f}ms", f"Detections: {num_det}", f"Camera: {cam_type.upper()}"]
    y = 25
    for s in stats:
        cv2.putText(frame, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y += 22
    return frame


# ============================================================================
# APLICACAO FLASK
# ============================================================================

config = DetectorConfig()
model_path = BASE_DIR / config.model_path
if not model_path.exists():
    print(f"AVISO: Modelo nao encontrado em {model_path}")
    model_path = None

camera_manager = UnifiedCameraManager(config)
detector = CardDetectorONNX(str(model_path), config) if model_path else None
detector_thread = None
capture_thread = None

stats = {
    'fps': 0.0, 'inference_time': 0.0, 'detections': 0,
    'camera_type': 'none', 'has_detection': False
}
stats_lock = threading.Lock()


def init_camera():
    global capture_thread, detector_thread
    cameras = camera_manager.list_cameras()
    if cameras:
        if camera_manager.select_camera(cameras[0]['id']):
            capture_thread = CaptureThread(camera_manager).start()
            if detector and detector.is_ready():
                detector_thread = DetectorThread(detector).start()
            with stats_lock:
                stats['camera_type'] = cameras[0]['type']


def capture_roi() -> dict:
    if capture_thread is None:
        return {'success': False, 'error': 'Camera nao inicializada'}
    if detector is None or not detector.is_ready():
        return {'success': False, 'error': 'Detector nao inicializado'}

    hd_frame = capture_thread.read_hd()
    if hd_frame is None:
        return {'success': False, 'error': 'Frame nao disponivel'}

    detections, inference_time = detector.detect(hd_frame)
    if not detections:
        return {'success': False, 'error': 'Nenhuma carta detectada'}

    best = max(detections, key=lambda x: x['confidence'])
    x1, y1, x2, y2 = best['bbox']
    real_h, real_w = hd_frame.shape[:2]

    # Ajuste de aspect ratio carta TCG (88mm x 63mm)
    CARD_RATIO = 88.0 / 63.0
    det_w, det_h = x2 - x1, y2 - y1
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
    mx = int((x2 - x1) * 0.03)
    my = int((y2 - y1) * 0.03)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(real_w, x2 + mx), min(real_h, y2 + my)

    roi = hd_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return {'success': False, 'error': 'ROI invalido'}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"card_{timestamp}.png"
    filepath = CAPTURES_DIR / filename
    cv2.imwrite(str(filepath), roi)
    print(f"Captura salva: {filepath} ({roi.shape[1]}x{roi.shape[0]})")

    return {
        'success': True,
        'filename': filename,
        'resolution': f"{roi.shape[1]}x{roi.shape[0]}",
        'confidence': best['confidence'],
        'inference_ms': inference_time,
        'path': f"/captures/{filename}"
    }


def generate_frames():
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
    frame_count = 0
    frame_interval = 1.0 / 30

    stream_fps = 0.0
    stream_fps_counter = 0
    stream_fps_start = time.perf_counter()

    while True:
        t0 = time.perf_counter()

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

            detections, inference_time, detect_fps = [], 0.0, 0.0
            has_detection = False
            if detector_thread:
                detections, inference_time, detect_fps = detector_thread.get_results()
                has_detection = len(detections) > 0

            stream_fps_counter += 1
            if stream_fps_counter >= 30:
                elapsed = time.perf_counter() - stream_fps_start
                if elapsed > 0:
                    stream_fps = stream_fps_counter / elapsed
                stream_fps_counter = 0
                stream_fps_start = time.perf_counter()

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

        sleep_time = frame_interval - (time.perf_counter() - t0)
        if sleep_time > 0:
            time.sleep(sleep_time)


# Flask App
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index_onnx_v4.html')


@app.route('/video_feed')
def video_feed():
    response = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/cameras')
def get_cameras():
    return jsonify(camera_manager.list_cameras())


@app.route('/camera/<camera_id>', methods=['POST'])
def set_camera(camera_id):
    global capture_thread
    if capture_thread:
        capture_thread.stop()
        capture_thread = None
    try:
        camera_id = int(camera_id)
    except ValueError:
        pass
    success = camera_manager.select_camera(camera_id)
    if success:
        capture_thread = CaptureThread(camera_manager).start()
    return jsonify({'success': success, 'camera_id': str(camera_id)})


@app.route('/camera/resolutions')
def get_resolutions():
    return jsonify(camera_manager.get_resolutions())


@app.route('/camera/resolution', methods=['POST'])
def set_resolution():
    data = request.get_json() or {}
    index = data.get('index', 0)
    success = camera_manager.set_resolution(int(index))
    return jsonify({'success': success, 'resolutions': camera_manager.get_resolutions()})


@app.route('/stats')
def get_stats():
    with stats_lock:
        return jsonify(stats)


@app.route('/capture', methods=['POST'])
def capture():
    return jsonify(capture_roi())


@app.route('/captures/<filename>')
def serve_capture(filename):
    return send_from_directory(str(CAPTURES_DIR), filename)


@app.route('/captures')
def list_captures():
    files = sorted(list(CAPTURES_DIR.glob("*.jpg")) + list(CAPTURES_DIR.glob("*.png")), reverse=True)
    return jsonify([{'filename': f.name, 'path': f"/captures/{f.name}", 'size': f.stat().st_size} for f in files[:50]])


@app.route('/camera/config', methods=['GET'])
def get_camera_config():
    return jsonify(camera_manager.get_camera_config())


@app.route('/camera/config', methods=['POST'])
def set_camera_config():
    data = request.get_json() or {}
    success = camera_manager.set_camera_config(data)
    return jsonify({'success': success, 'config': camera_manager.get_camera_config()})


def cleanup():
    if detector_thread:
        detector_thread.stop()
    if capture_thread:
        capture_thread.stop()
    camera_manager.stop()
    print("Recursos liberados")


def main():
    print("=" * 60)
    print("WEB DETECTOR ONNX v4 - Simplificado")
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
