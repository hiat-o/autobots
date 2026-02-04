"""
Web Detector ONNX v3 - Deteccao de Cartas TCG com Camera CSI (Raspberry Pi 5)
Streaming MJPEG com Flask + ONNX Runtime + Picamera2

Funcionalidades:
  - Suporte a camera CSI IMX219 via Picamera2 (Raspberry Pi)
  - Fallback para webcam USB via OpenCV
  - Deteccao de cartas usando modelo ONNX
  - Stream em HD 720p (1280x720)
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


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Aplica correcao de gama. gamma < 1 = mais claro, gamma > 1 = mais escuro."""
    if gamma == 1.0:
        return image
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def apply_color_brightness(image: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Aplica brilho de cor (vibrance). amount: -1.0 a 1.0"""
    if amount == 0.0:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + amount)  # Ajusta saturacao
    hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + amount * 0.3)  # Leve ajuste no brilho
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_tint(image: np.ndarray, tint: float = 0.0) -> np.ndarray:
    """Aplica tonalidade (tint). tint: -150 a 150 (verde a magenta)."""
    if tint == 0.0:
        return image
    # Converte para LAB e ajusta canal 'b' (amarelo-azul) e 'a' (verde-magenta)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Tint positivo = magenta, negativo = verde
    lab[:, :, 1] = lab[:, :, 1] + (tint * 0.5)  # Canal 'a'
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def temperature_to_rgb_gains(temperature_k: int) -> Tuple[float, float]:
    """Converte temperatura de cor (Kelvin) para ganhos RGB (red, blue)."""
    # Baseado em aproximacao de Tanner Helland
    temp = temperature_k / 100.0

    # Calcular Red
    if temp <= 66:
        red = 255
    else:
        red = temp - 60
        red = 329.698727446 * (red ** -0.1332047592)
        red = max(0, min(255, red))

    # Calcular Blue
    if temp >= 66:
        blue = 255
    elif temp <= 19:
        blue = 0
    else:
        blue = temp - 10
        blue = 138.5177312231 * np.log(blue) - 305.0447927307
        blue = max(0, min(255, blue))

    # Normalizar para ganhos (referencia = 6500K, neutro)
    ref_temp = 6500 / 100.0
    ref_red = 255 if ref_temp <= 66 else max(0, min(255, 329.698727446 * ((ref_temp - 60) ** -0.1332047592)))
    ref_blue = 255 if ref_temp >= 66 else max(0, min(255, 138.5177312231 * np.log(ref_temp - 10) - 305.0447927307))

    red_gain = (red / ref_red) if ref_red > 0 else 1.0
    blue_gain = (blue / ref_blue) if ref_blue > 0 else 1.0

    # Limitar range para Picamera2 (tipicamente 0.0 a 8.0)
    red_gain = max(0.5, min(4.0, red_gain))
    blue_gain = max(0.5, min(4.0, blue_gain))

    return (red_gain, blue_gain)


def iso_to_gain(iso: int) -> float:
    """Converte valor ISO para ganho analogico. ISO 100 = gain 1.0"""
    return iso / 100.0


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


def analyze_exposure(image: np.ndarray) -> dict:
    """
    Analisa a exposicao da imagem e retorna metricas.

    Metricas:
    - brightness: brilho medio (0-255), ideal entre 100-160
    - overexposed_pct: % de pixels saturados (>250), ideal < 5%
    - underexposed_pct: % de pixels escuros (<10), ideal < 10%
    - status: 'ok', 'overexposed', 'underexposed'
    """
    # Converte para grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    total_pixels = gray.size

    # Brilho medio
    brightness = float(np.mean(gray))

    # Pixels saturados (overexposed) - valores > 250
    overexposed_pixels = np.sum(gray > 250)
    overexposed_pct = (overexposed_pixels / total_pixels) * 100

    # Pixels muito escuros (underexposed) - valores < 10
    underexposed_pixels = np.sum(gray < 10)
    underexposed_pct = (underexposed_pixels / total_pixels) * 100

    # Determina status
    if overexposed_pct > 5:
        status = 'overexposed'
    elif underexposed_pct > 15 or brightness < 80:
        status = 'underexposed'
    elif brightness > 180:
        status = 'overexposed'
    else:
        status = 'ok'

    return {
        'brightness': round(brightness, 1),
        'overexposed_pct': round(overexposed_pct, 1),
        'underexposed_pct': round(underexposed_pct, 1),
        'status': status
    }


def analyze_image_quality(image: np.ndarray) -> dict:
    """
    Analisa metricas de qualidade da imagem para decisao de captura.

    Metricas:
    - S_std (Estabilidade): desvio padrao do canal S (saturacao) no espaco LAB
    - S_eff (Qualidade real): media efetiva da saturacao
    - sanity (Sanidade visual): S_mean / std(V) - razao entre saturacao e variacao de brilho
    - color_hash (Hash): a_std + b_std - variacao de cor nos canais a e b
    - edge_density (OCR): densidade de bordas para avaliacao de nitidez/texto
    """
    if image is None or image.size == 0:
        return {
            'S_std': 0.0,
            'S_eff': 0.0,
            'sanity': 0.0,
            'color_hash': 0.0,
            'edge_density': 0.0
        }

    # Converte para LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    # Converte para HSV para canal S (saturacao)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # S_std - Estabilidade (desvio padrao da saturacao)
    S_std = float(np.std(S))

    # S_eff - Qualidade real (media da saturacao)
    S_mean = float(np.mean(S))
    S_eff = S_mean

    # Sanidade visual: S_mean / std(V)
    V_std = float(np.std(V))
    sanity = S_mean / V_std if V_std > 0 else 0.0

    # Hash de cor: a_std + b_std
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    color_hash = a_std + b_std

    # Edge density - densidade de bordas (para OCR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / edges.size * 100  # Percentual

    return {
        'S_std': round(S_std, 2),
        'S_eff': round(S_eff, 2),
        'sanity': round(sanity, 2),
        'color_hash': round(color_hash, 2),
        'edge_density': round(edge_density, 2)
    }


@dataclass
class DetectorConfig:
    """Configuracoes do detector ONNX"""
    model_path: str = "data/best_card_detect.onnx"
    model_input_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    class_names: Dict[int, str] = field(default_factory=lambda: {0: "carta_tcg"})

    # Camera CSI - Captura em alta resolucao (Portrait 9:16)
    csi_capture_width: int = 1080
    csi_capture_height: int = 1920
    csi_max_width: int = 3280  # Resolucao maxima IMX219
    csi_max_height: int = 2464

    # Camera USB - Fallback
    usb_capture_width: int = 1280
    usb_capture_height: int = 720
    camera_fps: int = 30

    # Stream - 720p Portrait (9:16)
    stream_width: int = 720
    stream_height: int = 1280

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

        # Configuracoes atuais da camera (valores baseados no painel de referencia)
        self.current_config = {
            # Exposicao
            'exposure_time': 16667,  # microsegundos (1/60s)
            'ae_enable': True,       # Auto Exposure ON
            # ISO (mapeado para analogue_gain)
            'iso': 100,              # ISO 100-3200 (gain 1.0-32.0)
            'analogue_gain': 1.0,    # iso / 100 = gain
            # Balanco de branco
            'awb_enable': True,      # Auto White Balance ON
            'temperature': 4013,     # Temperatura em Kelvin (range: 3000-8000)
            'tint': 52,              # Tonalidade (range: -150 a 150)
            'colour_gains': (1.5, 1.5),  # Ganhos de cor (red, blue) - valores iniciais neutros
            # Ajustes de imagem (ranges oficiais Picamera2)
            'brightness': 0,         # -100 a 100 (%) -> mapeado para -1.0 a 1.0
            'saturation': 1.0,       # 0.0-32.0 (oficial), pratico: 0.8-2.0
            'color_brightness': 0.0, # Brilho da cor -1.0 a 1.0 (software)
            'contrast': 1.0,         # 0.0-32.0 (oficial), pratico: 1.0-4.0
            'gamma': 1.0,            # 0.0-3.5 (software), pratico: 0.7-1.3
            'sharpness': 1.0,        # 0.0-16.0 (oficial), pratico: 1.0-4.0
            # Outros
            'noise_reduction': 1,    # 0=Off, 1=Fast, 2=HighQuality
            'horizontal_flip': False,
            'vertical_flip': False
        }

    def start(self) -> bool:
        """Inicia a camera CSI"""
        if not PICAMERA2_AVAILABLE:
            print("Picamera2 nao disponivel")
            return False

        try:
            self.picam2 = Picamera2()

            # Configura para stream (resolucao menor) + captura (resolucao alta)
            # IMPORTANTE: Picamera2 tem nomenclatura invertida!
            # "RGB888" retorna BGR (compativel com OpenCV)
            # "BGR888" retorna RGB (incompativel com OpenCV)
            preview_config = self.picam2.create_preview_configuration(
                main={"size": (self.config.csi_capture_width, self.config.csi_capture_height), "format": "RGB888"},
                lores={"size": (self.config.stream_width, self.config.stream_height), "format": "RGB888"},
                buffer_count=4
            )
            self.picam2.configure(preview_config)

            # Configura controles (sem HorizontalFlip/VerticalFlip - nao suportados no Pi 5)
            # Converte brilho de -100 a 100 para -1.0 a 1.0
            brightness_picam = self.current_config['brightness'] / 100.0
            controls = {
                "ExposureTime": self.current_config['exposure_time'],
                "AnalogueGain": self.current_config['analogue_gain'],
                "Brightness": brightness_picam,
                "Contrast": self.current_config['contrast'],
                "Saturation": self.current_config['saturation'],
                "Sharpness": self.current_config['sharpness'],
                "AeEnable": self.current_config['ae_enable'],
                "AwbEnable": self.current_config['awb_enable'],
                "NoiseReductionMode": self.current_config['noise_reduction'],
            }
            # Colour gains apenas se definidos manualmente (quando AWB desligado)
            if self.current_config['colour_gains'][0] > 0:
                controls["ColourGains"] = self.current_config['colour_gains']

            self.picam2.set_controls(controls)

            self.picam2.start()
            self.is_running = True

            # Aguarda estabilizar
            time.sleep(0.5)

            print(f"Camera CSI iniciada: {self.config.csi_capture_width}x{self.config.csi_capture_height}")
            return True

        except Exception as e:
            print(f"Erro ao iniciar camera CSI: {e}")
            return False

    def _apply_software_effects(self, frame: np.ndarray) -> np.ndarray:
        """Aplica efeitos de software (gama, tonalidade, brilho da cor)"""
        if frame is None:
            return None

        # Aplicar gama (se diferente de 1.0)
        gamma = self.current_config.get('gamma', 1.0)
        if gamma != 1.0 and gamma > 0:
            frame = apply_gamma_correction(frame, gamma)

        # Aplicar tonalidade (se diferente de 0)
        tint = self.current_config.get('tint', 0)
        if tint != 0:
            frame = apply_tint(frame, tint)

        # Aplicar brilho da cor (se diferente de 0)
        color_brightness = self.current_config.get('color_brightness', 0.0)
        if color_brightness != 0:
            frame = apply_color_brightness(frame, color_brightness)

        return frame

    def read_stream(self) -> Optional[np.ndarray]:
        """Le frame em resolucao de stream (lores)"""
        if not self.is_running or self.picam2 is None:
            return None

        try:
            with self.lock:
                # Captura frame - RGB888 retorna BGR (compativel com OpenCV)
                frame = self.picam2.capture_array("lores")
                frame = self._apply_flips(frame)
                frame = self._apply_software_effects(frame)
                return frame
        except Exception as e:
            print(f"Erro ao ler frame stream: {e}")
            return None

    def read_hd(self) -> Optional[np.ndarray]:
        """Le frame em alta resolucao (main)"""
        if not self.is_running or self.picam2 is None:
            return None

        try:
            with self.lock:
                # Captura frame HD - RGB888 retorna BGR (compativel com OpenCV)
                frame = self.picam2.capture_array("main")
                frame = self._apply_flips(frame)
                frame = self._apply_software_effects(frame)
                return frame
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

                # Reconfigura para resolucao maxima
                # "RGB888" retorna BGR (compativel com OpenCV)
                still_config = self.picam2.create_still_configuration(
                    main={"size": (self.config.csi_max_width, self.config.csi_max_height), "format": "RGB888"}
                )
                self.picam2.configure(still_config)
                self.picam2.start()

                # Captura - RGB888 retorna BGR, compativel com OpenCV
                time.sleep(0.2)  # Aguarda estabilizar
                frame = self.picam2.capture_array("main")
                frame = self._apply_flips(frame)

                # Restaura configuracao de stream
                self.picam2.stop()
                preview_config = self.picam2.create_preview_configuration(
                    main={"size": (self.config.csi_capture_width, self.config.csi_capture_height), "format": "RGB888"},
                    lores={"size": (self.config.stream_width, self.config.stream_height), "format": "RGB888"},
                    buffer_count=4
                )
                self.picam2.configure(preview_config)
                self.picam2.start()

                return frame

        except Exception as e:
            print(f"Erro ao capturar full resolution: {e}")
            return None

    def set_exposure(self, exposure_us: int):
        """Define tempo de exposicao em microsegundos (0-999999). 0 = auto"""
        exposure_us = max(0, min(999999, exposure_us))
        if self.picam2 and self.is_running:
            self.current_config['exposure_time'] = exposure_us
            self.picam2.set_controls({"ExposureTime": exposure_us})

    def set_gain(self, gain: float):
        """Define ganho analogico (1.0-32.0). ISO = gain * 100"""
        gain = max(1.0, min(32.0, gain))
        if self.picam2 and self.is_running:
            self.current_config['analogue_gain'] = gain
            self.picam2.set_controls({"AnalogueGain": gain})

    def set_iso(self, iso: int):
        """Define ISO (100-3200). Mapeado para AnalogueGain (1.0-32.0)."""
        iso = max(100, min(3200, iso))
        if self.picam2 and self.is_running:
            self.current_config['iso'] = iso
            gain = iso_to_gain(iso)
            gain = max(1.0, min(32.0, gain))
            self.current_config['analogue_gain'] = gain
            self.picam2.set_controls({"AnalogueGain": gain})

    def set_brightness(self, brightness: int):
        """Define brilho (-100 a 100%). Mapeado para -1.0 a 1.0 do Picamera2."""
        # Mapeia -100 a 100 para -1.0 a 1.0
        picam_brightness = brightness / 100.0
        picam_brightness = max(-1.0, min(1.0, picam_brightness))
        self.current_config['brightness'] = brightness
        if self.picam2 and self.is_running:
            self.picam2.set_controls({"Brightness": picam_brightness})

    def set_contrast(self, contrast: float):
        """Define contraste (0.0-32.0). Pratico para cartas: 1.0-4.0"""
        contrast = max(0.0, min(32.0, contrast))
        if self.picam2 and self.is_running:
            self.current_config['contrast'] = contrast
            self.picam2.set_controls({"Contrast": contrast})

    def set_saturation(self, saturation: float):
        """Define saturacao (0.0-32.0). Pratico para cartas: 0.8-2.0"""
        saturation = max(0.0, min(32.0, saturation))
        if self.picam2 and self.is_running:
            self.current_config['saturation'] = saturation
            self.picam2.set_controls({"Saturation": saturation})

    def set_sharpness(self, sharpness: float):
        """Define nitidez (0.0-16.0). Pratico para cartas: 1.0-4.0"""
        sharpness = max(0.0, min(16.0, sharpness))
        if self.picam2 and self.is_running:
            self.current_config['sharpness'] = sharpness
            self.picam2.set_controls({"Sharpness": sharpness})

    def set_temperature(self, temperature_k: int):
        """Define temperatura de cor em Kelvin (3000-8000K)."""
        self.current_config['temperature'] = temperature_k
        # Calcula ganhos RGB a partir da temperatura
        red_gain, blue_gain = temperature_to_rgb_gains(temperature_k)
        self.current_config['colour_gains'] = (red_gain, blue_gain)
        if self.picam2 and self.is_running:
            # Desativa AWB automatico quando temperatura manual e definida
            if not self.current_config['awb_enable']:
                self.picam2.set_controls({"ColourGains": (red_gain, blue_gain)})

    def set_tint(self, tint: float):
        """Define tonalidade (-150 a 150). Aplicado via software."""
        self.current_config['tint'] = tint

    def set_color_brightness(self, amount: float):
        """Define brilho da cor/vibrance (0.0-1.0). Aplicado via software."""
        self.current_config['color_brightness'] = amount

    def set_gamma(self, gamma: float):
        """Define correcao gama (0.01-3.5). Pratico: 0.7-1.3. Aplicado via software."""
        gamma = max(0.01, min(3.5, gamma))  # Min 0.01 para evitar divisao por zero
        self.current_config['gamma'] = gamma

    def set_ae_enable(self, enable: bool):
        """Ativa/desativa Auto Exposure"""
        if self.picam2 and self.is_running:
            self.current_config['ae_enable'] = enable
            self.picam2.set_controls({"AeEnable": enable})

    def set_awb_enable(self, enable: bool):
        """Ativa/desativa Auto White Balance"""
        if self.picam2 and self.is_running:
            self.current_config['awb_enable'] = enable
            self.picam2.set_controls({"AwbEnable": enable})

    def set_noise_reduction(self, mode: int):
        """Define modo de reducao de ruido (0=Off, 1=Fast, 2=HighQuality)"""
        if self.picam2 and self.is_running:
            self.current_config['noise_reduction'] = mode
            self.picam2.set_controls({"NoiseReductionMode": mode})

    def set_horizontal_flip(self, flip: bool):
        """Espelha horizontalmente (via software - Pi 5 nao suporta via hardware)"""
        self.current_config['horizontal_flip'] = flip

    def set_vertical_flip(self, flip: bool):
        """Espelha verticalmente (via software - Pi 5 nao suporta via hardware)"""
        self.current_config['vertical_flip'] = flip

    def _apply_flips(self, frame: np.ndarray) -> np.ndarray:
        """Aplica flips via software se configurados"""
        if frame is None:
            return None
        if self.current_config['horizontal_flip'] and self.current_config['vertical_flip']:
            return cv2.flip(frame, -1)  # Ambos
        elif self.current_config['horizontal_flip']:
            return cv2.flip(frame, 1)   # Horizontal
        elif self.current_config['vertical_flip']:
            return cv2.flip(frame, 0)   # Vertical
        return frame

    def set_colour_gains(self, red_gain: float, blue_gain: float):
        """Define ganhos de cor manual. Desativa AWB automaticamente."""
        red_gain = max(0.5, min(4.0, red_gain))
        blue_gain = max(0.5, min(4.0, blue_gain))
        self.current_config['colour_gains'] = (red_gain, blue_gain)
        if self.picam2 and self.is_running:
            # Desativa AWB antes de aplicar ganhos manuais
            if self.current_config['awb_enable']:
                self.current_config['awb_enable'] = False
                self.picam2.set_controls({"AwbEnable": False})
            self.picam2.set_controls({"ColourGains": (red_gain, blue_gain)})
            print(f"ColourGains aplicado: red={red_gain:.2f}, blue={blue_gain:.2f}")

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
                # Toggles
                'ae_enable': config['ae_enable'],
                'awb_enable': config['awb_enable'],
                'horizontal_flip': config['horizontal_flip'],
                'vertical_flip': config['vertical_flip'],
                # Exposicao (oficial: 0-999999us, pratico: 1000-100000us)
                'exposure_time': {'value': config['exposure_time'], 'min': 0, 'max': 999999, 'unit': 'us'},
                # ISO (mapeado para AnalogueGain 1.0-32.0)
                'iso': {'value': config['iso'], 'min': 100, 'max': 3200},
                # Balanco de branco
                'temperature': {'value': config['temperature'], 'min': 3000, 'max': 8000, 'unit': 'K'},
                'tint': {'value': config['tint'], 'min': -150, 'max': 150},
                'red_gain': {'value': config['colour_gains'][0] if config['colour_gains'][0] > 0 else 1.5, 'min': 0.5, 'max': 4.0},
                'blue_gain': {'value': config['colour_gains'][1] if config['colour_gains'][1] > 0 else 1.5, 'min': 0.5, 'max': 4.0},
                # Ajustes de imagem (ranges oficiais Picamera2)
                'brightness': {'value': config['brightness'], 'min': -100, 'max': 100, 'unit': '%'},
                'saturation': {'value': config['saturation'], 'min': 0.0, 'max': 32.0},
                'color_brightness': {'value': config['color_brightness'], 'min': -1.0, 'max': 1.0},
                'contrast': {'value': config['contrast'], 'min': 0.0, 'max': 32.0},
                'gamma': {'value': config['gamma'], 'min': 0.0, 'max': 3.5},
                'sharpness': {'value': config['sharpness'], 'min': 0.0, 'max': 16.0},
                # Outros
                'noise_reduction': config['noise_reduction']
            }
        return {'type': 'usb'}

    def set_camera_config(self, config: dict) -> bool:
        """Aplica configuracao na camera"""
        if self.active_camera == 'csi' and self.pi_camera:
            try:
                # Toggles
                if 'ae_enable' in config:
                    self.pi_camera.set_ae_enable(bool(config['ae_enable']))
                if 'awb_enable' in config:
                    self.pi_camera.set_awb_enable(bool(config['awb_enable']))
                if 'horizontal_flip' in config:
                    self.pi_camera.set_horizontal_flip(bool(config['horizontal_flip']))
                if 'vertical_flip' in config:
                    self.pi_camera.set_vertical_flip(bool(config['vertical_flip']))
                # Exposicao
                if 'exposure_time' in config:
                    self.pi_camera.set_exposure(int(config['exposure_time']))
                # ISO
                if 'iso' in config:
                    self.pi_camera.set_iso(int(config['iso']))
                # Balanco de branco
                if 'temperature' in config:
                    self.pi_camera.set_temperature(int(config['temperature']))
                if 'tint' in config:
                    self.pi_camera.set_tint(float(config['tint']))
                # Ganhos de cor (vermelho/azul)
                if 'red_gain' in config and 'blue_gain' in config:
                    self.pi_camera.set_colour_gains(float(config['red_gain']), float(config['blue_gain']))
                elif 'red_gain' in config:
                    current = self.pi_camera.current_config['colour_gains']
                    self.pi_camera.set_colour_gains(float(config['red_gain']), current[1] if current[1] > 0 else 1.5)
                elif 'blue_gain' in config:
                    current = self.pi_camera.current_config['colour_gains']
                    self.pi_camera.set_colour_gains(current[0] if current[0] > 0 else 1.5, float(config['blue_gain']))
                # Ajustes de imagem
                if 'brightness' in config:
                    self.pi_camera.set_brightness(int(config['brightness']))
                if 'saturation' in config:
                    self.pi_camera.set_saturation(float(config['saturation']))
                if 'color_brightness' in config:
                    self.pi_camera.set_color_brightness(float(config['color_brightness']))
                if 'contrast' in config:
                    self.pi_camera.set_contrast(float(config['contrast']))
                if 'gamma' in config:
                    self.pi_camera.set_gamma(float(config['gamma']))
                if 'sharpness' in config:
                    self.pi_camera.set_sharpness(float(config['sharpness']))
                # Outros
                if 'noise_reduction' in config:
                    self.pi_camera.set_noise_reduction(int(config['noise_reduction']))
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
    """Thread para captura continua - otimizado para stream apenas"""

    def __init__(self, camera_manager: UnifiedCameraManager, config: DetectorConfig):
        self.camera_manager = camera_manager
        self.config = config

        self.stream_frame = None
        self.stream_lock = threading.Lock()

        self.running = False
        self.thread = None

        # Controle de tempo para manter FPS estavel
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps

    def start(self):
        """Inicia thread de captura"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        """Loop de captura - apenas stream frame"""
        last_capture = time.perf_counter()

        while self.running:
            now = time.perf_counter()
            elapsed = now - last_capture

            # Captura apenas stream (HD e capturado sob demanda)
            stream_frame = self.camera_manager.read_stream()
            if stream_frame is not None:
                with self.stream_lock:
                    self.stream_frame = stream_frame

            last_capture = time.perf_counter()

            # Sleep adaptativo para manter FPS alvo
            process_time = last_capture - now
            sleep_time = self.frame_interval - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read_stream(self) -> Optional[np.ndarray]:
        """Le frame de stream"""
        with self.stream_lock:
            return self.stream_frame.copy() if self.stream_frame is not None else None

    def read_hd(self) -> Optional[np.ndarray]:
        """Le frame HD - captura sob demanda"""
        return self.camera_manager.read_hd()

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
    'has_detection': False,
    'exposure': {
        'brightness': 0.0,
        'overexposed_pct': 0.0,
        'underexposed_pct': 0.0,
        'status': 'unknown'
    },
    'quality': {
        'S_std': 0.0,       # Estabilidade
        'S_eff': 0.0,       # Qualidade real
        'sanity': 0.0,      # Sanidade visual
        'color_hash': 0.0,  # Hash
        'edge_density': 0.0 # OCR
    }
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
    frame_interval = 1.0 / 30  # 30 FPS alvo

    stream_fps = 0.0
    stream_fps_counter = 0
    stream_fps_start = time.perf_counter()

    while True:
        frame_start = time.perf_counter()

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
                stream_elapsed = time.perf_counter() - stream_fps_start
                if stream_elapsed > 0:
                    stream_fps = stream_fps_counter / stream_elapsed
                stream_fps_counter = 0
                stream_fps_start = time.perf_counter()

            camera_type = camera_manager.active_camera or 'none'

            # Analisa exposicao e qualidade a cada 45 frames (~1.5s a 30fps)
            exposure_info = None
            quality_info = None
            if frame_count % 45 == 0:
                exposure_info = analyze_exposure(frame)
                quality_info = analyze_image_quality(frame)

            frame = draw_detections(frame, detections, config)
            frame = draw_stats(frame, stream_fps, inference_time, len(detections), camera_type)

            with stats_lock:
                stats['fps'] = stream_fps
                stats['inference_time'] = inference_time
                stats['detections'] = len(detections)
                stats['has_detection'] = has_detection
                stats['camera_type'] = camera_type
                if exposure_info:
                    stats['exposure'] = exposure_info
                if quality_info:
                    stats['quality'] = quality_info

        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        # Controle de tempo para FPS estavel
        frame_time = time.perf_counter() - frame_start
        sleep_time = frame_interval - frame_time
        if sleep_time > 0:
            time.sleep(sleep_time)


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
