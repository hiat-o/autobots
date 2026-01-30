# src/web_detector.py
"""
Detector de Faces via Web - Streaming MJPEG com Flask
Acesse http://<ip-do-raspberry>:5000 no navegador

Arquitetura com Threading para Raspberry Pi 5:
  - Thread 1: Captura de frames (nao bloqueia)
  - Thread 2: Deteccao de faces (roda em paralelo)
  - Thread principal: Streaming MJPEG
"""

import cv2
import os
import signal
import sys
import time
import threading
from flask import Flask, Response, render_template
from config import Config

# Diretorio base do projeto (pai de src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_cascade(use_lbp=True):
    """Encontra o arquivo cascade em diferentes instalacoes do OpenCV"""
    if use_lbp:
        cascades_to_try = [
            ('lbpcascade_frontalface.xml', 'lbpcascades', 'LBP'),
            ('haarcascade_frontalface_default.xml', 'haarcascades', 'Haar'),
        ]
    else:
        cascades_to_try = [
            ('haarcascade_frontalface_default.xml', 'haarcascades', 'Haar'),
        ]

    for cascade_file, cascade_dir, cascade_type in cascades_to_try:
        # 1. Tenta pasta local do projeto (cascades/)
        local_path = os.path.join(BASE_DIR, 'cascades', cascade_file)
        if os.path.exists(local_path):
            return local_path, cascade_type

        # 2. Tenta cv2.data (pip install)
        if hasattr(cv2, 'data'):
            cascade_path = os.path.join(
                os.path.dirname(cv2.data.haarcascades),
                cascade_dir,
                cascade_file
            )
            if os.path.exists(cascade_path):
                return cascade_path, cascade_type

            cascade_path = cv2.data.haarcascades + cascade_file
            if os.path.exists(cascade_path):
                return cascade_path, cascade_type

        # 3. Caminhos comuns no Raspberry Pi
        system_paths = [
            f'/usr/share/opencv4/{cascade_dir}/',
            f'/usr/share/opencv/{cascade_dir}/',
            f'/usr/local/share/opencv4/{cascade_dir}/',
            '/usr/share/opencv4/haarcascades/',
            '/usr/share/opencv/haarcascades/',
        ]
        for path in system_paths:
            full_path = os.path.join(path, cascade_file)
            if os.path.exists(full_path):
                return full_path, cascade_type

    raise FileNotFoundError("Nenhum cascade encontrado (LBP ou Haar)")


class CameraThread:
    """Thread para captura continua de frames"""

    def __init__(self, camera_id, width, height, fps):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps

        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.cap = None

    def start(self):
        """Inicia a thread de captura"""
        self.cap = self._open_camera()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _open_camera(self):
        """Abre a camera USB com V4L2"""
        attempts = [
            (f"/dev/video{self.camera_id}", cv2.CAP_V4L2, "V4L2 por path"),
            (self.camera_id, cv2.CAP_V4L2, "V4L2 por indice"),
            (self.camera_id, cv2.CAP_ANY, "backend padrao"),
        ]

        cap = None
        for src, backend, desc in attempts:
            print(f"Tentando abrir camera: {desc} ({src})...")
            cap = cv2.VideoCapture(src, backend)
            if cap.isOpened():
                print(f"Camera aberta via: {desc}")
                break
            cap.release()
            cap = None
            time.sleep(0.3)

        if cap is None:
            raise RuntimeError(f"Camera {self.camera_id} nao encontrada.")

        # Configurar camera
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Descartar primeiros frames
        for _ in range(5):
            cap.read()

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

        return cap

    def _capture_loop(self):
        """Loop de captura em thread separada"""
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame

    def read(self):
        """Retorna o ultimo frame capturado (thread-safe)"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Para a thread e libera a camera"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


class DetectorThread:
    """Thread para deteccao de faces em paralelo"""

    def __init__(self, cascade, config):
        self.cascade = cascade
        self.config = config

        self.faces = []
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

    def submit_frame(self, frame):
        """Envia um frame para processamento"""
        with self.input_lock:
            self.input_frame = frame
        self.new_frame_event.set()

    def _detection_loop(self):
        """Loop de deteccao em thread separada"""
        fps_counter = 0
        fps_start = time.time()

        detect_width = self.config.get('performance', 'detect_width')
        detect_height = self.config.get('performance', 'detect_height')
        scale_factor = self.config.get('detection', 'scale_factor')
        min_neighbors = self.config.get('detection', 'min_neighbors')
        min_size = tuple(self.config.get('detection', 'min_size'))

        while self.running:
            # Espera novo frame (com timeout para poder sair)
            if not self.new_frame_event.wait(timeout=0.1):
                continue
            self.new_frame_event.clear()

            # Pega o frame
            with self.input_lock:
                frame = self.input_frame
                self.input_frame = None

            if frame is None:
                continue

            # Processa
            original_h, original_w = frame.shape[:2]
            detect_frame = cv2.resize(frame, (detect_width, detect_height))
            gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)

            # Ajustar min_size
            scale = detect_width / original_w
            scaled_min_size = (max(10, int(min_size[0] * scale)),
                               max(10, int(min_size[1] * scale)))

            detected = self.cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=scaled_min_size
            )

            # Escalar coordenadas de volta
            faces = []
            if len(detected) > 0:
                scale_x = original_w / detect_width
                scale_y = original_h / detect_height
                faces = [(int(x * scale_x), int(y * scale_y),
                          int(w * scale_x), int(h * scale_y))
                         for (x, y, w, h) in detected]

            # Atualiza resultado (thread-safe)
            with self.lock:
                self.faces = faces

            # Calcula FPS
            fps_counter += 1
            if fps_counter >= 10:
                elapsed = time.time() - fps_start
                if elapsed > 0:
                    with self.lock:
                        self.fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

    def get_results(self):
        """Retorna faces detectadas e FPS"""
        with self.lock:
            return list(self.faces), self.fps

    def stop(self):
        """Para a thread"""
        self.running = False
        self.new_frame_event.set()  # Desbloqueia se estiver esperando
        if self.thread is not None:
            self.thread.join(timeout=1.0)


# --- Configuracao global ---

config_path = os.path.join(BASE_DIR, "config.json")
config = Config(config_path)

# Face Cascade
use_lbp = config.get('detection', 'use_lbp')
cascade_path, cascade_type = find_cascade(use_lbp=use_lbp)
face_cascade = cv2.CascadeClassifier(cascade_path)
print(f"{cascade_type} cascade: {cascade_path}")

# Camera Thread
camera_thread = CameraThread(
    camera_id=config.get('camera', 'id'),
    width=config.get('camera', 'width'),
    height=config.get('camera', 'height'),
    fps=config.get('camera', 'fps')
).start()

# Detector Thread
detector_thread = DetectorThread(face_cascade, config).start()


def draw_overlay(frame, faces, fps):
    """Desenha retangulos e info no frame"""
    color = tuple(config.get('display', 'rectangle_color'))
    thickness = config.get('display', 'rectangle_thickness')

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        if config.get('display', 'show_face_size'):
            cv2.putText(frame, f"{w}x{h}px", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if config.get('display', 'show_fps'):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def generate_frames():
    """Gerador de frames para streaming MJPEG com controle de latencia"""
    jpeg_quality = config.get('performance', 'jpeg_quality')
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    frame_skip = config.get('performance', 'frame_skip')
    frame_count = 0

    # Controle de taxa para reduzir latencia
    target_fps = 30
    frame_time = 1.0 / target_fps
    last_frame_time = time.time()

    # FPS do streaming (diferente do FPS de deteccao)
    stream_fps = 0.0
    stream_fps_counter = 0
    stream_fps_start = time.time()

    while True:
        # Controle de taxa - evita enviar frames muito rapido
        now = time.time()
        elapsed = now - last_frame_time
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)
        last_frame_time = time.time()

        # Pega frame da camera (nao bloqueia)
        frame = camera_thread.read()
        if frame is None:
            continue

        frame_count += 1

        # Envia para deteccao (a cada N frames)
        if frame_count % frame_skip == 0:
            detector_thread.submit_frame(frame.copy())

        # Pega resultado da deteccao (nao bloqueia)
        faces, detect_fps = detector_thread.get_results()

        # Calcula FPS do streaming
        stream_fps_counter += 1
        if stream_fps_counter >= 30:
            stream_elapsed = time.time() - stream_fps_start
            if stream_elapsed > 0:
                stream_fps = stream_fps_counter / stream_elapsed
            stream_fps_counter = 0
            stream_fps_start = time.time()

        # Desenha overlay (mostra FPS do streaming)
        draw_overlay(frame, faces, stream_fps)

        # Codifica e envia
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# --- Flask App ---

app = Flask(__name__)


@app.route('/')
def index():
    """Pagina principal"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Endpoint de streaming MJPEG com headers anti-buffering"""
    response = Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    # Headers para reduzir latencia/buffering
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


def cleanup():
    """Libera recursos"""
    detector_thread.stop()
    camera_thread.stop()
    print("Recursos liberados")


def main():
    host = config.get('web', 'host')
    port = config.get('web', 'port')
    debug = config.get('web', 'debug')

    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))

    print(f"\nServidor web iniciado em http://{host}:{port}")
    print(f"Acesse de outro dispositivo na rede local")
    print(f"Deteccao em: {config.get('performance', 'detect_width')}x{config.get('performance', 'detect_height')}")
    print(f"JPEG quality: {config.get('performance', 'jpeg_quality')}")
    print(f"Frame skip: {config.get('performance', 'frame_skip')}")
    print(f"Modo: Threading (captura + deteccao em paralelo)")
    print("Pressione Ctrl+C para encerrar\n")

    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario")
    finally:
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
