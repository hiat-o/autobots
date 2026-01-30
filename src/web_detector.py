# src/web_detector.py
"""
Detector de Faces via Web - Streaming MJPEG com Flask
Acesse http://<ip-do-raspberry>:5000 no navegador

Arquitetura otimizada para Raspberry Pi 5 (8GB):
  - Fluxo unico: captura -> deteccao -> streaming
  - Suporte a resolucoes maiores (1280x720, 1920x1080)
  - Melhor qualidade de imagem com hardware mais potente
"""

import cv2
import os
import signal
import sys
import time
from flask import Flask, Response, render_template
from config import Config

# Diretorio base do projeto (pai de src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_haarcascade():
    """Encontra o arquivo haarcascade em diferentes instalacoes do OpenCV"""
    cascade_file = 'haarcascade_frontalface_default.xml'

    # Tenta cv2.data (pip install)
    if hasattr(cv2, 'data'):
        return cv2.data.haarcascades + cascade_file

    # Caminhos comuns no Raspberry Pi (apt install)
    system_paths = [
        '/usr/share/opencv4/haarcascades/',
        '/usr/share/opencv/haarcascades/',
        '/usr/local/share/opencv4/haarcascades/',
    ]
    for path in system_paths:
        full_path = os.path.join(path, cascade_file)
        if os.path.exists(full_path):
            return full_path

    raise FileNotFoundError(f"Haar cascade nao encontrado: {cascade_file}")


def open_camera(camera_id, width, height, fps):
    """Abre a camera USB com V4L2, configurando MJPG para reduzir banda USB"""
    attempts = [
        (f"/dev/video{camera_id}", cv2.CAP_V4L2, "V4L2 por path"),
        (camera_id, cv2.CAP_V4L2, "V4L2 por indice"),
        (camera_id, cv2.CAP_ANY, "backend padrao"),
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
        time.sleep(0.5)

    if cap is None:
        raise RuntimeError(
            f"Camera {camera_id} nao encontrada. "
            "Execute: v4l2-ctl --list-devices"
        )

    # MJPG reduz banda USB e evita frames congelados
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Descartar primeiros frames (cameras USB costumam mandar lixo no inicio)
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Camera aberta mas nao retornou frame")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera aberta: id={camera_id} {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    return cap


# --- Configuracao global ---

config_path = os.path.join(BASE_DIR, "config.json")
config = Config(config_path)

# Haar Cascade
cascade_path = find_haarcascade()
face_cascade = cv2.CascadeClassifier(cascade_path)
print(f"Haar cascade: {cascade_path}")

# Camera
camera = open_camera(
    camera_id=config.get('camera', 'id'),
    width=config.get('camera', 'width'),
    height=config.get('camera', 'height'),
    fps=config.get('camera', 'fps')
)


# --- Deteccao e Streaming (fluxo unico) ---

def detect_faces(frame):
    """Detecta faces no frame e retorna lista de coordenadas"""
    resize_factor = config.get('performance', 'resize_factor')
    detect_frame = frame

    if resize_factor != 1.0:
        w = int(frame.shape[1] * resize_factor)
        h = int(frame.shape[0] * resize_factor)
        detect_frame = cv2.resize(frame, (w, h))

    gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=config.get('detection', 'scale_factor'),
        minNeighbors=config.get('detection', 'min_neighbors'),
        minSize=tuple(config.get('detection', 'min_size'))
    )

    # Ajustar coordenadas se houve resize
    if resize_factor != 1.0 and len(faces) > 0:
        faces = [(int(x/resize_factor), int(y/resize_factor),
                 int(w/resize_factor), int(h/resize_factor))
                for (x, y, w, h) in faces]

    return faces


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
    """Gerador de frames: captura -> deteccao -> encode -> streaming"""
    jpeg_quality = config.get('performance', 'jpeg_quality')
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    frame_skip = config.get('performance', 'frame_skip')

    frame_count = 0
    fps = 0.0
    fps_counter = 0
    fps_start = time.time()
    last_faces = []

    while True:
        # 1. Capturar frame
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1

        # 2. Deteccao (a cada N frames para poupar CPU)
        if frame_count % frame_skip == 0:
            last_faces = detect_faces(frame)
            fps_counter += 1

            # Calcular FPS real
            if fps_counter >= 10:
                elapsed = time.time() - fps_start
                if elapsed > 0:
                    fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

        # 3. Desenhar overlay
        draw_overlay(frame, last_faces, fps)

        # 4. Codificar JPEG e enviar
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
    """Endpoint de streaming MJPEG"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def cleanup():
    """Libera a camera de forma segura"""
    camera.release()
    print("Camera liberada")


def main():
    host = config.get('web', 'host')
    port = config.get('web', 'port')
    debug = config.get('web', 'debug')

    # Garantir liberacao da camera em qualquer saida
    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))

    print(f"\nServidor web iniciado em http://{host}:{port}")
    print(f"Acesse de outro dispositivo na rede local")
    print(f"Resize factor: {config.get('performance', 'resize_factor')}")
    print(f"JPEG quality: {config.get('performance', 'jpeg_quality')}")
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
