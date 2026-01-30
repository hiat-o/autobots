# src/detector_v2.py
"""
Detector de Faces - Versao 2 (Com Configuracao + Threading)
"""

import cv2
import os
import time
from config import Config
from threaded_camera import ThreadedCamera

# Diretorio base do projeto (pai de src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_haarcascade():
    """Encontra o arquivo haarcascade em diferentes instalacoes do OpenCV"""
    cascade_file = 'haarcascade_frontalface_default.xml'

    if hasattr(cv2, 'data'):
        return cv2.data.haarcascades + cascade_file

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


class FaceDetector:
    """Detector de faces configuravel com captura em thread separada"""

    def __init__(self, config_path="config.json"):
        print("Inicializando detector...")

        # Carregar configuracoes
        self.config = Config(config_path)

        # Carregar Haar Cascade
        cascade_path = find_haarcascade()
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        print(f"Haar cascade: {cascade_path}")

        # Abrir camera
        camera_id = self.config.get('camera', 'id')
        use_threading = self.config.get('performance', 'use_threading')

        if use_threading:
            self.cap = ThreadedCamera(
                camera_id=camera_id,
                width=self.config.get('camera', 'width'),
                height=self.config.get('camera', 'height'),
                fps=self.config.get('camera', 'fps')
            )
            self.cap.start()
            self.use_threading = True
        else:
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('camera', 'width'))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('camera', 'height'))
            self.cap.set(cv2.CAP_PROP_FPS, self.config.get('camera', 'fps'))
            self.use_threading = False

        # Variaveis de controle
        self.frame_count = 0
        self.frame_skip = self.config.get('performance', 'frame_skip')

        print("Detector inicializado!")
        self._print_config()

    def _print_config(self):
        """Imprime configuracoes atuais"""
        w = self.config.get('camera', 'width')
        h = self.config.get('camera', 'height')
        fps = self.config.get('camera', 'fps')
        print(f"\nConfiguracoes:")
        print(f"  Camera: {w}x{h} @ {fps}fps")
        print(f"  Frame Skip: 1/{self.config.get('performance', 'frame_skip')}")
        print(f"  Resize Factor: {self.config.get('performance', 'resize_factor')}")
        print(f"  Threading: {'ATIVO' if self.use_threading else 'DESATIVADO'}")
        print()

    def detect_faces(self, frame):
        """Detecta faces no frame"""
        resize_factor = self.config.get('performance', 'resize_factor')
        detect_frame = frame

        if resize_factor != 1.0:
            width = int(frame.shape[1] * resize_factor)
            height = int(frame.shape[0] * resize_factor)
            detect_frame = cv2.resize(frame, (width, height))

        gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.get('detection', 'scale_factor'),
            minNeighbors=self.config.get('detection', 'min_neighbors'),
            minSize=tuple(self.config.get('detection', 'min_size'))
        )

        # Ajustar coordenadas se houve resize
        if resize_factor != 1.0 and len(faces) > 0:
            faces = [(int(x/resize_factor), int(y/resize_factor),
                     int(w/resize_factor), int(h/resize_factor))
                    for (x, y, w, h) in faces]

        return faces

    def draw_faces(self, frame, faces):
        """Desenha retangulos nas faces"""
        color = tuple(self.config.get('display', 'rectangle_color'))
        thickness = self.config.get('display', 'rectangle_thickness')

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

            if self.config.get('display', 'show_face_size'):
                text = f"{w}x{h}px"
                cv2.putText(frame, text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def run(self):
        """Loop principal"""
        print("Iniciando deteccao (pressione 'q' para sair)...")

        start_time = time.time()
        processed_frames = 0
        fps = 0
        last_faces = []

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            self.frame_count += 1

            # Frame skipping
            if self.frame_count % self.frame_skip == 0:
                last_faces = self.detect_faces(frame)
                processed_frames += 1

                # Calcular FPS a cada 10 frames processados
                if processed_frames % 10 == 0:
                    end_time = time.time()
                    elapsed = end_time - start_time
                    if elapsed > 0:
                        fps = 10.0 / elapsed
                    start_time = time.time()

            # Desenhar faces (mesmo em frames nao processados, usa ultima deteccao)
            frame = self.draw_faces(frame, last_faces)

            # Info na tela
            cv2.putText(frame, f"Faces: {len(last_faces)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if self.config.get('display', 'show_fps'):
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Face Detector v2', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if self.use_threading:
            self.cap.stop()
        else:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Detector encerrado")

def main():
    config_path = os.path.join(BASE_DIR, "config.json")
    try:
        detector = FaceDetector(config_path)
        detector.run()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario")
    except Exception as e:
        print(f"Erro: {e}")
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
