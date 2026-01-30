# src/threaded_camera.py
"""
Captura de camera em thread dedicada.
Baseado no padrao PyImageSearch/imutils para Raspberry Pi.
A thread de captura roda continuamente em background,
mantendo apenas o frame mais recente disponivel.
"""

import cv2
import threading
import time


class ThreadedCamera:
    """
    Camera com captura em thread separada.
    Elimina bloqueio de I/O no loop principal,
    aumentando o FPS de processamento.
    """

    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        """
        Args:
            camera_id: ID da camera (0 = primeira)
            width: Largura do frame
            height: Altura do frame
            fps: FPS desejado da camera
        """
        self.camera_id = camera_id
        self._fps = fps
        self.cap = None

        # Tentar multiplas formas de abrir a camera
        attempts = [
            (f"/dev/video{camera_id}", cv2.CAP_V4L2, "V4L2 por path"),
            (camera_id, cv2.CAP_V4L2, "V4L2 por indice"),
            (camera_id, cv2.CAP_ANY, "backend padrao"),
        ]

        for src, backend, desc in attempts:
            print(f"Tentando abrir camera: {desc} ({src})...")
            cap = cv2.VideoCapture(src, backend)
            if cap.isOpened():
                self.cap = cap
                print(f"Camera aberta via: {desc}")
                break
            cap.release()
            time.sleep(0.5)

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(
                f"Camera {camera_id} nao encontrada apos todas as tentativas. "
                "Execute: v4l2-ctl --list-devices"
            )

        # Forcar formato MJPG para reduzir banda USB (evita frames congelados)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Configurar camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Descartar primeiros frames (cameras USB costumam mandar lixo no inicio)
        for _ in range(5):
            self.cap.read()

        # Ler frame inicial
        self._grabbed, self._frame = self.cap.read()
        if not self._grabbed:
            raise RuntimeError("Camera aberta mas nao retornou frame")

        # Controle da thread
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = None
        self._error_count = 0

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera aberta: id={camera_id} {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    def start(self):
        """Inicia a thread de captura"""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stopped = False
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self):
        """Loop da thread de captura (roda em background)"""
        while not self._stopped:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self._error_count += 1
                    time.sleep(0.03)
                    continue

            except cv2.error:
                self._error_count += 1
                time.sleep(0.03)
                continue

            # Frame capturado com sucesso
            self._error_count = 0
            with self._lock:
                self._grabbed = True
                self._frame = frame

    def read(self):
        """
        Retorna o frame mais recente (thread-safe).
        Returns:
            (success, frame) - mesmo formato de cv2.VideoCapture.read()
        """
        with self._lock:
            return self._grabbed, self._frame.copy() if self._frame is not None else None

    def is_opened(self):
        """Verifica se a camera esta aberta"""
        return self.cap.isOpened()

    def stop(self):
        """Para a thread de captura e libera a camera"""
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.cap.release()
        print("Camera liberada")

    def release(self):
        """Alias para stop() - compatibilidade com cv2.VideoCapture"""
        self.stop()
