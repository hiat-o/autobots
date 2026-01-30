# src/detector_v1.py
"""
Detector de Faces - Versão 1 (Básica)
Objetivo: Funcionar e ser fácil de entender
"""

import cv2
import time

class FaceDetector:
    """Detector simples de faces usando Haar Cascade"""
    
    def __init__(self, camera_id=0):
        """
        Inicializa o detector
        
        Args:
            camera_id: ID da câmera (0 = primeira câmera)
        """
        print("🔧 Inicializando detector...")
        
        # Carregar classificador Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if self.face_cascade.empty():
            raise Exception("❌ Erro ao carregar Haar Cascade")
        
        # Abrir câmera
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise Exception(f"❌ Não consegui abrir câmera {camera_id}")
        
        print("✅ Detector inicializado!")
    
    def detect_faces(self, frame):
        """
        Detecta faces em um frame
        
        Args:
            frame: Imagem (numpy array)
        
        Returns:
            Lista de tuplas (x, y, w, h) para cada face
        """
        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,    # Quanto reduz imagem a cada escala
            minNeighbors=5,     # Quantos vizinhos para confirmar detecção
            minSize=(30, 30)    # Tamanho mínimo da face
        )
        
        return faces
    
    def draw_faces(self, frame, faces):
        """
        Desenha retângulos nas faces detectadas
        
        Args:
            frame: Imagem original
            faces: Lista de faces (x, y, w, h)
        
        Returns:
            Frame com retângulos desenhados
        """
        for (x, y, w, h) in faces:
            # Desenhar retângulo verde
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Adicionar texto com o tamanho da face
            text = f"{w}x{h}px"
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def run(self, show_fps=True):
        """
        Loop principal de detecção
        
        Args:
            show_fps: Se deve mostrar FPS na tela
        """
        print("🚀 Iniciando detecção (pressione 'q' para sair)...")
        
        # Variáveis para calcular FPS
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        while True:
            # Capturar frame
            ret, frame = self.cap.read()
            
            if not ret:
                print("❌ Erro ao capturar frame")
                break
            
            # Detectar faces
            faces = self.detect_faces(frame)
            
            # Desenhar faces
            frame = self.draw_faces(frame, faces)
            
            # Calcular FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Atualiza a cada 30 frames
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = time.time()
            
            # Adicionar informações na tela
            info_text = f"Faces: {len(faces)}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if show_fps:
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Mostrar frame
            cv2.imshow('Face Detector', frame)
            
            # Verificar se usuário apertou 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Encerrando...")
                break
        
        # Limpar recursos
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Detector encerrado")

def main():
    """Função principal"""
    try:
        detector = FaceDetector(camera_id=0)
        detector.run(show_fps=True)
    except Exception as e:
        print(f"❌ Erro: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())