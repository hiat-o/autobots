# src/test_camera.py
import cv2
import sys

def test_camera():
    """Testa se a câmera está funcionando"""
    print("🎥 Tentando abrir câmera...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro: Não consegui abrir a câmera")
        print("Verifique se:")
        print("  - A câmera está conectada")
        print("  - Permissões estão OK")
        return False
    
    print("✅ Câmera aberta com sucesso!")
    
    # Capturar um frame de teste
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Erro: Não consegui capturar frame")
        cap.release()
        return False
    
    height, width, channels = frame.shape
    print(f"📐 Resolução: {width}x{height}")
    print(f"🎨 Canais: {channels}")
    
    # Mostrar por 3 segundos
    print("👀 Mostrando preview (pressione ESC para fechar)...")
    cv2.imshow('Test Camera', frame)
    cv2.waitKey(3000)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("✅ Teste completo!")
    return True

if __name__ == "__main__":
    success = test_camera()
    sys.exit(0 if success else 1)