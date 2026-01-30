# teste sync
# src/config.py
"""Configurações do projeto"""

import json
from pathlib import Path

class Config:
    """Gerenciador de configurações"""
    
    # Configurações padrão para Raspberry Pi 5 (8GB)
    DEFAULT_CONFIG = {
        "camera": {
            "id": 0,
            "width": 1280,
            "height": 720,
            "fps": 30
        },
        "detection": {
            "scale_factor": 1.1,
            "min_neighbors": 5,
            "min_size": [30, 30]
        },
        "display": {
            "show_fps": True,
            "show_face_size": True,
            "rectangle_color": [0, 255, 0],
            "rectangle_thickness": 2
        },
        "performance": {
            "frame_skip": 1,  # Processar 1 a cada N frames
            "resize_factor": 0.75,  # 0.75 = 75% da resolucao (RPi 5 suporta mais)
            "jpeg_quality": 85,  # Qualidade JPEG para streaming (1-100)
            "use_threading": True  # Captura em thread separada
        },
        "web": {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": False
        }
    }
    
    def __init__(self, config_path="config.json"):
        """
        Carrega configurações do arquivo ou usa padrões
        
        Args:
            config_path: Caminho do arquivo de configuração
        """
        self.config_path = Path(config_path)
        
        if self.config_path.exists():
            self.load()
        else:
            print(f"⚠️  Arquivo {config_path} não encontrado, usando configurações padrão")
            self.config = self.DEFAULT_CONFIG.copy()
            self.save()  # Criar arquivo com padrões
    
    def _deep_merge(self, base, override):
        """Merge recursivo: base fornece defaults, override sobrescreve"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def load(self):
        """Carrega configuracoes do arquivo JSON, com fallback para defaults"""
        with open(self.config_path, 'r') as f:
            loaded = json.load(f)
        # Merge: DEFAULT_CONFIG fornece chaves faltantes, loaded sobrescreve
        self.config = self._deep_merge(self.DEFAULT_CONFIG, loaded)
        print(f"Configuracoes carregadas de {self.config_path}")
    
    def save(self):
        """Salva configurações no arquivo JSON"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"💾 Configurações salvas em {self.config_path}")
    
    def get(self, *keys):
        """
        Acessa configuração por caminho
        
        Exemplo:
            config.get('camera', 'width')  # Retorna 640
        """
        value = self.config
        for key in keys:
            value = value[key]
        return value
    
    def set(self, *keys, value):
        """
        Define configuração por caminho
        
        Exemplo:
            config.set('camera', 'width', value=320)
        """
        target = self.config
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value