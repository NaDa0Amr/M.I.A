import yaml
import torch
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device(config: dict) -> torch.device:
    """Get the device from config. Safely falls back to CPU if CUDA is requested but unavailable."""
    device_str = config.get('training', config.get('model', {})).get('device', 'auto')
    if device_str == 'auto' or device_str == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif device_str == 'cuda':
            print("⚠ Warning: 'cuda' was requested but this PyTorch build is CPU-only. Falling back to 'cpu'.")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
            
    return torch.device(device_str)
