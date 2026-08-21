import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List
import config

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility (torch, numpy, random)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def denormalize(tensor: torch.Tensor, mean: Optional[List[float]] = None, std: Optional[List[float]] = None) -> torch.Tensor:
    """Reverse ImageNet normalization, return tensor in [0,1]."""
    if mean is None:
        mean = config.IMAGENET_MEAN
    if std is None:
        std = config.IMAGENET_STD
        
    mean_tensor = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std_tensor = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    
    # Broadcast and reverse normalize
    tensor = tensor * std_tensor + mean_tensor
    return torch.clamp(tensor, 0.0, 1.0)

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert CxHxW tensor to HxWxC numpy in [0,1] (denormalized)."""
    tensor = denormalize(tensor).detach().cpu()
    # Handle batch dimension if present (assume 1st image)
    if tensor.dim() == 4:
        tensor = tensor[0]
    return tensor.permute(1, 2, 0).numpy()

def save_model(model: nn.Module, path: str) -> None:
    """Save model state dict."""
    torch.save(model.state_dict(), path)

def load_model(model: nn.Module, path: str) -> nn.Module:
    """Load model state dict, return model."""
    model.load_state_dict(torch.load(path, map_location=next(model.parameters()).device, weights_only=True))
    return model
