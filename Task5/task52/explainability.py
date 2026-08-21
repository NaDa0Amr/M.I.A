import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F

def compute_saliency(model: nn.Module, image: torch.Tensor, target_class: int, device: torch.device) -> np.ndarray:
    """
    Vanilla gradient saliency map.
    1. Set model to eval mode
    2. Clone image, enable gradients on it
    3. Forward pass, get target class score (not softmax, raw logit)
    4. Backward pass
    5. Get absolute value of gradient, take max across channels
    6. Normalize to [0,1]
    Returns: HxW numpy array
    """
    model.eval()
    
    image = image.clone().detach().to(device)
    image.requires_grad_(True)
    
    output = model(image.unsqueeze(0) if image.dim() == 3 else image)
    
    score = output[0, target_class]
    model.zero_grad()
    score.backward()
    
    # Get absolute value of gradient, take max across channels
    saliency, _ = torch.max(image.grad.data.abs(), dim=0 if image.dim() == 3 else 1)
    if saliency.dim() == 3:
        saliency = saliency.squeeze(0)
        
    # Normalize to [0,1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency.cpu().numpy()


def compute_gradcam(model: nn.Module, image: torch.Tensor, target_class: int, target_layer: nn.Module, device: torch.device) -> np.ndarray:
    """
    Gradient-weighted Class Activation Mapping.
    1. Register forward hook on target_layer to capture activations
    2. Register backward hook on target_layer to capture gradients
    3. Forward pass, backprop target class score
    4. Global average pool gradients over spatial dims -> channel weights
    5. Weighted sum of activations * weights -> ReLU -> normalize to [0,1]
    6. Resize to input image size (224x224) using bilinear interpolation
    7. Remove hooks!
    Returns: HxW numpy array in [0,1]
    """
    model.eval()
    
    activations = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()
        
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()
        
    # 1. & 2. Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    
    image = image.clone().detach().to(device)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image.requires_grad_(True)
    
    # 3. Forward pass, backprop target class score
    output = model(image)
    score = output[0, target_class]
    model.zero_grad()
    score.backward()
    
    # 4. Global average pool gradients over spatial dims -> channel weights
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    
    # 5. Weighted sum of activations * weights -> ReLU -> normalize to [0,1]
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)
    
    # 6. Resize to input image size (224x224) using bilinear interpolation
    cam = F.interpolate(cam, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False)
    
    cam = cam.squeeze()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # 7. Remove hooks!
    handle_forward.remove()
    handle_backward.remove()
    
    return cam.cpu().numpy()


def overlay_heatmap(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap=cm.jet) -> np.ndarray:
    """
    Overlay a heatmap on an image using matplotlib colormap.
    image_np: HxWx3 in [0,1]
    heatmap: HxW in [0,1]
    Returns: HxWx3 in [0,1]
    """
    # Colorize heatmap
    heatmap_colored = colormap(heatmap)[:, :, :3]  # Drop alpha channel from colormap if present
    
    # Overlay
    overlaid = (1.0 - alpha) * image_np + alpha * heatmap_colored
    overlaid = np.clip(overlaid, 0, 1)
    
    return overlaid
