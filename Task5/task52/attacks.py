import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config


def clamp_to_valid_image(x_norm: torch.Tensor) -> torch.Tensor:

    mean = torch.tensor(config.IMAGENET_MEAN, device=x_norm.device).view(1, -1, 1, 1)
    std = torch.tensor(config.IMAGENET_STD, device=x_norm.device).view(1, -1, 1, 1)
    x_pixel = torch.clamp(x_norm * std + mean, 0.0, 1.0)
    return (x_pixel - mean) / std


def _project(x: torch.Tensor) -> torch.Tensor:
    """Clamp an adversarial batch to the configured valid range."""
    if getattr(config, 'CLAMP_VALID_RANGE', False):
        return clamp_to_valid_image(x)
    return torch.clamp(x, -3.0, 3.0)



def fgsm_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epsilon: float, criterion, device: torch.device) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) - single-step white-box attack.
    x_adv = x + epsilon * sign(grad_x L(theta, x, y))
    
    Args:
        model: Target model (must be in eval mode externally, but we need gradients)
        images: Batch of input images (BxCxHxW), already normalized
        labels: True labels
        epsilon: Perturbation magnitude (in normalized space)
        criterion: Loss function (CrossEntropyLoss)
        device: Computation device
    Returns:
        Adversarial images (detached, clamped to valid range)
    """
    # Clone images and set requires_grad_(True)
    images_adv = images.clone().detach().to(device)
    images_adv.requires_grad_(True)
    
    # Forward pass, compute loss, backward
    outputs = model(images_adv)
    loss = criterion(outputs, labels.to(device))
    
    model.zero_grad()
    loss.backward()
    
    # Compute sign of gradient
    data_grad = images_adv.grad.data
    sign_data_grad = data_grad.sign()
    
    # Add epsilon * sign to original images
    perturbed_images = images_adv + epsilon * sign_data_grad
    
    # Clamp result to a reasonable range for normalized ImageNet images
    perturbed_images = _project(perturbed_images)
    
    return perturbed_images.detach()


def pgd_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epsilon: float, criterion, device: torch.device, alpha: float = 0.01, num_steps: int = 7) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) - iterative white-box attack.
    x_0 = x + uniform(-epsilon, epsilon)
    x_{t+1} = Proj_{B_eps(x)}(x_t + alpha * sign(grad L))
    
    Args: 
        model: Target model
        images: Batch of input images
        labels: True labels
        epsilon: Maximum perturbation magnitude
        alpha: Step size
        num_steps: Number of iterations
        criterion: Loss function
        device: Computation device
    Returns: 
        Adversarial images (detached, clamped)
    """
    original_images = images.clone().detach().to(device)
    
    # Start with random perturbation within epsilon ball
    perturbed_images = original_images + torch.empty_like(original_images).uniform_(-epsilon, epsilon)
    perturbed_images = _project(perturbed_images).detach()
    
    for _ in range(num_steps):
        perturbed_images.requires_grad_(True)
        
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels.to(device))
        
        model.zero_grad()
        loss.backward()
        
        adv_images = perturbed_images + alpha * perturbed_images.grad.sign()
        
        # Project back to epsilon ball
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        perturbed_images = _project(original_images + eta).detach()
        
    return perturbed_images


def evaluate_under_attack(model: nn.Module, loader: DataLoader, attack_fn, epsilon: float, criterion, device: torch.device, **attack_kwargs) -> float:
    """
    Evaluate model accuracy under a given attack across the full dataset.
    Returns: accuracy (float)
    """
    correct = 0
    total = 0
    
    model.eval()
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        if attack_fn is not None and epsilon > 0:
            adv_images = attack_fn(model, images, labels, epsilon=epsilon, criterion=criterion, device=device, **attack_kwargs)
        else:
            adv_images = images
            
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total if total > 0 else 0.0


def epsilon_sweep(model: nn.Module, loader: DataLoader, attack_fn, epsilons: list, criterion, device: torch.device, **attack_kwargs) -> dict:
    """
    Sweep across epsilon values and return {epsilon: accuracy} dict.
    Print progress.
    """
    results = {}
    print(f"Starting epsilon sweep for {len(epsilons)} values...")
    
    for i, eps in enumerate(epsilons):
        accuracy = evaluate_under_attack(model, loader, attack_fn, eps, criterion, device, **attack_kwargs)
        results[eps] = accuracy
        print(f"  [{i+1}/{len(epsilons)}] Epsilon: {eps:.4f} - Accuracy: {accuracy:.4f}")
        
    return results
