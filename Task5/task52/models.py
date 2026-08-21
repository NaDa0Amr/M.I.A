import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights, mobilenet_v2, MobileNet_V2_Weights

def build_resnet34(num_classes: int, device: torch.device) -> nn.Module:
    """Build and modify ResNet34 for fine-tuning."""
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    
    # Freeze ALL layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # Replace fc
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    model = model.to(device)
    return model

def build_mobilenetv2(num_classes: int, device: torch.device) -> nn.Module:
    """Build and modify MobileNetV2 for fine-tuning."""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    
    # Freeze ALL layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze features[18]
    for param in model.features[18].parameters():
        param.requires_grad = True
        
    # Replace classifier[1]
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    model = model.to(device)
    return model
