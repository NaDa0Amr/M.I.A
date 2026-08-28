import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    """
    CNN-based image feature extractor using pretrained ResNet-50.
    
    Two modes:
    1. Raw feature extraction (for offline caching): outputs 2048-d vector
    2. Full encoder (for end-to-end training): projects 2048 -> embed_dim
    """
    
    SUPPORTED_MODELS = {'resnet50', 'resnet101'}
    
    def __init__(self, embed_dim: int = 256, model_name: str = 'resnet50'):
        super().__init__()
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {self.SUPPORTED_MODELS}")
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        
        # Load pretrained CNN and remove final FC layer
        if model_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        
        # Keep everything except the final FC layer
        # Output: (batch, 2048, 1, 1) after adaptive avg pool
        modules = list(resnet.children())[:-1]  # Remove final FC
        self.cnn = nn.Sequential(*modules)
        
        # Freeze all CNN parameters (transfer learning)
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Learnable projection layer: 2048 -> embed_dim
        self.projection = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract raw 2048-d features without projection. Used for offline caching."""
        self.cnn.eval()
        with torch.no_grad():
            features = self.cnn(images)  # (batch, 2048, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # (batch, 2048)
        return features
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Project precomputed 2048-d features to embed_dim.
        Input: (batch, 2048)
        Output: (batch, embed_dim)
        """
        return self.projection(features)
