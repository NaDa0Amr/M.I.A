import torch
import torch.nn as nn
from .encoder import ImageEncoder
from .decoder import CaptionDecoder

class ImageCaptioner(nn.Module):
    """
    End-to-end image captioner combining encoder and decoder.
    
    During training, uses precomputed 2048-d features.
    The encoder's projection layer is trainable.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, vocab_size: int, num_layers: int = 1, dropout: float = 0.5, encoder_name: str = 'resnet50'):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim=embed_dim, model_name=encoder_name)
        self.decoder = CaptionDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.embed_dim = embed_dim
    
    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Training forward pass with precomputed features."""
        projected = self.encoder(features)  # (batch, embed_dim)
        logits = self.decoder(projected, captions)  # (batch, seq_len, vocab_size)
        return logits
    
    @classmethod
    def from_config(cls, config: dict, vocab_size: int) -> 'ImageCaptioner':
        """Create model from config dict."""
        model_cfg = config['model']
        return cls(
            embed_dim=model_cfg['embed_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            vocab_size=vocab_size,
            num_layers=model_cfg.get('num_layers', 1),
            dropout=model_cfg.get('dropout', 0.5),
            encoder_name=model_cfg.get('encoder', 'resnet50')
        )
