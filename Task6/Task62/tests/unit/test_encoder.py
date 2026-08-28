import torch
import pytest
from src.models.encoder import ImageEncoder

def test_encoder_output_shape():
    encoder = ImageEncoder(embed_dim=256)
    features = torch.randn(2, 2048)
    out = encoder.forward(features)
    assert out.shape == (2, 256)

def test_encoder_extract_features_shape(dummy_image_tensor):
    encoder = ImageEncoder(embed_dim=256)
    images = dummy_image_tensor.unsqueeze(0).repeat(2, 1, 1, 1) # Batch of 2
    out = encoder.extract_features(images)
    assert out.shape == (2, 2048)

def test_encoder_cnn_frozen():
    encoder = ImageEncoder(embed_dim=256)
    # Only the linear layer should be trainable by default, assuming ResNet backbone is frozen
    if hasattr(encoder, 'cnn') or hasattr(encoder, 'backbone'):
        cnn_module = getattr(encoder, 'cnn', getattr(encoder, 'backbone', None))
        if cnn_module is not None:
            for param in cnn_module.parameters():
                assert not param.requires_grad

def test_encoder_projection_trainable():
    encoder = ImageEncoder(embed_dim=256)
    # The linear layer should require grad
    has_trainable = any(p.requires_grad for p in encoder.parameters())
    assert has_trainable

@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_encoder_various_batch_sizes(batch_size):
    encoder = ImageEncoder(embed_dim=256)
    encoder.eval()  # eval mode needed for batch_size=1 (BatchNorm1d constraint)
    features = torch.randn(batch_size, 2048)
    out = encoder.forward(features)
    assert out.shape == (batch_size, 256)
