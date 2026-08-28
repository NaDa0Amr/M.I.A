import pytest
import torch
import numpy as np
from PIL import Image
from src.data.vocabulary import Vocabulary

@pytest.fixture
def dummy_image():
    """Create a random 224x224 RGB PIL Image."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)

@pytest.fixture
def dummy_image_tensor():
    """Create a random preprocessed image tensor (3, 224, 224)."""
    return torch.randn(3, 224, 224)

@pytest.fixture
def mock_vocab():
    """Create a small vocabulary for testing."""
    vocab = Vocabulary(freq_threshold=1)
    captions = [
        "a dog running in the park",
        "a cat sitting on a mat",
        "a dog playing with a ball",
        "a cat on the grass",
        "the dog runs fast",
        "a brown dog in the field",
        "two dogs playing together",
        "a small cat sleeping",
    ]
    vocab.build_vocabulary(captions)
    return vocab

@pytest.fixture
def mock_features():
    """Create random feature tensors."""
    return torch.randn(1, 2048)

@pytest.fixture
def mock_projected_features():
    """Create random projected feature tensors."""
    return torch.randn(1, 256)

@pytest.fixture
def sample_captions():
    """Sample captions for testing."""
    return [
        "a dog running in the park",
        "a cat sitting on a mat",
        "a dog playing with a ball",
    ]
