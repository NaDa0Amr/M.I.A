import torch
from src.data.transforms import get_train_transforms, get_eval_transforms
from PIL import Image
import numpy as np

def create_dummy_pil():
    arr = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def test_train_transform_output_shape():
    transform = get_train_transforms()
    img = create_dummy_pil()
    out = transform(img)
    assert out.shape == (3, 224, 224)

def test_eval_transform_output_shape():
    transform = get_eval_transforms()
    img = create_dummy_pil()
    out = transform(img)
    assert out.shape == (3, 224, 224)

def test_transform_type():
    transform = get_train_transforms()
    img = create_dummy_pil()
    out = transform(img)
    assert isinstance(out, torch.Tensor)

def test_transform_normalized_range():
    transform = get_eval_transforms()
    img = create_dummy_pil()
    out = transform(img)
    # Usually image net normalization will put values in roughly [-3, 3] range
    assert out.min() >= -3.5
    assert out.max() <= 3.5
