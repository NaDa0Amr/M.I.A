import pytest
import torch
from src.data.dataset import FlickrDataset, CaptionCollate
from src.data.vocabulary import Vocabulary


@pytest.fixture
def small_vocab():
    """Create a small vocabulary for dataset testing."""
    vocab = Vocabulary(freq_threshold=1)
    captions = [
        "a dog running in the park",
        "a cat sitting on a mat",
        "a dog playing with a ball",
    ]
    vocab.build_vocabulary(captions)
    return vocab


def test_flickr_dataset_getitem(small_vocab):
    """Test FlickrDataset returns correct types."""
    features = {
        "img1.jpg": torch.randn(2048),
        "img2.jpg": torch.randn(2048),
    }
    captions = {
        "img1.jpg": ["a dog running in the park"],
        "img2.jpg": ["a cat sitting on a mat"],
    }
    dataset = FlickrDataset(features, captions, small_vocab)

    feature, caption = dataset[0]
    assert isinstance(feature, torch.Tensor)
    assert feature.shape == (2048,)
    assert isinstance(caption, torch.Tensor)
    assert caption.dtype == torch.long
    # Caption should start with <start> and end with <end>
    assert caption[0].item() == small_vocab.start_idx
    assert caption[-1].item() == small_vocab.end_idx


def test_caption_collate():
    """Test CaptionCollate pads captions correctly."""
    collate_fn = CaptionCollate(pad_idx=0)

    # Create mock batch with features (2048-d) and variable-length captions
    batch = [
        (torch.randn(2048), torch.tensor([1, 4, 5, 2])),
        (torch.randn(2048), torch.tensor([1, 8, 9, 10, 11, 2])),
        (torch.randn(2048), torch.tensor([1, 6, 2])),
    ]

    features, captions = collate_fn(batch)

    assert features.shape == (3, 2048)
    # The max length in batch is 6
    assert captions.shape == (3, 6)

    # Check padding is applied correctly
    assert captions[0, 4].item() == 0  # Pad idx
    assert captions[2, 3].item() == 0  # Pad idx


def test_no_data_leakage():
    """Verify train/val/test image sets don't overlap."""
    train_ids = set(["img1", "img2", "img3"])
    val_ids = set(["img4", "img5"])
    test_ids = set(["img6", "img7"])

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
