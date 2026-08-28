import pytest
import torch
from src.inference.pipeline import CaptionPipeline
from src.models.captioner import ImageCaptioner
from src.data.transforms import get_eval_transforms

@pytest.fixture
def dummy_pipeline(mock_vocab):
    model = ImageCaptioner(embed_dim=256, hidden_dim=512, vocab_size=len(mock_vocab))
    transform = get_eval_transforms()
    device = torch.device('cpu')
    return CaptionPipeline(model, mock_vocab, transform, device)

def test_pipeline_end_to_end(dummy_pipeline, dummy_image):
    result = dummy_pipeline.generate(dummy_image, strategy='greedy', max_length=10)
    assert result is not None
    assert 'caption' in result

def test_pipeline_output_format(dummy_pipeline, dummy_image):
    result = dummy_pipeline.generate(dummy_image, strategy='greedy')
    assert 'caption' in result
    assert 'tokens' in result
    assert 'token_ids' in result
    assert 'score' in result
    assert isinstance(result['caption'], str)

def test_pipeline_greedy_strategy(dummy_pipeline, dummy_image):
    result = dummy_pipeline.generate(dummy_image, strategy='greedy', max_length=5)
    assert isinstance(result['caption'], str)
    assert len(result['token_ids']) <= 5

def test_pipeline_beam_strategy(dummy_pipeline, dummy_image):
    result = dummy_pipeline.generate(dummy_image, strategy='beam', beam_size=3, max_length=5)
    assert isinstance(result['caption'], str)
    assert len(result['token_ids']) <= 5
