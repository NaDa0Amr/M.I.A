import torch
import pytest
from src.inference.predictor import greedy_decode, beam_search_decode
from src.models.captioner import ImageCaptioner

@pytest.fixture
def dummy_model(mock_vocab):
    return ImageCaptioner(embed_dim=256, hidden_dim=512, vocab_size=len(mock_vocab))

def test_greedy_decode_returns_dict(dummy_model, mock_vocab):
    features = torch.randn(1, 2048)
    result = greedy_decode(dummy_model, features, mock_vocab, max_len=10)
    
    assert isinstance(result, dict)
    assert 'caption' in result
    assert 'tokens' in result
    assert 'token_ids' in result
    assert 'score' in result

def test_greedy_decode_caption_is_string(dummy_model, mock_vocab):
    features = torch.randn(1, 2048)
    result = greedy_decode(dummy_model, features, mock_vocab, max_len=10)
    
    assert isinstance(result['caption'], str)

def test_greedy_decode_max_length(dummy_model, mock_vocab):
    features = torch.randn(1, 2048)
    max_len = 5
    result = greedy_decode(dummy_model, features, mock_vocab, max_len=max_len)
    
    assert len(result['token_ids']) <= max_len

def test_beam_search_returns_dict(dummy_model, mock_vocab):
    features = torch.randn(1, 2048)
    result = beam_search_decode(dummy_model, features, mock_vocab, beam_size=3, max_len=10)
    
    assert isinstance(result, dict)
    assert 'caption' in result
    assert 'tokens' in result
    assert 'token_ids' in result
    assert 'score' in result

def test_beam_search_caption_is_string(dummy_model, mock_vocab):
    features = torch.randn(1, 2048)
    result = beam_search_decode(dummy_model, features, mock_vocab, beam_size=2, max_len=10)
    
    assert isinstance(result['caption'], str)
