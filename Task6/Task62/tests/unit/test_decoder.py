import torch
import pytest
from src.models.decoder import CaptionDecoder

def test_decoder_output_shape():
    decoder = CaptionDecoder(embed_dim=256, hidden_dim=512, vocab_size=1000)
    features = torch.randn(2, 256)
    captions = torch.randint(0, 1000, (2, 15))
    logits = decoder.forward(features, captions)
    assert logits.shape == (2, 15, 1000)

def test_decoder_generate_step_output():
    decoder = CaptionDecoder(embed_dim=256, hidden_dim=512, vocab_size=1000)
    input_embed = torch.randn(2, 1, 256)
    hidden_state = None
    logits, next_hidden = decoder.generate_step(input_embed, hidden_state)
    assert logits.shape == (2, 1000)
    assert next_hidden is not None

def test_decoder_generate_step_hidden_state():
    decoder = CaptionDecoder(embed_dim=256, hidden_dim=512, vocab_size=1000, num_layers=1)
    input_embed = torch.randn(2, 1, 256)
    
    # LSTM hidden state is a tuple (h_0, c_0)
    h_0 = torch.randn(1, 2, 512)
    c_0 = torch.randn(1, 2, 512)
    hidden_state = (h_0, c_0)
    
    logits, next_hidden = decoder.generate_step(input_embed, hidden_state)
    assert isinstance(next_hidden, tuple)
    assert next_hidden[0].shape == (1, 2, 512)
    assert next_hidden[1].shape == (1, 2, 512)

@pytest.mark.parametrize("seq_len", [5, 10, 20])
def test_decoder_various_seq_lengths(seq_len):
    decoder = CaptionDecoder(embed_dim=256, hidden_dim=512, vocab_size=100)
    features = torch.randn(4, 256)
    captions = torch.randint(0, 100, (4, seq_len))
    logits = decoder.forward(features, captions)
    assert logits.shape == (4, seq_len, 100)
