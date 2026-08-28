import os
import tempfile
import pytest
from src.data.vocabulary import Vocabulary

def test_special_token_indices():
    vocab = Vocabulary()
    assert vocab.pad_idx == 0
    assert vocab.start_idx == 1
    assert vocab.end_idx == 2
    assert vocab.unk_idx == 3
    assert vocab.PAD_TOKEN == '<pad>'
    assert vocab.START_TOKEN == '<start>'
    assert vocab.END_TOKEN == '<end>'
    assert vocab.UNK_TOKEN == '<unk>'

def test_build_vocabulary(mock_vocab):
    assert "dog" in mock_vocab.word2idx
    assert "cat" in mock_vocab.word2idx
    assert mock_vocab.word2idx["dog"] > 3

def test_frequency_threshold():
    vocab = Vocabulary(freq_threshold=2)
    captions = ["a dog runs", "a dog barks", "a cat sleeps"]
    vocab.build_vocabulary(captions)
    assert "dog" in vocab.word2idx
    assert "cat" not in vocab.word2idx

def test_numericalize(mock_vocab):
    caption = "a dog"
    tokens = mock_vocab.numericalize(caption)
    assert tokens[0] == mock_vocab.start_idx
    assert tokens[-1] == mock_vocab.end_idx

def test_denumericalize(mock_vocab):
    caption = "a dog running"
    tokens = mock_vocab.numericalize(caption)
    decoded = mock_vocab.denumericalize(tokens)
    assert isinstance(decoded, str)

def test_unknown_words(mock_vocab):
    tokens = mock_vocab.numericalize("xyz123abc")
    assert mock_vocab.unk_idx in tokens

def test_clean_caption():
    cleaned = Vocabulary.clean_caption("Hello, World! A dog.")
    assert "hello" in cleaned
    assert "world" in cleaned
    assert "a" in cleaned  # 'a' is a single-char exception, kept by clean_caption
    assert "," not in " ".join(cleaned)  # punctuation removed
    assert "." not in " ".join(cleaned)  # punctuation removed

def test_save_and_load(mock_vocab):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "vocab.pkl")
        mock_vocab.save(path)
        loaded_vocab = Vocabulary.load(path)
        assert len(loaded_vocab) == len(mock_vocab)

def test_vocab_length(mock_vocab):
    assert len(mock_vocab) == len(mock_vocab.word2idx)
