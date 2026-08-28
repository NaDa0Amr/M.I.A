import pytest
from src.evaluation.metrics import compute_bleu, compute_rouge, compute_meteor, evaluate_all


def test_bleu_perfect_match():
    # compute_bleu expects tokenized inputs: list[list[str]] and list[list[list[str]]]
    hypotheses = [["the", "cat", "sat", "on", "the", "mat"]]
    references = [[["the", "cat", "sat", "on", "the", "mat"]]]
    scores = compute_bleu(hypotheses, references)
    assert "BLEU-1" in scores
    # A perfect match should yield high BLEU scores
    assert scores["BLEU-1"] > 90


def test_bleu_no_match():
    hypotheses = [["a", "dog", "barked", "loudly"]]
    references = [[["the", "cat", "sat", "on", "the", "mat"]]]
    scores = compute_bleu(hypotheses, references)
    assert scores["BLEU-1"] < 10


def test_rouge_perfect_match():
    # compute_rouge expects raw strings
    hypotheses = ["the cat sat on the mat"]
    references = [["the cat sat on the mat"]]
    scores = compute_rouge(hypotheses, references)
    assert "ROUGE-L" in scores
    assert scores["ROUGE-L"] > 90


def test_meteor_perfect_match():
    # compute_meteor expects tokenized inputs
    hypotheses = [["the", "cat", "sat", "on", "the", "mat"]]
    references = [[["the", "cat", "sat", "on", "the", "mat"]]]
    scores = compute_meteor(hypotheses, references)
    assert "METEOR" in scores
    assert scores["METEOR"] > 90


def test_evaluate_all_returns_all_metrics():
    # evaluate_all expects {image_name: "caption_string"} dicts
    hyp_dict = {"img1": "the cat sat on the mat"}
    ref_dict = {"img1": ["the cat sat on the mat"]}

    results = evaluate_all(hyp_dict, ref_dict)
    assert "BLEU-1" in results
    assert "BLEU-4" in results
    assert "ROUGE-L" in results
    assert "METEOR" in results


def test_bleu_returns_correct_keys():
    hypotheses = [["the", "cat", "sat", "on", "the", "mat"]]
    references = [[["the", "cat", "sat", "on", "the", "mat"]]]
    scores = compute_bleu(hypotheses, references)
    for i in range(1, 5):
        assert f"BLEU-{i}" in scores
