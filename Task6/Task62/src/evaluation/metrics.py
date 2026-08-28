import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from rouge_score import rouge_scorer

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

def compute_bleu(hypotheses: list[list[str]], references: list[list[list[str]]]) -> dict:
    """
    Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4.
    
    Args:
        hypotheses: list of tokenized hypothesis captions [["a", "dog", ...], ...]
        references: list of lists of tokenized reference captions [[["a", "dog"], ["the", "dog"]], ...]
    
    Returns:
        dict with keys 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4' (values 0-100)
    """
    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    return {'BLEU-1': bleu1 * 100, 'BLEU-2': bleu2 * 100, 'BLEU-3': bleu3 * 100, 'BLEU-4': bleu4 * 100}

def compute_rouge(hypotheses: list[str], references: list[list[str]]) -> dict:
    """Compute ROUGE-L F-measure. Takes raw strings."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for hyp, refs in zip(hypotheses, references):
        best = max(scorer.score(ref, hyp)['rougeL'].fmeasure for ref in refs)
        scores.append(best)
    return {'ROUGE-L': sum(scores) / len(scores) * 100 if scores else 0.0}

def compute_meteor(hypotheses: list[list[str]], references: list[list[list[str]]]) -> dict:
    """Compute METEOR. Takes tokenized inputs."""
    scores = []
    for hyp, refs in zip(hypotheses, references):
        score = nltk_meteor(refs, hyp)
        scores.append(score)
    return {'METEOR': sum(scores) / len(scores) * 100 if scores else 0.0}

def evaluate_all(hypotheses_dict: dict, references_dict: dict) -> dict:
    """
    Evaluate all metrics.
    
    Args:
        hypotheses_dict: {image_name: "generated caption string"}
        references_dict: {image_name: ["ref1", "ref2", "ref3", "ref4", "ref5"]}
    
    Returns:
        dict with all metric scores
    """
    # Build aligned lists
    hyp_strings = []
    hyp_tokens = []
    ref_strings = []
    ref_tokens = []
    
    for img_name in hypotheses_dict:
        if img_name not in references_dict:
            continue
        hyp_str = hypotheses_dict[img_name]
        ref_strs = references_dict[img_name]
        
        hyp_strings.append(hyp_str)
        hyp_tokens.append(hyp_str.lower().split())
        ref_strings.append(ref_strs)
        ref_tokens.append([ref.lower().split() for ref in ref_strs])
    
    results = {}
    if hyp_strings:
        results.update(compute_bleu(hyp_tokens, ref_tokens))
        results.update(compute_rouge(hyp_strings, ref_strings))
        results.update(compute_meteor(hyp_tokens, ref_tokens))
    
    return results
