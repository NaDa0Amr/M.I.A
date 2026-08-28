import torch
import torch.nn.functional as F
from src.data.vocabulary import Vocabulary


def greedy_decode(
    model,
    features: torch.Tensor,
    vocab: Vocabulary,
    max_len: int = 20,
    temperature: float = 1.0,
) -> dict:
    """
    Greedy decoding: pick argmax at each timestep.
    
    Args:
        model: ImageCaptioner in eval mode
        features: precomputed CNN features (1, 2048)
        vocab: Vocabulary instance
        max_len: maximum caption length
        temperature: softmax temperature
    
    Returns:
        dict with 'caption' (str), 'tokens' (list[str]), 'token_ids' (list[int]), 'score' (float)
    """
    model.eval()
    device = features.device
    
    # Project image features
    projected = model.encoder(features)  # (1, embed_dim)
    
    # Start with image feature
    input_embed = projected.unsqueeze(1)  # (1, 1, embed_dim)
    hidden = None
    
    token_ids = []
    log_probs = []
    
    for step in range(max_len):
        logits, hidden = model.decoder.generate_step(input_embed, hidden)
        # Apply temperature
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        
        # Greedy: pick highest probability
        token_id = logits.argmax(dim=-1).item()
        token_log_prob = torch.log(probs[0, token_id]).item()
        
        if token_id == vocab.end_idx:
            break
        
        token_ids.append(token_id)
        log_probs.append(token_log_prob)
        
        # Next input: embedding of predicted word
        word_tensor = torch.tensor([[token_id]], device=device)
        input_embed = model.decoder.embedding(word_tensor)  # (1, 1, embed_dim)
    
    # Convert to words
    tokens = [vocab.idx2word.get(tid, vocab.UNK_TOKEN) for tid in token_ids]
    caption = ' '.join(tokens)
    score = sum(log_probs) / max(len(log_probs), 1)  # average log probability
    
    return {
        'caption': caption,
        'tokens': tokens,
        'token_ids': token_ids,
        'score': score
    }


def beam_search_decode(
    model,
    features: torch.Tensor,
    vocab: Vocabulary,
    beam_size: int = 3,
    max_len: int = 20,
    temperature: float = 1.0,
) -> dict:
    """
    Beam search decoding.
    
    Maintains top beam_size hypotheses at each step.
    Uses length normalization for scoring.
    
    Returns dict with 'caption', 'tokens', 'token_ids', 'score' for the best hypothesis.
    """
    model.eval()
    device = features.device
    
    # 1. Start with projected features as first input for all beams
    projected = model.encoder(features)  # (1, embed_dim)
    initial_input = projected.unsqueeze(1)  # (1, 1, embed_dim)
    
    beams = [{'token_ids': [], 'log_prob': 0.0, 'hidden': None, 'input_embed': initial_input, 'done': False}]
    completed_beams = []
    
    # 2. At each step, expand each beam with top beam_size words
    for step in range(max_len):
        new_beams = []
        for beam in beams:
            if beam['done']:
                new_beams.append(beam)
                continue
            
            logits, hidden = model.decoder.generate_step(beam['input_embed'], beam['hidden'])
            logits = logits / temperature
            log_probs = F.log_softmax(logits, dim=-1)
            
            top_log_probs, top_indices = log_probs[0].topk(beam_size)
            
            for i in range(beam_size):
                token_id = top_indices[i].item()
                log_prob = top_log_probs[i].item()
                
                new_token_ids = beam['token_ids'] + [token_id]
                new_log_prob = beam['log_prob'] + log_prob
                
                # 4. A hypothesis is complete when it generates <end>
                if token_id == vocab.end_idx:
                    completed_beams.append({
                        'token_ids': new_token_ids[:-1],
                        'log_prob': new_log_prob,
                        'done': True
                    })
                else:
                    word_tensor = torch.tensor([[token_id]], device=device)
                    new_input_embed = model.decoder.embedding(word_tensor)
                    
                    new_beams.append({
                        'token_ids': new_token_ids,
                        'log_prob': new_log_prob,
                        'hidden': hidden,
                        'input_embed': new_input_embed,
                        'done': False
                    })
        
        # 6. Score = sum(log_probs) / len^alpha (alpha=0.7 for length normalization)
        def score_fn(b):
            length = max(len(b['token_ids']), 1)
            return b['log_prob'] / (length ** 0.7)
            
        # 3. Keep top beam_size hypotheses globally
        new_beams.sort(key=score_fn, reverse=True)
        beams = new_beams[:beam_size]
        
        # 5. Continue until all beams complete or max_len reached
        if all(b.get('done', False) for b in beams):
            break

    # Add any remaining incomplete beams to completed list for final evaluation
    completed_beams.extend([b for b in beams if not b.get('done', False)])
    
    def score_fn(b):
        length = max(len(b['token_ids']), 1)
        return b['log_prob'] / (length ** 0.7)
        
    completed_beams.sort(key=score_fn, reverse=True)
    # 7. Return the best completed hypothesis
    best_beam = completed_beams[0] if completed_beams else beams[0]
    
    token_ids = best_beam['token_ids']
    tokens = [vocab.idx2word.get(tid, vocab.UNK_TOKEN) for tid in token_ids]
    caption = ' '.join(tokens)
    score = score_fn(best_beam)
    
    return {
        'caption': caption,
        'tokens': tokens,
        'token_ids': token_ids,
        'score': score
    }
