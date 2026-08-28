import json
import torch
from pathlib import Path

def save_model_artifacts(model, vocab, model_config: dict, output_dir: str) -> None:
    """Save model weights, vocabulary, and config to output_dir."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict as .pt file
    torch.save(model.state_dict(), output_path / 'model.pt')
    
    # Save vocabulary
    vocab.save(str(output_path / 'vocab.json'))
    
    # Save model config
    with open(output_path / 'config.json', 'w') as f:
        json.dump(model_config, f, indent=2)

def load_model_artifacts(artifact_dir: str, device: str = 'cpu') -> tuple:
    """Load model state dict, vocabulary, and config from artifact_dir."""
    artifact_path = Path(artifact_dir)
    
    # Load config
    with open(artifact_path / 'config.json', 'r') as f:
        model_config = json.load(f)
    
    # Load vocabulary
    from src.data.vocabulary import Vocabulary
    vocab = Vocabulary.load(str(artifact_path / 'vocab.json'))
    
    # Load state dict
    state_dict = torch.load(
        artifact_path / 'model.pt',
        map_location=device,
        weights_only=True
    )
    
    return state_dict, vocab, model_config
