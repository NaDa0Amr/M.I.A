import torch
from src.inference.predictor import greedy_decode, beam_search_decode
from src.models.captioner import ImageCaptioner


class CaptionPipeline:
    """
    End-to-end inference pipeline: PIL Image -> Caption string.
    
    Handles:
    1. Loading model from checkpoint or HuggingFace Hub
    2. Image preprocessing (eval transforms)
    3. Feature extraction (CNN forward pass)
    4. Caption generation (greedy or beam search)
    """
    def __init__(self, model, vocab, transform, device):
        self.model = model
        self.vocab = vocab
        self.transform = transform
        self.device = device
        self.model.eval()
        self.encoder_name = model.encoder.model_name if hasattr(model.encoder, 'model_name') else 'resnet50'
        self.decoder_name = 'LSTM'
    
    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, device: str = 'auto') -> 'CaptionPipeline':
        """Load pipeline from local checkpoint directory."""
        from src.utils.serialization import load_model_artifacts
        from src.utils.config import get_device
        from src.data.transforms import get_eval_transforms
        
        if device == 'auto':
            device = get_device({'training': {'device': 'auto'}})
        else:
            device = torch.device(device)
        
        state_dict, vocab, model_config = load_model_artifacts(checkpoint_dir, str(device))
        
        model_cfg = model_config.get('model', model_config)
        model = ImageCaptioner(
            embed_dim=model_cfg['embed_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            vocab_size=len(vocab),
            num_layers=model_cfg.get('num_layers', 1),
            dropout=model_cfg.get('dropout', 0.5),
            encoder_name=model_cfg.get('encoder', 'resnet50')
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        transform = get_eval_transforms()
        return cls(model, vocab, transform, device)
    
    def generate(self, image, strategy='beam', beam_size=3, max_length=20, temperature=1.0) -> dict:
        """Generate caption for a PIL Image."""
        # 1. Convert PIL Image to RGB
        image = image.convert('RGB')
        
        # 2. Apply eval transforms
        image_tensor = self.transform(image)
        
        # 3. Add batch dimension, move to device
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 4. Extract raw 2048-d features using model.encoder.extract_features()
            features = self.model.encoder.extract_features(image_tensor)
            
            # 5. Call appropriate decode function
            if strategy == 'greedy':
                result = greedy_decode(
                    self.model, features, self.vocab, max_len=max_length, temperature=temperature
                )
            elif strategy == 'beam':
                result = beam_search_decode(
                    self.model, features, self.vocab, beam_size=beam_size, 
                    max_len=max_length, temperature=temperature
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
        # 6. Return result
        return result
