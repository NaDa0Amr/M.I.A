import torch
import random
import json
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.vocabulary import Vocabulary

class FlickrDataset(torch.utils.data.Dataset):
    """
    Dataset that uses precomputed CNN features (not raw images).
    Each item returns (feature_vector, caption_tensor).
    During training, randomly selects one of the 5 captions per image.
    """
    def __init__(self, features: dict, captions: dict, vocab: Vocabulary):
        # features: {image_name: tensor(2048)}
        # captions: {image_name: [caption1, caption2, ..., caption5]}
        # Store image names as a list for indexing
        self.image_names = list(features.keys())
        self.features = features
        self.captions = captions
        self.vocab = vocab
    
    def __len__(self) -> int: return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        feature = self.features[image_name]  # tensor(2048)
        # Randomly pick one of the 5 captions
        caption = random.choice(self.captions[image_name])
        caption_tensor = torch.tensor(self.vocab.numericalize(caption), dtype=torch.long)
        return feature, caption_tensor

class CaptionCollate:
    """Custom collate function that pads captions to equal length."""
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        features = torch.stack([item[0] for item in batch], dim=0)
        captions = [item[1] for item in batch]
        captions = torch.nn.utils.rnn.pad_sequence(
            captions, batch_first=True, padding_value=self.pad_idx
        )
        return features, captions

def create_data_loaders(config: dict, vocab: Vocabulary) -> tuple:
    """Create train, val, test DataLoaders from precomputed features."""
    features_dir = Path(config['data']['features_dir'])
    
    # Load features
    train_features = torch.load(features_dir / 'train_features.pt')
    val_features = torch.load(features_dir / 'val_features.pt')
    test_features = torch.load(features_dir / 'test_features.pt')
    
    # Load splits
    with open(features_dir / 'splits.json', 'r', encoding='utf-8') as f:
        splits = json.load(f)
        
    train_captions = splits['train']
    val_captions = splits['val']
    test_captions = splits['test']
    
    # Create datasets
    train_dataset = FlickrDataset(train_features, train_captions, vocab)
    val_dataset = FlickrDataset(val_features, val_captions, vocab)
    test_dataset = FlickrDataset(test_features, test_captions, vocab)
    
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 4)
    
    collate_fn = CaptionCollate(pad_idx=vocab.pad_idx)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader
