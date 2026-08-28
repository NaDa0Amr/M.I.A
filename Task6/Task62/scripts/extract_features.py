import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path so 'src' can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from src.models.encoder import ImageEncoder
from src.utils.config import load_config, get_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Extract and cache CNN features offline")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)
    
    encoder_name = config['model'].get('encoder', 'resnet50')
    encoder = ImageEncoder(model_name=encoder_name).to(device)
    
    splits_path = config['data'].get('splits_path', 'data/processed/splits.json')
    images_dir = config['data'].get('images_dir', 'data/raw/Images')
    features_dir = config['data'].get('features_dir', 'data/processed')
    
    Path(features_dir).mkdir(parents=True, exist_ok=True)
    
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        
    eval_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    batch_size = 32
    start_time = time.time()
    
    for split_name, split_data in splits.items():
        # Handle both dict {img_name: [captions]} and list [img_name, ...]
        image_list = list(split_data.keys()) if isinstance(split_data, dict) else split_data
        logger.info(f"Processing split '{split_name}' with {len(image_list)} images...")
        features_dict = {}
        
        for i in tqdm(range(0, len(image_list), batch_size), desc=f"Extracting {split_name}"):
            batch_images = image_list[i:i + batch_size]
            tensors = []
            valid_images = []
            
            for img_name in batch_images:
                img_path = Path(images_dir) / img_name
                if not img_path.exists():
                    logger.warning(f"Image not found: {img_path}")
                    continue
                try:
                    img = Image.open(img_path).convert('RGB')
                    tensors.append(eval_transforms(img))
                    valid_images.append(img_name)
                except Exception as e:
                    logger.warning(f"Failed to load image {img_name}: {e}")
            
            if not tensors:
                continue
                
            batch_tensor = torch.stack(tensors).to(device)
            with torch.no_grad():
                batch_features = encoder.extract_features(batch_tensor)
                
            for img_name, feature in zip(valid_images, batch_features):
                features_dict[img_name] = feature.cpu()
                
        out_path = Path(features_dir) / f"{split_name}_features.pt"
        torch.save(features_dict, out_path)
        logger.info(f"Saved {len(features_dict)} features for {split_name} to {out_path}")
        
    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
