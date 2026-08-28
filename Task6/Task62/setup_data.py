import os
import random
import json
import argparse
from pathlib import Path
from collections import defaultdict
import yaml

from src.data.vocabulary import Vocabulary

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Prepare Flickr8k dataset")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Handle config if it doesn't exist to make the script run for now
    if not os.path.exists(args.config):
        print(f"Warning: Config file {args.config} not found. Using defaults.")
        config = {'data': {'min_freq': 3, 'processed_dir': 'data/processed'}}
    else:
        config = load_config(args.config)
        
    min_freq = config.get('data', {}).get('min_freq', 3)
    processed_dir = Path(config.get('data', {}).get('processed_dir', 'data/processed'))
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = Path('data/raw/Images')
    captions_file = Path('data/raw/captions.txt')
    
    if not images_dir.exists() or not captions_file.exists():
        print("Error: Dataset not found.")
        print("Please download the Flickr8k dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/adityajn105/flickr8k")
        print("Extract it so that data/raw/Images/ and data/raw/captions.txt exist.")
        return

    print("Parsing captions...")
    image_captions = defaultdict(list)
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # Check if CSV format (adityajn105/flickr8k)
        if len(lines) > 0 and 'image,caption' in lines[0].lower():
            for line in lines[1:]:
                line = line.strip()
                if not line: continue
                # Split by first comma
                parts = line.split(',', 1)
                if len(parts) == 2:
                    img_name, caption = parts
                    image_captions[img_name.strip()].append(caption.strip())
        else:
            # Flickr8k.token.txt format
            for line in lines:
                line = line.strip()
                if not line: continue
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    img_id, caption = parts
                    img_name = img_id.split('#')[0]
                    image_captions[img_name.strip()].append(caption.strip())
                    
    print(f"Found captions for {len(image_captions)} images.")
    
    # Create splits
    image_ids = list(image_captions.keys())
    random.seed(42)
    random.shuffle(image_ids)
    
    # 6000 train, 1000 val, 1000 test (or whatever is left)
    train_ids = image_ids[:6000]
    val_ids = image_ids[6000:7000]
    test_ids = image_ids[7000:]
    
    splits = {
        "train": {img: image_captions[img] for img in train_ids},
        "val": {img: image_captions[img] for img in val_ids},
        "test": {img: image_captions[img] for img in test_ids}
    }
    
    print("Building vocabulary from training captions...")
    train_captions_list = []
    for captions in splits["train"].values():
        train_captions_list.extend(captions)
        
    vocab = Vocabulary(freq_threshold=min_freq)
    vocab.build_vocabulary(train_captions_list)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Save splits and vocab
    splits_path = processed_dir / 'splits.json'
    with open(splits_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits to {splits_path}")
    
    vocab_path = processed_dir / 'vocab.json'
    vocab.save(str(vocab_path))
    print(f"Saved vocab to {vocab_path}")
    
    print("\nDataset Statistics:")
    print(f"Train images: {len(train_ids)}")
    print(f"Val images: {len(val_ids)}")
    print(f"Test images: {len(test_ids)}")
    
    print("\nSample Training Captions:")
    if train_ids:
        sample_img = train_ids[0]
        for i, cap in enumerate(splits["train"][sample_img][:3]):
            print(f"  {i+1}. {cap}")

if __name__ == "__main__":
    main()
