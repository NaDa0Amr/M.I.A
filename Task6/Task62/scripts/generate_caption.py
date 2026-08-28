import argparse
import os
import sys
from pathlib import Path
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import CaptionPipeline

def main():
    # 1. Parse args
    parser = argparse.ArgumentParser(description="Generate caption for a single image")
    parser.add_argument('--image', type=str, required=True, help='Path to the image')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory with model checkpoint')
    parser.add_argument('--strategy', type=str, default='beam', choices=['greedy', 'beam'], help='Decoding strategy')
    parser.add_argument('--beam-size', type=int, default=3, help='Beam size for beam search')
    parser.add_argument('--max-length', type=int, default=20, help='Max caption length')
    args = parser.parse_args()

    # 2. Load pipeline from checkpoint
    print(f"Loading pipeline from {args.checkpoint_dir}...")
    pipeline = CaptionPipeline.from_checkpoint(args.checkpoint_dir)
    
    # 3. Load and display the image
    print(f"Processing image {args.image}...")
    img = Image.open(args.image)
    
    # 4. Generate caption
    result = pipeline.generate(
        img, 
        strategy=args.strategy, 
        beam_size=args.beam_size, 
        max_length=args.max_length
    )
    
    # 5. Print the generated caption and score
    print("\nResult:")
    print(f"Caption: {result['caption']}")
    print(f"Score: {result['score']:.4f}")

if __name__ == '__main__':
    main()
