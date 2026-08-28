import argparse
import json
import os
import sys
import torch
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import CaptionPipeline
from src.evaluation.metrics import evaluate_all
from src.evaluation.visualize import display_examples
from src.inference.predictor import greedy_decode, beam_search_decode

def main():
    # 1. Parse args
    parser = argparse.ArgumentParser(description="Evaluate Image Captioning Model")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory containing model artifacts')
    parser.add_argument('--images-dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output JSON file for metrics')
    parser.add_argument('--num-examples', type=int, default=10, help='Number of examples to visualize')
    parser.add_argument('--strategy', type=str, default='beam', choices=['greedy', 'beam'])
    args = parser.parse_args()

    # 2. Load model via CaptionPipeline.from_checkpoint()
    print(f"Loading pipeline from {args.checkpoint_dir}...")
    pipeline = CaptionPipeline.from_checkpoint(args.checkpoint_dir)
    
    # 3. Load test split from splits.json
    splits_path = Path('data/processed/splits.json')
    if not splits_path.exists():
        print(f"{splits_path} not found, assuming all images in images_dir are test images.")
        test_images = [p.name for p in Path(args.images_dir).glob('*.jpg')]
        references = {img: ["dummy reference caption"] for img in test_images}
    else:
        with open(splits_path, 'r', encoding='utf-8') as f:
            splits = json.load(f)
            
        test_split = splits.get('test', {})
        if isinstance(test_split, dict):
            references = test_split
            test_images = list(test_split.keys())
        else:
            test_images = test_split
            references = {img: ["dummy reference caption"] for img in test_images}

    hypotheses = {}
    print(f"Evaluating {len(test_images)} images...")
    
    # 5. Generate captions for all test images using the pipeline's model
    # (Assuming we use pipeline.generate directly instead of test_features.pt)
    from tqdm import tqdm
    for img_name in tqdm(test_images, desc="Generating Captions"):
        img_path = Path(args.images_dir) / img_name
        try:
            from PIL import Image
            img = Image.open(img_path)
            res = pipeline.generate(img, strategy=args.strategy)
            hypotheses[img_name] = res['caption']
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    # 6. Compute all metrics via evaluate_all()
    metrics = evaluate_all(hypotheses, references)
    
    # 7. Print formatted results table
    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
        
    # 8. Save results to JSON
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # 9. Generate qualitative examples visualization
    vis_path = Path(args.output).with_suffix('.png')
    display_examples(args.images_dir, hypotheses, references, args.num_examples, str(vis_path))
    print(f"Saved visualization to {vis_path}")

    # 10. Print some example image -> generated caption -> reference captions
    print("\nExamples:")
    for img_name in list(hypotheses.keys())[:3]:
        print(f"Image: {img_name}")
        print(f"Gen: {hypotheses[img_name]}")
        print(f"Ref: {references.get(img_name, [])[0] if references.get(img_name) else 'N/A'}\n")

if __name__ == '__main__':
    main()
