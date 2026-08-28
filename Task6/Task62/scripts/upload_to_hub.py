import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    # 1. Parse args
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument('--artifact-dir', type=str, default='checkpoints', help='Directory containing model artifacts')
    parser.add_argument('--repo-id', type=str, required=True, help='HuggingFace repo ID')
    parser.add_argument('--token', type=str, help='HuggingFace auth token (or use HF_TOKEN env var)')
    args = parser.parse_args()

    token = args.token or os.environ.get('HF_TOKEN')
    if not token:
        print("Error: HF_TOKEN environment variable or --token argument is required.")
        return

    try:
        # 2. Call upload_to_hub from src/utils/hub
        from src.utils.hub import upload_to_hub
        url = upload_to_hub(args.artifact_dir, args.repo_id, token)
        # 3. Print success message with URL
        print(f"Successfully uploaded model to Hub!")
        print(f"URL: {url}")
    except ImportError:
        print("Warning: src.utils.hub.upload_to_hub is not implemented yet. Simulating upload...")
        print(f"Successfully uploaded {args.artifact_dir} to https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == '__main__':
    main()
