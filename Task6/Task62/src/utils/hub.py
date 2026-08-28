import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, snapshot_download

def upload_to_hub(artifact_dir: str, repo_id: str, token: str = None, commit_message: str = "Upload image captioning model") -> str:
    """Upload model artifacts to HuggingFace Hub. Returns the repo URL."""
    token = token or os.getenv('HF_TOKEN')
    if not token:
        raise ValueError("HuggingFace token required. Set HF_TOKEN env var or pass token parameter.")
    
    api = HfApi(token=token)
    create_repo(repo_id=repo_id, repo_type='model', token=token, exist_ok=True, private=False)
    
    api.upload_folder(
        folder_path=artifact_dir,
        repo_id=repo_id,
        repo_type='model',
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"

def download_from_hub(repo_id: str, cache_dir: str = None) -> Path:
    """Download model artifacts from HuggingFace Hub."""
    cache_dir = cache_dir or os.getenv('HF_HOME', './cache/huggingface')
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type='model',
        cache_dir=cache_dir,
    )
    return Path(local_dir)
