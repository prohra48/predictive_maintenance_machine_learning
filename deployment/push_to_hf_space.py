from __future__ import annotations

import os
from pathlib import Path
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACE_REPO = os.getenv("HF_SPACE_REPO")
HF_TOKEN = os.getenv("HF_TOKEN")

def upload(path: Path, path_in_repo: str, api: HfApi) -> None:
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo,
        repo_id=SPACE_REPO,
        repo_type="space",
    )

def main() -> None:
    if not HF_TOKEN or not SPACE_REPO:
        raise RuntimeError("Set HF_TOKEN and HF_SPACE_REPO before pushing to a Hugging Face Space.")

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True)

    # Uploading files to the root of the repo so Dockerfile can find them easily
    upload(PROJECT_ROOT / "deployment" / "app.py", "app.py", api)
    upload(PROJECT_ROOT / "deployment" / "requirements.txt", "requirements.txt", api)
    upload(PROJECT_ROOT / "deployment" / "Dockerfile", "Dockerfile", api)
    upload(PROJECT_ROOT / "models" / "best_model.joblib", "models/best_model.joblib", api)
    upload(PROJECT_ROOT / "models" / "model_metadata.json", "models/model_metadata.json", api)
    print(f"Uploaded Streamlit deployment files to Hugging Face Space: {SPACE_REPO}")

if __name__ == "__main__":
    main()
