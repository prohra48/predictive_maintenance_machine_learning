from __future__ import annotations

import os
from datetime import datetime, timezone

from config import LOGS_DIR, RAW_DATA_FILE, ensure_directories
from utils import clean_engine_data, hf_is_configured, load_raw_data, print_step, write_json


def upload_raw_data_to_hugging_face() -> str:
    if not hf_is_configured("HF_DATASET_REPO"):
        return "skipped: set HF_TOKEN and HF_DATASET_REPO to upload the raw dataset"

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = os.environ["HF_DATASET_REPO"]
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(RAW_DATA_FILE),
        path_in_repo="engine_data.csv",
        repo_id=repo_id,
        repo_type="dataset",
    )
    return f"uploaded to Hugging Face dataset repo: {repo_id}"


def main() -> None:
    ensure_directories()
    print_step("Data registration")

    raw_df = load_raw_data()
    clean_df = clean_engine_data(raw_df)
    upload_status = upload_raw_data_to_hugging_face()

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": str(RAW_DATA_FILE),
        "rows": int(clean_df.shape[0]),
        "columns": int(clean_df.shape[1]),
        "missing_values": int(clean_df.isna().sum().sum()),
        "duplicates_after_cleaning": int(clean_df.duplicated().sum()),
        "hugging_face_dataset_status": upload_status,
    }

    write_json(LOGS_DIR / "data_registration_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
