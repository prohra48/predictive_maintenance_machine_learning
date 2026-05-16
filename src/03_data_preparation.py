from __future__ import annotations

import os
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split

from config import (
    CLEAN_DATA_FILE,
    FEATURE_COLUMNS,
    LOGS_DIR,
    TARGET_COLUMN,
    TEST_DATA_FILE,
    TEST_SIZE,
    TRAIN_DATA_FILE,
    ensure_directories,
)
from utils import clean_engine_data, hf_is_configured, load_raw_data, print_step, write_json


def upload_processed_data_to_hugging_face() -> str:
    if not hf_is_configured("HF_DATASET_REPO"):
        return "skipped: set HF_TOKEN and HF_DATASET_REPO to upload train/test datasets"

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = os.environ["HF_DATASET_REPO"]
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    for local_path, remote_path in [
        (CLEAN_DATA_FILE, "processed/engine_data_clean.csv"),
        (TRAIN_DATA_FILE, "processed/train.csv"),
        (TEST_DATA_FILE, "processed/test.csv"),
    ]:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
    return f"uploaded clean/train/test datasets to Hugging Face dataset repo: {repo_id}"


def main() -> None:
    ensure_directories()
    print_step("Data preparation")

    df = clean_engine_data(load_raw_data())
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    df.to_csv(CLEAN_DATA_FILE, index=False)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=df[TARGET_COLUMN],
    )
    train_df.to_csv(TRAIN_DATA_FILE, index=False)
    test_df.to_csv(TEST_DATA_FILE, index=False)

    upload_status = upload_processed_data_to_hugging_face()
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "clean_rows": int(df.shape[0]),
        "clean_columns": int(df.shape[1]),
        "train_rows": int(train_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "test_size": TEST_SIZE,
        "target_train_distribution": train_df[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict(),
        "target_test_distribution": test_df[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict(),
        "hugging_face_dataset_status": upload_status,
    }
    write_json(LOGS_DIR / "data_preparation_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
