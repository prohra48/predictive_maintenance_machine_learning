from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from config import DISPLAY_NAMES, FEATURE_COLUMNS, RAW_DATA_FILE, TARGET_COLUMN


def clean_column_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def dataset_source_path(local_path: Path, path_in_repo: str) -> Path:
    """Return a local path, or download from the configured Hugging Face dataset repo."""
    use_hf_data = os.getenv("USE_HF_DATA", "0") == "1"
    dataset_repo = os.getenv("HF_DATASET_REPO")
    if use_hf_data and dataset_repo:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=dataset_repo,
            filename=path_in_repo,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN"),
        )
        return Path(downloaded)
    return local_path


def load_raw_data(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    source = dataset_source_path(path, "engine_data.csv")
    if not source.exists():
        raise FileNotFoundError(f"Raw dataset not found: {source}")
    return pd.read_csv(source)


def clean_engine_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [clean_column_name(col) for col in cleaned.columns]

    expected_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = expected_columns.difference(cleaned.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    cleaned = cleaned[FEATURE_COLUMNS + [TARGET_COLUMN]]
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def hf_is_configured(*repo_env_names: str) -> bool:
    if not os.getenv("HF_TOKEN"):
        return False
    return all(os.getenv(name) for name in repo_env_names)


def label_for(column: str) -> str:
    return DISPLAY_NAMES.get(column, column.replace("_", " ").title())


def print_step(message: str) -> None:
    print(f"\n=== {message} ===")
