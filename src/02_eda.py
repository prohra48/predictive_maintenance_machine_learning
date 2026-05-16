from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import (
    CLEAN_DATA_FILE,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    LOGS_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
    ensure_directories,
)
from utils import clean_engine_data, label_for, load_raw_data, print_step, write_json


plt.switch_backend("Agg")
sns.set_theme(style="whitegrid", palette="Set2")


def save_target_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Normal / no maintenance", "Needs maintenance"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts.values, color=["#4C78A8", "#F58518"])
    ax.set_title("Figure 1. Engine Condition Distribution")
    ax.set_ylabel("Number of Records")
    ax.set_xlabel("Engine Condition")
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_01_target_distribution.png", dpi=180)
    plt.close(fig)


def save_feature_histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for ax, column in zip(axes, FEATURE_COLUMNS):
        sns.histplot(df[column], kde=True, ax=ax, color="#4C78A8")
        ax.set_title(label_for(column))
        ax.set_xlabel("")
        ax.set_ylabel("Count")
    fig.suptitle("Figure 2. Univariate Distribution of Engine Sensor Readings", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_02_feature_histograms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_boxplots_by_condition(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for ax, column in zip(axes, FEATURE_COLUMNS):
        sns.boxplot(data=df, x=TARGET_COLUMN, y=column, ax=ax, hue=TARGET_COLUMN, legend=False)
        ax.set_title(label_for(column))
        ax.set_xlabel("Engine Condition")
        ax.set_ylabel("")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal", "Maintenance"])
    fig.suptitle("Figure 3. Sensor Readings by Engine Condition", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_03_boxplots_by_condition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, ax=ax)
    ax.set_title("Figure 4. Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_04_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def outlier_summary(df: pd.DataFrame) -> dict[str, int]:
    counts = {}
    for column in FEATURE_COLUMNS:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        counts[column] = int(((df[column] < lower) | (df[column] > upper)).sum())
    return counts


def main() -> None:
    ensure_directories()
    print_step("Exploratory data analysis")

    raw_df = load_raw_data()
    df = clean_engine_data(raw_df)
    df.to_csv(CLEAN_DATA_FILE, index=False)

    describe = df.describe().T.round(2)
    describe.to_csv(TABLES_DIR / "table_01_data_overview.csv")

    target_counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .rename_axis("engine_condition")
        .reset_index(name="records")
    )
    target_counts["percentage"] = (target_counts["records"] / len(df) * 100).round(2)
    target_counts.to_csv(TABLES_DIR / "table_02_target_distribution.csv", index=False)

    grouped_means = df.groupby(TARGET_COLUMN)[FEATURE_COLUMNS].mean().round(2)
    grouped_means.to_csv(TABLES_DIR / "table_03_feature_means_by_condition.csv")

    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr(numeric_only=True).round(3)
    corr.to_csv(TABLES_DIR / "table_04_correlation_matrix.csv")

    save_target_distribution(df)
    save_feature_histograms(df)
    save_boxplots_by_condition(df)
    save_correlation_heatmap(df)

    correlations_with_target = (
        corr[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .to_dict()
    )

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "target_distribution": target_counts.to_dict(orient="records"),
        "outlier_counts_iqr": outlier_summary(df),
        "feature_means_by_condition": json.loads(grouped_means.to_json(orient="index")),
        "correlations_with_target": correlations_with_target,
        "figures": [
            "figure_01_target_distribution.png",
            "figure_02_feature_histograms.png",
            "figure_03_boxplots_by_condition.png",
            "figure_04_correlation_heatmap.png",
        ],
    }
    write_json(LOGS_DIR / "eda_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
