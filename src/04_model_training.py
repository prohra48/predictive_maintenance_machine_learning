from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import (
    EXPERIMENTS_DIR,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
    TEST_DATA_FILE,
    TRAIN_DATA_FILE,
    ensure_directories,
)
from utils import dataset_source_path, hf_is_configured, label_for, print_step, write_json


plt.switch_backend("Agg")
sns.set_theme(style="whitegrid", palette="Set2")
N_JOBS = int(os.getenv("ML_N_JOBS", "1"))
FAST_NOTEBOOK = os.getenv("FAST_NOTEBOOK", "0") == "1"
MODEL_N_JOBS = N_JOBS if N_JOBS > 0 else 1


def model_grid() -> dict[str, tuple[Pipeline, dict[str, list]]]:
    if FAST_NOTEBOOK:
        return {
            "Decision Tree": (
                Pipeline(
                    steps=[
                        ("model", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
                    ]
                ),
                {
                    "model__max_depth": [5],
                    "model__min_samples_split": [10],
                    "model__min_samples_leaf": [5],
                },
            ),
            "Random Forest": (
                Pipeline(
                    steps=[
                        ("model", RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=N_JOBS)),
                    ]
                ),
                {
                    "model__n_estimators": [80],
                    "model__max_depth": [10],
                    "model__min_samples_leaf": [5],
                },
            ),
            "Gradient Boosting": (
                Pipeline(
                    steps=[
                        ("model", GradientBoostingClassifier(random_state=42)),
                    ]
                ),
                {
                    "model__n_estimators": [100],
                    "model__learning_rate": [0.05],
                    "model__max_depth": [2],
                },
            ),
            "AdaBoost": (
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        ("model", AdaBoostClassifier(random_state=42)),
                    ]
                ),
                {
                    "model__n_estimators": [150],
                    "model__learning_rate": [0.05],
                },
            ),
            "XGBoost": (
                Pipeline(
                    steps=[
                        (
                            "model",
                            XGBClassifier(
                                objective="binary:logistic",
                                eval_metric="logloss",
                                random_state=42,
                                n_jobs=MODEL_N_JOBS,
                                tree_method="hist",
                            ),
                        ),
                    ]
                ),
                {
                    "model__n_estimators": [120],
                    "model__learning_rate": [0.05],
                    "model__max_depth": [3],
                    "model__subsample": [0.9],
                    "model__colsample_bytree": [0.9],
                },
            ),
        }

    return {
        "Decision Tree": (
            Pipeline(
                steps=[
                    ("model", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
                ]
            ),
            {
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_split": [2, 10, 30],
                "model__min_samples_leaf": [1, 5, 15],
            },
        ),
        "Random Forest": (
            Pipeline(
                steps=[
                    ("model", RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=N_JOBS)),
                ]
            ),
            {
                "model__n_estimators": [120, 250],
                "model__max_depth": [5, 10, None],
                "model__min_samples_leaf": [1, 5],
            },
        ),
        "Gradient Boosting": (
            Pipeline(
                steps=[
                    ("model", GradientBoostingClassifier(random_state=42)),
                ]
            ),
            {
                "model__n_estimators": [100, 180],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
        "AdaBoost": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", AdaBoostClassifier(random_state=42)),
                ]
            ),
            {
                "model__n_estimators": [80, 150],
                "model__learning_rate": [0.05, 0.1, 0.5],
            },
        ),
        "XGBoost": (
            Pipeline(
                steps=[
                    (
                        "model",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            random_state=42,
                            n_jobs=MODEL_N_JOBS,
                            tree_method="hist",
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.9],
            },
        ),
    }


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        y_score = y_pred

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }


def save_confusion_matrix(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> list[list[int]]:
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Maintenance"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Figure 5. Confusion Matrix for Best Model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_05_confusion_matrix.png", dpi=180)
    plt.close(fig)
    return cm.tolist()


def save_feature_importance(model_name: str, model: Pipeline) -> list[dict[str, float]]:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return []

    importances = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importances["feature_label"] = importances["feature"].map(label_for)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=importances, x="importance", y="feature_label", ax=ax, color="#4C78A8")
    ax.set_title(f"Figure 6. Feature Importance for {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_06_feature_importance.png", dpi=180)
    plt.close(fig)

    importances[["feature", "importance"]].to_csv(TABLES_DIR / "table_06_feature_importance.csv", index=False)
    return importances[["feature", "importance"]].round(6).to_dict(orient="records")


def upload_model_to_hugging_face() -> str:
    if not hf_is_configured("HF_MODEL_REPO"):
        return "skipped: set HF_TOKEN and HF_MODEL_REPO to upload the model"

    from huggingface_hub import HfApi

    upload_files = [
        (MODELS_DIR / "best_model.joblib", "best_model.joblib"),
        (MODELS_DIR / "model_metadata.json", "model_metadata.json"),
        (TABLES_DIR / "table_05_model_results.csv", "model_results.csv"),
    ]
    missing_files = [str(local_path) for local_path, _ in upload_files if not local_path.is_file()]
    if missing_files:
        return "skipped: model upload files are missing locally: " + ", ".join(missing_files)

    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = os.environ["HF_MODEL_REPO"]
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    for local_path, remote_path in upload_files:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
        )
    return f"uploaded best model to Hugging Face model repo: {repo_id}"


def load_train_test_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, int, int]:
    print("Loading train and test datasets...")
    train_path = dataset_source_path(TRAIN_DATA_FILE, "processed/train.csv")
    test_path = dataset_source_path(TEST_DATA_FILE, "processed/test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].astype(int)
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].astype(int)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    return x_train, y_train, x_test, y_test, int(train_df.shape[0]), int(test_df.shape[0])


def run_model_experiments(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, GridSearchCV]]:
    experiment_rows = []
    fitted_searches = {}

    for model_name, (pipeline, params) in model_grid().items():
        print(f"Training {model_name}...")
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring="f1",
            cv=3 if FAST_NOTEBOOK else 5,
            n_jobs=N_JOBS,
            refit=True,
            verbose=0,
        )
        search.fit(x_train, y_train)
        fitted_searches[model_name] = search
        metrics = evaluate_model(model_name, search.best_estimator_, x_test, y_test)
        experiment_rows.append(
            {
                **metrics,
                "best_cv_f1": search.best_score_,
                "best_params": json.dumps(search.best_params_),
            }
        )

    results = pd.DataFrame(experiment_rows).sort_values(["f1", "recall", "roc_auc"], ascending=False)
    return results, fitted_searches


def save_experiment_tracking(results: pd.DataFrame) -> pd.DataFrame:
    results_rounded = results.copy()
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "best_cv_f1"]
    results_rounded[metric_cols] = results_rounded[metric_cols].round(4)
    results_rounded.to_csv(EXPERIMENTS_DIR / "experiment_tracking.csv", index=False)
    results_rounded.to_csv(TABLES_DIR / "table_05_model_results.csv", index=False)
    return results_rounded


def save_best_model_artifacts(
    results: pd.DataFrame,
    results_rounded: pd.DataFrame,
    fitted_searches: dict[str, GridSearchCV],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    training_rows: int,
    testing_rows: int,
) -> dict:
    best_model_name = results.iloc[0]["model"]
    best_model = fitted_searches[best_model_name].best_estimator_
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    print(f"Saved best model: {MODELS_DIR / 'best_model.joblib'}")

    y_pred = best_model.predict(x_test)
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.round(4)
    report_df.to_csv(TABLES_DIR / "table_07_classification_report.csv")

    cm = save_confusion_matrix(best_model, x_test, y_test)
    feature_importance = save_feature_importance(best_model_name, best_model)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "best_model": best_model_name,
        "best_params": fitted_searches[best_model_name].best_params_,
        "test_metrics": results_rounded.iloc[0].to_dict(),
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
        "training_rows": training_rows,
        "testing_rows": testing_rows,
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "hugging_face_model_status": "not attempted yet",
    }

    write_json(MODELS_DIR / "model_metadata.json", metadata)
    write_json(LOGS_DIR / "model_training_summary.json", metadata)
    print(f"Saved metadata: {MODELS_DIR / 'model_metadata.json'}")
    return metadata


def main() -> None:
    ensure_directories()
    print_step("Model building with experimentation tracking")

    x_train, y_train, x_test, y_test, training_rows, testing_rows = load_train_test_data()
    results, fitted_searches = run_model_experiments(x_train, y_train, x_test, y_test)
    results_rounded = save_experiment_tracking(results)
    metadata = save_best_model_artifacts(
        results=results,
        results_rounded=results_rounded,
        fitted_searches=fitted_searches,
        x_test=x_test,
        y_test=y_test,
        training_rows=training_rows,
        testing_rows=testing_rows,
    )

    upload_status = upload_model_to_hugging_face()
    metadata["hugging_face_model_status"] = upload_status
    write_json(MODELS_DIR / "model_metadata.json", metadata)
    write_json(LOGS_DIR / "model_training_summary.json", metadata)

    print(results_rounded.to_string(index=False))
    print(f"Best model: {metadata['best_model']}")
    print(f"Model upload status: {upload_status}")


if __name__ == "__main__":
    main()
