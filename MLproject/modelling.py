from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Telco Churn baseline model.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the root.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=None,
        help="Preprocessed train CSV path.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=None,
        help="Preprocessed test CSV path.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI. Defaults to <project-root>/mlruns.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="telco-churn-baseline",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for local evaluation artifacts.",
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--C", type=float, default=1.0)
    return parser.parse_args()


def resolve_path(path: Path | None, default: Path) -> Path:
    return path.resolve() if path is not None else default.resolve()


def load_preprocessed_data(
    train_path: Path, test_path: Path
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if not train_path.is_file():
        raise FileNotFoundError(f"Preprocessed train dataset not found: {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Preprocessed test dataset not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if "Churn" not in train_df.columns or "Churn" not in test_df.columns:
        raise ValueError("Both preprocessed datasets must contain a 'Churn' target column.")

    x_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"].astype("int64")
    x_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"].astype("int64")
    if list(x_train.columns) != list(x_test.columns):
        raise ValueError("Train and test feature columns are not identical.")
    if x_train.isna().any().any() or x_test.isna().any().any():
        raise ValueError("Preprocessed datasets contain missing feature values.")
    return x_train, y_train, x_test, y_test


def save_evaluation_artifacts(
    output_dir: Path,
    y_test: pd.Series,
    predictions: pd.Series,
    probabilities: pd.Series,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "confusion_matrix.png"
    report_path = output_dir / "classification_report.json"
    metrics_path = output_dir / "evaluation_metrics.json"

    matrix = confusion_matrix(y_test, predictions)
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=["No churn", "Churn"]
    ).plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Telco Churn — Confusion Matrix")
    figure.tight_layout()
    figure.savefig(matrix_path, dpi=150)
    plt.close(figure)

    report = classification_report(
        y_test, predictions, target_names=["No churn", "Churn"], output_dict=True
    )
    report_path.write_text(json.dumps(report, indent=2))
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return {
        "confusion_matrix": matrix_path,
        "classification_report": report_path,
        "evaluation_metrics": metrics_path,
    }


def train(args: argparse.Namespace) -> str:
    project_root = args.project_root.resolve()
    train_path = resolve_path(
        args.train_path, project_root / "telco_customer_churn_preprocessing" / "telco_churn_train.csv"
    )
    test_path = resolve_path(
        args.test_path, project_root / "telco_customer_churn_preprocessing" / "telco_churn_test.csv"
    )
    output_dir = resolve_path(
        args.output_dir, project_root / "telco_customer_churn_preprocessing" / "artifacts"
    )
    tracking_uri = args.tracking_uri or str(project_root / "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    project_run_id = os.environ.get("MLFLOW_RUN_ID")
    if not project_run_id:
        mlflow.set_experiment(args.experiment_name)

    x_train, y_train, x_test, y_test = load_preprocessed_data(train_path, test_path)
    model = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        solver="liblinear",
        random_state=42,
    )

    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=True,
        log_models=True,
        silent=False,
    )
    run_kwargs = {"run_name": "logistic-regression-baseline"}
    if project_run_id:
        run_kwargs["run_id"] = project_run_id
    with mlflow.start_run(**run_kwargs) as run:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]

        signature = infer_signature(x_train, model.predict(x_train))
        mlflow.set_tag("dataset", "Telco Customer Churn preprocessed")
        mlflow.set_tag("target", "Churn")
        mlflow.set_tag("model_type", "LogisticRegression")
        mlflow.set_tag("data_source", str(train_path))
        mlflow.log_params(
            {
                "custom_test_rows": len(x_test),
                "custom_feature_count": x_train.shape[1],
                "custom_random_state": 42,
            }
        )
        mlflow.log_metrics(
            {
                "custom_accuracy": accuracy_score(y_test, predictions),
                "custom_precision": precision_score(y_test, predictions, zero_division=0),
                "custom_recall": recall_score(y_test, predictions, zero_division=0),
                "custom_f1": f1_score(y_test, predictions, zero_division=0),
                "custom_roc_auc": roc_auc_score(y_test, probabilities),
            }
        )
        artifacts = save_evaluation_artifacts(
            output_dir, y_test, predictions, probabilities
        )
        mlflow.log_artifacts(str(output_dir), artifact_path="evaluation")
        mlflow.log_text(json.dumps(signature.to_dict(), indent=2), "model_signature.json")
        print(f"Run ID: {run.info.run_id}")
        print(f"Tracking URI: {tracking_uri}")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        print(f"F1: {f1_score(y_test, predictions, zero_division=0):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
        print("Evaluation artifacts:")
        for name, path in artifacts.items():
            print(f"  {name}: {path}")
        return run.info.run_id


if __name__ == "__main__":
    train(parse_args())
