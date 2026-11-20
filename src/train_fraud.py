
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "raw" / "creditcard_2023.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

REGISTERED_MODEL_NAME = "fraud-detection-model"


def calculate_metrics(model, X_test, y_test, total_rows, fraud_percentage):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "target": "Class",
        "dataset": "creditcard_2023.csv",
        "rows": int(total_rows),
        "features": int(X_test.shape[1]),
        "fraud_percentage": float(fraud_percentage),
    }


def register_champion(model_uri):
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
        await_registration_for=300,
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered_model.version,
    )

    print("\nMODEL REGISTRY")
    print("Registered model:", REGISTERED_MODEL_NAME)
    print("Version:", registered_model.version)
    print("Alias: champion")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    target = "Class"

    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    df = df.dropna(subset=[target])

    X = df.drop(columns=[target, "id"], errors="ignore")
    y = df[target].astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            f"Expected Class values 0 and 1, got {sorted(y.unique())}"
        )

    print("Dataset shape:", df.shape)
    print("\nClass distribution:")
    print(y.value_counts())
    print("\nFraud percentage:", round(y.mean() * 100, 3), "%")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-detection")

    results = []
    best_model = None
    best_metrics = None
    best_score = -1
    best_name = None
    best_model_uri = None

    for name, classifier in models.items():
        print(f"\nTraining {name}...")

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])

        with mlflow.start_run(run_name=f"creditcard-2023-{name}"):
            model.fit(X_train, y_train)

            metrics = calculate_metrics(
                model,
                X_test,
                y_test,
                len(df),
                y.mean() * 100,
            )
            metrics["model"] = name

            mlflow.log_params({
                "model": name,
                "dataset": "creditcard_2023.csv",
                "test_size": 0.20,
                "random_state": 42,
                "identifier_removed": "id",
            })

            mlflow.log_metrics({
                key: value
                for key, value in metrics.items()
                if isinstance(value, float)
            })

            model_info = mlflow.sklearn.log_model(
                model,
                name="fraud-detection-model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree",
                    "numpy.dtype",
                ],
            )

            results.append(metrics)
            print(json.dumps(metrics, indent=4))

            if metrics["f1_score"] > best_score:
                best_score = metrics["f1_score"]
                best_model = model
                best_metrics = metrics
                best_name = name
                best_model_uri = model_info.model_uri

    best_metrics["selected_metric"] = "f1_score"

    joblib.dump(best_model, ARTIFACT_DIR / "fraud.joblib")

    with open(ARTIFACT_DIR / "fraud_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=4)

    with open(ARTIFACT_DIR / "fraud_model_comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    # Register only the best-F1 model.
    register_champion(best_model_uri)

    print("\nFRAUD MODEL COMPARISON")
    print(pd.DataFrame(results)[
        ["model", "accuracy", "precision", "recall",
         "f1_score", "roc_auc", "pr_auc"]
    ].to_string(index=False))

    print(f"\nBest model: {best_name}")
    print("Best F1 score:", best_score)
    print("Saved:", ARTIFACT_DIR / "fraud.joblib")


if __name__ == "__main__":
    main()
