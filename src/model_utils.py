
from pathlib import Path
import json
import os

import joblib
import mlflow
import mlflow.sklearn


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# Use an environment override when provided; otherwise use project-root DB.
DEFAULT_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
)

# Local artifact names -> MLflow registered model names.
REGISTERED_MODELS = {
    "credit": "credit-scoring-model",
    "fraud": "fraud-detection-model",
}


def save_bundle(name, model, metrics):
    joblib.dump(model, ARTIFACTS / f"{name}.joblib")

    (ARTIFACTS / f"{name}_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def load_bundle(name):
    if name not in REGISTERED_MODELS:
        raise ValueError(f"Unknown model name: {name}")

    registered_name = REGISTERED_MODELS[name]
    registry_uri = f"models:/{registered_name}@champion"

    # First attempt: load the champion model from MLflow Registry.
    try:
        print(f"Loading {name} model from MLflow Registry...")

        model = mlflow.sklearn.load_model(registry_uri)

        print(f"Successfully loaded {registered_name} champion model.")
        return model

    except Exception as registry_error:
        print(
            f"MLflow Registry loading failed for {name}: "
            f"{registry_error}"
        )
        print("Falling back to local joblib model...")

    # Fallback: load the existing local model.
    model_path = ARTIFACTS / f"{name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Neither MLflow champion nor local model is available. "
            f"Expected fallback file: {model_path}"
        )

    model = joblib.load(model_path)

    print(f"Successfully loaded local fallback: {model_path}")
    return model


def load_metrics(name):
    path = ARTIFACTS / f"{name}_metrics.json"

    if not path.exists():
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "roc_auc": None,
        }

    return json.loads(path.read_text(encoding="utf-8"))
