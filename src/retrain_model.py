"""Task C: retrain a CANDIDATE model with old training data + new labelled batch.

- The champion file (artifacts/credit.joblib) is NEVER overwritten.
- The candidate is saved in artifacts/candidate/credit_candidate.joblib
- The run is tracked in MLflow (same experiment "credit-scoring" as train_credit.py)
"""
import hashlib

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import clone

from src.common import (
    CANDIDATE_DIR, CHAMPION_PATH, REPORT_DIR, get_metrics, load_original_split,
    save_json,
)
from src.evaluate_model import get_holdout_and_new_data
from src.model_utils import load_bundle


def file_hash(path):
    """Fingerprint of a file - if the file changes, the hash changes."""
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    champion = load_bundle("credit")
    hash_before = file_hash(CHAMPION_PATH)

    # old training data + new labelled batch
    X_train_old, X_test, y_train_old, y_test = load_original_split()
    X_new, y_new, X_hold, y_hold = get_holdout_and_new_data()

    X_all = pd.concat([X_train_old, X_new])
    y_all = pd.concat([y_train_old, y_new])
    print(f"Old training rows: {len(X_train_old)} | New rows: {len(X_new)} | Total: {len(X_all)}")

    # clone() makes a fresh, untrained copy of the champion pipeline:
    # same preprocessing (imputer, scaler, one-hot) + same classifier settings.
    candidate = clone(champion)
    candidate.fit(X_all, y_all)

    # evaluate the candidate
    hold_metrics = get_metrics(candidate, X_hold, y_hold)
    new_metrics = get_metrics(candidate, X_new, y_new)

    # save candidate in a SEPARATE folder
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = CANDIDATE_DIR / "credit_candidate.joblib"
    joblib.dump(candidate, candidate_path)
    print("Candidate saved:", candidate_path)

    # prove that the champion file was not touched
    champion_unchanged = file_hash(CHAMPION_PATH) == hash_before
    print("Champion file unchanged:", champion_unchanged)

    # ---------- MLflow tracking ----------
    mlflow.set_experiment("credit-scoring")
    with mlflow.start_run(run_name="retrain-candidate") as run:
        mlflow.log_params({
            "model": type(candidate.steps[-1][1]).__name__,
            "old_training_rows": len(X_train_old),
            "new_batch_rows": len(X_new),
            "total_training_rows": len(X_all),
            "role": "candidate",
        })
        mlflow.log_metrics({
            "holdout_accuracy": hold_metrics["accuracy"],
            "holdout_precision": hold_metrics["precision"],
            "holdout_recall": hold_metrics["recall"],
            "holdout_f1_score": hold_metrics["f1_score"],
        })
        mlflow.sklearn.log_model(
            candidate,
            name="credit-candidate-model",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        run_id = run.info.run_id
    print("MLflow run id:", run_id)

    report = {
        "candidate_model_path": str(candidate_path.relative_to(CANDIDATE_DIR.parents[1])),
        "old_training_rows": len(X_train_old),
        "new_batch_rows": len(X_new),
        "total_training_rows": len(X_all),
        "holdout_metrics": hold_metrics,
        "new_batch_metrics": new_metrics,
        "note": "new_batch_metrics is optimistic because the candidate trained on those rows",
        "champion_file_unchanged": champion_unchanged,
        "mlflow_run_id": run_id,
    }
    save_json(report, REPORT_DIR / "candidate_training_report.json")

    print(f"Candidate on hold-out: Accuracy={hold_metrics['accuracy']} "
          f"Precision={hold_metrics['precision']} Recall={hold_metrics['recall']} "
          f"F1={hold_metrics['f1_score']}")


if __name__ == "__main__":
    main()
