"""Small helper file used by all the monitoring / retraining scripts.

It keeps the paths, the train/test split and the metric calculation in ONE place
so every script uses exactly the same settings.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "raw" / "GermanCredit.csv"      # original data
NEW_DATA_PATH = ROOT / "datasets" / "new" / "new_batch.csv"     # incoming batch
REPORT_DIR = ROOT / "reports"                                   # all JSON/CSV reports
CANDIDATE_DIR = ROOT / "artifacts" / "candidate"                # candidate model folder
CHAMPION_PATH = ROOT / "artifacts" / "credit.joblib"            # existing champion

TARGET = "credit_risk"   # original column: 1 = good credit, 0 = bad credit
ID_COLUMN = "row_id"     # id column of the new batch (not a model feature)


def make_xy(df):
    """Split a dataframe into X (features) and y (1 = bad credit risk).

    Same rule as src/train_credit.py, so the model sees the same labels.
    """
    y = (df[TARGET].astype(int) == 0).astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def load_original_split():
    """Re-create the SAME train/test split that train_credit.py used."""
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET])
    X, y = make_xy(df)
    # same settings as train_credit.py (test_size=0.20, random_state=42, stratify)
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


def get_metrics(model, X, y):
    """Accuracy, Precision, Recall, F1 (positive class = bad credit)."""
    pred = model.predict(X)
    return {
        "rows": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y, pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path):
    with open(path) as f:
        return json.load(f)
