"""Task B: evaluate the EXISTING champion model.

We evaluate it on:
  1. the new labelled batch  (Task B requirement)
  2. the old hold-out test set (bigger, so the comparison in the quality gate is fair)
"""
import pandas as pd

from src.common import (
    ID_COLUMN, NEW_DATA_PATH, REPORT_DIR, get_metrics, load_original_split,
    make_xy, save_json,
)
from src.model_utils import load_bundle


def get_holdout_and_new_data():
    """Return (X_new, y_new, X_hold, y_hold). Used by evaluate and retrain scripts."""
    new_df = pd.read_csv(NEW_DATA_PATH)
    X_new, y_new = make_xy(new_df.drop(columns=[ID_COLUMN]))

    X_train, X_test, y_train, y_test = load_original_split()

    # the new batch was taken from the old test split, so remove those rows
    # from the hold-out set (otherwise the candidate would be tested on rows it trained on)
    keep = ~X_test.index.isin(new_df[ID_COLUMN])
    return X_new, y_new, X_test[keep], y_test[keep]


def main():
    champion = load_bundle("credit")   # existing function of the project

    X_new, y_new, X_hold, y_hold = get_holdout_and_new_data()

    report = {
        "model": "champion (existing credit model)",
        "positive_class": "bad credit risk",
        "new_batch_metrics": get_metrics(champion, X_new, y_new),
        "holdout_metrics": get_metrics(champion, X_hold, y_hold),
    }
    save_json(report, REPORT_DIR / "champion_evaluation_report.json")

    for name in ["new_batch_metrics", "holdout_metrics"]:
        m = report[name]
        print(f"\nChampion on {name} ({m['rows']} rows)")
        print(f"  Accuracy : {m['accuracy']}")
        print(f"  Precision: {m['precision']}")
        print(f"  Recall   : {m['recall']}")
        print(f"  F1-score : {m['f1_score']}")


if __name__ == "__main__":
    main()
