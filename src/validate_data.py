"""Task A (part 2): validate the new data batch.

Checks: required columns, missing values, duplicates, dataset size,
target distribution. Result is saved in reports/data_validation_report.json.
"""
import pandas as pd

from src.common import (
    DATA_PATH, ID_COLUMN, NEW_DATA_PATH, REPORT_DIR, TARGET, save_json,
)

MIN_ROWS = 2  # a batch smaller than this is not useful


def validate(df, required_columns):
    """Return a report dictionary. Errors stop the workflow, warnings do not."""
    errors = []
    warnings = []

    # 1. required columns
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")

    # 2. dataset size
    if len(df) < MIN_ROWS:
        errors.append(f"Too few rows: {len(df)} (minimum {MIN_ROWS})")

    # 3. missing values
    missing_values = df.isnull().sum()
    missing_values = {c: int(n) for c, n in missing_values.items() if n > 0}
    if TARGET in missing_values:
        errors.append("Target column has missing values (cannot train without labels)")
    other_missing = [c for c in missing_values if c != TARGET]
    if other_missing:
        # the existing pipeline has an imputer, so this is only a warning
        warnings.append(f"Missing feature values in {other_missing} (pipeline imputer will fill them)")

    # 4. duplicates
    feature_df = df.drop(columns=[ID_COLUMN], errors="ignore")
    duplicate_rows = int(feature_df.duplicated().sum())
    if duplicate_rows > 0:
        errors.append(f"{duplicate_rows} duplicate rows found")

    duplicate_ids = 0
    if ID_COLUMN in df.columns:
        duplicate_ids = int(df[ID_COLUMN].duplicated().sum())
        if duplicate_ids > 0:
            errors.append(f"{duplicate_ids} duplicate ids found")

    # 5. target distribution (0 = bad credit, 1 = good credit)
    target_distribution = {}
    if TARGET in df.columns:
        counts = df[TARGET].value_counts()
        target_distribution = {
            "bad_credit(0)": int(counts.get(0, 0)),
            "good_credit(1)": int(counts.get(1, 0)),
        }
        wrong_values = set(df[TARGET].dropna().unique()) - {0, 1}
        if wrong_values:
            errors.append(f"Target has unexpected values: {wrong_values}")

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "target_distribution": target_distribution,
        "warnings": warnings,
        "errors": errors,
        "status": "FAILED" if errors else "PASSED",
    }


def main():
    new_df = pd.read_csv(NEW_DATA_PATH)

    # the required columns = all columns of the original training file
    required_columns = list(pd.read_csv(DATA_PATH, nrows=0).columns)

    report = validate(new_df, required_columns)
    save_json(report, REPORT_DIR / "data_validation_report.json")

    print("Dataset size:", report["rows"], "rows,", report["columns"], "columns")
    print("Target distribution:", report["target_distribution"])
    print("Missing values:", report["missing_values"] or "none")
    print("Duplicate rows:", report["duplicate_rows"], "| Duplicate ids:", report["duplicate_ids"])
    for w in report["warnings"]:
        print("WARNING:", w)
    for e in report["errors"]:
        print("ERROR:", e)
    print("Validation status:", report["status"])

    return report["status"] == "PASSED"


if __name__ == "__main__":
    main()
