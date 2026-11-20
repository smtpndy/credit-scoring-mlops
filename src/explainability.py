
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "raw"
ARTIFACT_DIR = ROOT / "artifacts"

ARTIFACT_DIR.mkdir(exist_ok=True)

CREDIT_FEATURES = [
    "status", "duration", "credit_history", "purpose", "amount",
    "savings", "employment_duration", "installment_rate",
    "personal_status_sex", "other_debtors", "present_residence",
    "property", "age", "other_installment_plans", "housing",
    "number_credits", "job", "people_liable", "telephone",
    "foreign_worker"
]

FRAUD_FEATURES = [
    f"V{i}" for i in range(1, 29)
] + ["Amount"]


def get_positive_class_shap(values):
    """Extract SHAP values for class 1."""
    if isinstance(values, list):
        return np.asarray(values[1])

    values = np.asarray(values)

    if values.ndim == 3:
        return values[:, :, 1]

    return values


def save_importance(name, feature_names, shap_values):
    importance = np.abs(shap_values).mean(axis=0)

    results = [
        {
            "feature": str(feature),
            "importance": float(value)
        }
        for feature, value in zip(feature_names, importance)
    ]

    results.sort(
        key=lambda item: item["importance"],
        reverse=True
    )

    output = ARTIFACT_DIR / f"{name}_shap_importance.json"

    with open(output, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n{name.upper()} SHAP FEATURE IMPORTANCE")
    print("-" * 45)

    for item in results[:15]:
        print(
            f"{item['feature']:<35} "
            f"{item['importance']:.6f}"
        )

    print(f"\nSaved: {output}")


def explain_credit():
    print("\nCalculating CREDIT SHAP values...")

    df = pd.read_csv(DATA_DIR / "GermanCredit.csv")

    X = df[CREDIT_FEATURES]

    if len(X) > 300:
        X = X.sample(n=300, random_state=42)

    pipeline = joblib.load(ARTIFACT_DIR / "credit.joblib")

    preprocessor = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.LinearExplainer(
        classifier,
        X_transformed
    )

    shap_values = explainer.shap_values(X_transformed)

    shap_values = get_positive_class_shap(shap_values)

    save_importance(
        "credit",
        feature_names,
        shap_values
    )


def explain_fraud():
    print("\nCalculating FRAUD SHAP values...")

    df = pd.read_csv(DATA_DIR / "creditcard_2023.csv")

    X = df[FRAUD_FEATURES]

    if len(X) > 1000:
        X = X.sample(n=1000, random_state=42)

    pipeline = joblib.load(ARTIFACT_DIR / "fraud.joblib")

    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]

    X_imputed = imputer.transform(X)
    X_transformed = scaler.transform(X_imputed)

    background = X_transformed[:min(100, len(X_transformed))]
    evaluation = X_transformed

    explainer = shap.TreeExplainer(
        classifier,
        data=background,
        feature_perturbation="interventional",
        model_output="raw"
    )

    shap_values = explainer.shap_values(evaluation)

    shap_values = get_positive_class_shap(shap_values)

    save_importance(
        "fraud",
        FRAUD_FEATURES,
        shap_values
    )


if __name__ == "__main__":
    print("SHAP VERSION:", shap.__version__)

    explain_credit()
    explain_fraud()

    print("\nBoth SHAP analyses completed.")
