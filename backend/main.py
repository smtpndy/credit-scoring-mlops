
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import sys
import json
import sqlite3

import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_utils import load_bundle, load_metrics


# --------------------------------------------------
# PATHS AND FEATURE COLUMNS
# --------------------------------------------------

DATA_DIR = ROOT / "datasets" / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
SHAP_DIR = ARTIFACT_DIR

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = ARTIFACT_DIR / "prediction_history.db"

CREDIT_FEATURES = [
    "status", "duration", "credit_history", "purpose", "amount",
    "savings", "employment_duration", "installment_rate",
    "personal_status_sex", "other_debtors", "present_residence",
    "property", "age", "other_installment_plans", "housing",
    "number_credits", "job", "people_liable", "telephone",
    "foreign_worker",
]

FRAUD_FEATURES = [
    f"V{i}" for i in range(1, 29)
] + ["Amount"]


# --------------------------------------------------
# SQLITE DATABASE
# --------------------------------------------------

def get_db_connection():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_history_database():
    with get_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                probability REAL NOT NULL,
                decision TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_domain
            ON prediction_history(domain)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_timestamp
            ON prediction_history(timestamp)
        """)

        connection.commit()

    print(f"SQLite database initialized: {DB_PATH}")


def load_history_from_database():
    return get_history_from_database(limit=1000)


def get_history_from_database(domain=None, limit=1000):
    with get_db_connection() as connection:
        if domain:
            rows = connection.execute("""
                SELECT id, domain, probability, decision, timestamp
                FROM prediction_history
                WHERE domain = ?
                ORDER BY id DESC
                LIMIT ?
            """, (domain, limit)).fetchall()
        else:
            rows = connection.execute("""
                SELECT id, domain, probability, decision, timestamp
                FROM prediction_history
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()

    return [dict(row) for row in rows]


def save_prediction_to_database(
    domain,
    probability,
    decision,
    timestamp
):
    with get_db_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO prediction_history (
                domain, probability, decision, timestamp
            )
            VALUES (?, ?, ?, ?)
        """, (
            domain,
            float(probability),
            str(decision),
            timestamp,
        ))

        connection.commit()
        return cursor.lastrowid


initialize_history_database()
history = load_history_from_database()


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="Credit Scoring & Fraud Detection API",
    description="Credit scoring and fraud detection using Kaggle datasets.",
    version="4.2.0",
)

# UPDATED CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# MODELS AND CACHES
# --------------------------------------------------

models = {
    "credit": None,
    "fraud": None,
}

explainers = {
    "credit": None,
    "fraud": None,
}


@app.on_event("startup")
def load_models():
    for domain in ("credit", "fraud"):
        if models[domain] is None:
            try:
                models[domain] = load_bundle(domain)
                print(f"{domain.title()} model loaded successfully.")
            except FileNotFoundError:
                models[domain] = None
                print(f"{domain.title()} model artifact not found.")

    restored_count = len(get_history_from_database())
    print(f"Restored {restored_count} predictions from SQLite.")


# --------------------------------------------------
# INPUT SCHEMAS
# --------------------------------------------------

class CreditInput(BaseModel):
    status: str = Field(min_length=1)
    duration: int = Field(ge=1, le=120)
    credit_history: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    amount: float = Field(gt=0, le=10_000_000)
    savings: str = Field(min_length=1)
    employment_duration: str = Field(min_length=1)
    installment_rate: int = Field(ge=1, le=4)
    personal_status_sex: str = Field(min_length=1)
    other_debtors: str = Field(min_length=1)
    present_residence: int = Field(ge=1, le=4)
    property: str = Field(min_length=1)
    age: int = Field(ge=18, le=100)
    other_installment_plans: str = Field(min_length=1)
    housing: str = Field(min_length=1)
    number_credits: int = Field(ge=1, le=10)
    job: str = Field(min_length=1)
    people_liable: int = Field(ge=1, le=2)
    telephone: str = Field(min_length=1)
    foreign_worker: str = Field(min_length=1)


FraudInput = create_model(
    "FraudInput",
    **{
        feature: (float, ...)
        for feature in FRAUD_FEATURES
    },
)


# --------------------------------------------------
# GENERAL HELPERS
# --------------------------------------------------

def get_model(domain):
    model = models.get(domain)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{domain.title()} model is not trained or loaded. "
                "Run the matching training script from the project root."
            ),
        )

    return model


def input_to_dict(data):
    if hasattr(data, "model_dump"):
        return data.model_dump()

    return data.dict()


def record(domain, probability, decision):
    probability = round(float(probability), 4)
    timestamp = datetime.now(timezone.utc).isoformat()

    prediction_id = save_prediction_to_database(
        domain=domain,
        probability=probability,
        decision=decision,
        timestamp=timestamp,
    )

    entry = {
        "id": prediction_id,
        "domain": domain,
        "probability": probability,
        "decision": decision,
        "timestamp": timestamp,
    }

    history.insert(0, entry)
    del history[1000:]

    return entry


def get_positive_class_values(values):
    if isinstance(values, list):
        values = values[1]

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 1]

    if values.ndim == 2:
        values = values[0]

    return values.flatten()


# --------------------------------------------------
# SHAP EXPLAINER INITIALIZATION
# --------------------------------------------------

def initialize_credit_explainer():
    print("Initializing Credit SHAP explainer...")

    pipeline = get_model("credit")
    dataset_path = DATA_DIR / "GermanCredit.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Credit dataset not found: {dataset_path}"
        )

    df = pd.read_csv(
        dataset_path,
        usecols=CREDIT_FEATURES,
    )

    background_df = df.sample(
        n=min(300, len(df)),
        random_state=42,
    )

    preprocessor = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]

    background = preprocessor.transform(background_df)

    if hasattr(background, "toarray"):
        background = background.toarray()

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.LinearExplainer(
        classifier,
        background,
    )

    explainers["credit"] = {
        "explainer": explainer,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }

    print("Credit SHAP explainer initialized.")


def initialize_fraud_explainer():
    print("Initializing Fraud SHAP explainer...")

    pipeline = get_model("fraud")
    dataset_path = DATA_DIR / "creditcard_2023.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Fraud dataset not found: {dataset_path}"
        )

    df = pd.read_csv(
        dataset_path,
        usecols=FRAUD_FEATURES,
    )

    background_df = df.sample(
        n=min(1000, len(df)),
        random_state=42,
    )

    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]

    background = imputer.transform(background_df)
    background = scaler.transform(background)
    background = background[:min(100, len(background))]

    explainer = shap.TreeExplainer(
        classifier,
        data=background,
        feature_perturbation="interventional",
        model_output="raw",
    )

    explainers["fraud"] = {
        "explainer": explainer,
        "imputer": imputer,
        "scaler": scaler,
        "feature_names": FRAUD_FEATURES,
    }

    print("Fraud SHAP explainer initialized.")


def get_explainer(domain):
    if explainers[domain] is None:
        try:
            if domain == "credit":
                initialize_credit_explainer()
            elif domain == "fraud":
                initialize_fraud_explainer()

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Unable to initialize {domain} SHAP: {str(exc)}",
            )

    return explainers[domain]


# --------------------------------------------------
# INDIVIDUAL SHAP EXPLANATIONS
# --------------------------------------------------

def explain_individual_prediction(domain, features):
    cached = get_explainer(domain)

    try:
        if domain == "credit":
            transformed = cached["preprocessor"].transform(features)

            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            shap_values = cached["explainer"].shap_values(transformed)

            feature_names = cached["feature_names"]
            display_values = np.asarray(transformed)[0]

            positive_class = "Bad credit"

        else:
            transformed = cached["imputer"].transform(features)
            transformed = cached["scaler"].transform(transformed)

            shap_values = cached["explainer"].shap_values(transformed)

            feature_names = cached["feature_names"]
            display_values = features.iloc[0].to_numpy()

            positive_class = "Fraud"

        values = get_positive_class_values(shap_values)

        if len(values) != len(feature_names):
            raise ValueError(
                "SHAP feature count does not match model feature count."
            )

        results = []

        for name, value, input_value in zip(
            feature_names,
            values,
            display_values,
        ):
            contribution = float(value)

            results.append({
                "feature": str(name),
                "input_value": float(input_value),
                "shap_value": round(contribution, 8),
                "direction": (
                    "towards_positive_class"
                    if contribution > 0
                    else "away_from_positive_class"
                    if contribution < 0
                    else "neutral"
                ),
            })

        results.sort(
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )

        return {
            "positive_class": positive_class,
            "explanation_type": "individual_prediction",
            "top_features": results[:10],
            "all_features": results,
            "interpretation": (
                "Positive SHAP values push the prediction towards "
                "the positive class. Negative values push it away. "
                "SHAP values are model-output contributions, "
                "not probability percentages."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{domain.title()} SHAP calculation failed: {str(exc)}",
        )


# --------------------------------------------------
# GENERAL ENDPOINTS
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Credit & Fraud API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    stored_predictions = len(get_history_from_database())

    return {
        "status": "healthy",
        "credit_model_loaded": models["credit"] is not None,
        "fraud_model_loaded": models["fraud"] is not None,
        "credit_shap_initialized": explainers["credit"] is not None,
        "fraud_shap_initialized": explainers["fraud"] is not None,
        "database": "SQLite",
        "database_path": str(DB_PATH),
        "stored_predictions": stored_predictions,
    }


# --------------------------------------------------
# CREDIT SCORING ENDPOINTS
# --------------------------------------------------

@app.get("/api/credit/metrics")
def credit_metrics():
    return load_metrics("credit")


@app.post("/api/credit/predict")
def credit_predict(data: CreditInput):
    model = get_model("credit")

    features = pd.DataFrame(
        [input_to_dict(data)],
        columns=CREDIT_FEATURES,
    )

    try:
        probability = float(model.predict_proba(features)[0][1])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Credit model prediction failed: {str(exc)}",
        )

    risk = (
        "High Risk"
        if probability >= 0.65
        else "Medium Risk"
        if probability >= 0.35
        else "Low Risk"
    )

    decision = (
        "Manual review recommended"
        if risk == "High Risk"
        else "Eligible for further assessment"
    )

    record("credit", probability, risk)

    return {
        "default_probability": round(probability, 4),
        "risk_band": risk,
        "message": decision,
    }


@app.post("/api/credit/explain")
def credit_explain(data: CreditInput):
    model = get_model("credit")

    features = pd.DataFrame(
        [input_to_dict(data)],
        columns=CREDIT_FEATURES,
    )

    try:
        probability = float(model.predict_proba(features)[0][1])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Credit prediction failed: {str(exc)}",
        )

    risk = (
        "High Risk"
        if probability >= 0.65
        else "Medium Risk"
        if probability >= 0.35
        else "Low Risk"
    )

    explanation = explain_individual_prediction(
        "credit",
        features,
    )

    record("credit", probability, risk)

    return {
        "default_probability": round(probability, 4),
        "risk_band": risk,
        "message": (
            "Manual review recommended"
            if risk == "High Risk"
            else "Eligible for further assessment"
        ),
        "explanation": explanation,
    }


# --------------------------------------------------
# FRAUD DETECTION ENDPOINTS
# --------------------------------------------------

@app.get("/api/fraud/metrics")
def fraud_metrics():
    return load_metrics("fraud")


@app.post("/api/fraud/predict")
def fraud_predict(data: FraudInput):
    model = get_model("fraud")

    features = pd.DataFrame(
        [input_to_dict(data)],
        columns=FRAUD_FEATURES,
    )

    try:
        probability = float(model.predict_proba(features)[0][1])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Fraud model prediction failed: {str(exc)}",
        )

    decision = (
        "Flag for investigation"
        if probability >= 0.5
        else "No fraud signal detected"
    )

    record(
        "fraud",
        probability,
        "Flagged" if probability >= 0.5 else "Normal",
    )

    return {
        "fraud_probability": round(probability, 4),
        "decision": decision,
    }


@app.post("/api/fraud/explain")
def fraud_explain(data: FraudInput):
    model = get_model("fraud")

    features = pd.DataFrame(
        [input_to_dict(data)],
        columns=FRAUD_FEATURES,
    )

    try:
        probability = float(model.predict_proba(features)[0][1])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Fraud prediction failed: {str(exc)}",
        )

    decision = (
        "Flag for investigation"
        if probability >= 0.5
        else "No fraud signal detected"
    )

    explanation = explain_individual_prediction(
        "fraud",
        features,
    )

    record(
        "fraud",
        probability,
        "Flagged" if probability >= 0.5 else "Normal",
    )

    return {
        "fraud_probability": round(probability, 4),
        "decision": decision,
        "explanation": explanation,
    }


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

@app.get("/api/history")
def get_history(domain: Optional[str] = None):
    if domain and domain not in ("credit", "fraud"):
        raise HTTPException(
            status_code=400,
            detail="domain must be credit or fraud",
        )

    return get_history_from_database(domain=domain)


# --------------------------------------------------
# MONITORING: SUMMARY
# --------------------------------------------------

@app.get("/api/monitoring/summary")
def monitoring_summary():
    records = get_history_from_database()

    total = len(records)

    credit_records = [
        item for item in records
        if item["domain"] == "credit"
    ]

    fraud_records = [
        item for item in records
        if item["domain"] == "fraud"
    ]

    high_risk_credit = [
        item for item in credit_records
        if item["decision"] == "High Risk"
    ]

    medium_risk_credit = [
        item for item in credit_records
        if item["decision"] == "Medium Risk"
    ]

    low_risk_credit = [
        item for item in credit_records
        if item["decision"] == "Low Risk"
    ]

    flagged_fraud = [
        item for item in fraud_records
        if item["decision"] == "Flagged"
    ]

    normal_fraud = [
        item for item in fraud_records
        if item["decision"] == "Normal"
    ]

    average_probability = (
        float(np.mean([
            item["probability"]
            for item in records
        ]))
        if total else 0.0
    )

    return {
        "total_predictions": total,

        "prediction_counts": {
            "credit": len(credit_records),
            "fraud": len(fraud_records),
        },

        "credit_risk_distribution": {
            "high_risk": len(high_risk_credit),
            "medium_risk": len(medium_risk_credit),
            "low_risk": len(low_risk_credit),
        },

        "fraud_distribution": {
            "flagged": len(flagged_fraud),
            "normal": len(normal_fraud),
        },

        "alerts": {
            "high_risk_credit": len(high_risk_credit),
            "fraud_flagged": len(flagged_fraud),
            "total_alerts": (
                len(high_risk_credit) + len(flagged_fraud)
            ),
        },

        "average_probability": round(average_probability, 4),
        "recent_predictions": records[:10],
    }


# --------------------------------------------------
# MONITORING: DAILY TRENDS
# --------------------------------------------------

@app.get("/api/monitoring/trends")
def monitoring_trends(domain: Optional[str] = None):
    if domain and domain not in ("credit", "fraud"):
        raise HTTPException(
            status_code=400,
            detail="domain must be credit or fraud",
        )

    filtered_history = get_history_from_database(domain=domain)
    daily_data = {}

    for item in filtered_history:
        try:
            timestamp = datetime.fromisoformat(item["timestamp"])
            date_key = timestamp.date().isoformat()
        except (ValueError, TypeError):
            continue

        if date_key not in daily_data:
            daily_data[date_key] = {
                "date": date_key,
                "credit": 0,
                "fraud": 0,
                "total": 0,
            }

        daily_data[date_key][item["domain"]] += 1
        daily_data[date_key]["total"] += 1

    trends = sorted(
        daily_data.values(),
        key=lambda item: item["date"],
    )

    return {
        "domain": domain or "all",
        "total_days": len(trends),
        "trends": trends,
    }


# --------------------------------------------------
# GLOBAL SHAP FEATURE IMPORTANCE
# --------------------------------------------------

def load_shap_importance(domain):
    if domain not in ("credit", "fraud"):
        raise HTTPException(
            status_code=400,
            detail="Invalid domain",
        )

    path = SHAP_DIR / f"{domain}_shap_importance.json"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{domain.title()} SHAP results not found",
        )

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load SHAP results: {str(exc)}",
        )


@app.get("/api/credit/shap")
def credit_shap():
    return load_shap_importance("credit")


@app.get("/api/fraud/shap")
def fraud_shap():
    return load_shap_importance("fraud")
