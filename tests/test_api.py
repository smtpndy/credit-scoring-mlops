
from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.main import app, models


# --------------------------------------------------
# FAKE MODEL FOR API TESTING
# --------------------------------------------------

class FakeModel:
    def __init__(self, probability=0.2):
        self.probability = probability

    def predict_proba(self, features):
        return np.array([
            [1 - self.probability, self.probability]
        ])


# --------------------------------------------------
# TEST INPUTS - REAL KAGGLE FEATURES
# --------------------------------------------------

CREDIT_PAYLOAD = {
    "status": "A11",
    "duration": 24,
    "credit_history": "A34",
    "purpose": "A43",
    "amount": 20000,
    "savings": "A65",
    "employment_duration": "A73",
    "installment_rate": 2,
    "personal_status_sex": "A93",
    "other_debtors": "A101",
    "present_residence": 2,
    "property": "A121",
    "age": 30,
    "other_installment_plans": "A143",
    "housing": "A152",
    "number_credits": 1,
    "job": "A173",
    "people_liable": 1,
    "telephone": "A191",
    "foreign_worker": "A201"
}


FRAUD_PAYLOAD = {
    **{
        f"V{i}": 0.0
        for i in range(1, 29)
    },
    "Amount": 2500.0
}


# --------------------------------------------------
# FIXTURES
# --------------------------------------------------

@pytest.fixture(autouse=True)
def inject_models():
    old = models.copy()

    models["credit"] = FakeModel(0.2)
    models["fraud"] = FakeModel(0.8)

    yield

    models.update(old)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------
# HEALTH TEST
# --------------------------------------------------

def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# --------------------------------------------------
# CREDIT PREDICTION TEST
# --------------------------------------------------

def test_credit_prediction(client):
    response = client.post(
        "/api/credit/predict",
        json=CREDIT_PAYLOAD
    )

    assert response.status_code == 200

    result = response.json()

    assert result["risk_band"] == "Low Risk"
    assert result["default_probability"] == 0.2


# --------------------------------------------------
# FRAUD PREDICTION TEST
# --------------------------------------------------

def test_fraud_prediction(client):
    response = client.post(
        "/api/fraud/predict",
        json=FRAUD_PAYLOAD
    )

    assert response.status_code == 200

    result = response.json()

    assert result["decision"] == "Flag for investigation"
    assert result["fraud_probability"] == 0.8


# --------------------------------------------------
# INVALID CREDIT INPUT TEST
# --------------------------------------------------

def test_invalid_credit_input(client):
    invalid_payload = CREDIT_PAYLOAD.copy()

    invalid_payload["amount"] = -10

    response = client.post(
        "/api/credit/predict",
        json=invalid_payload
    )

    assert response.status_code == 422
