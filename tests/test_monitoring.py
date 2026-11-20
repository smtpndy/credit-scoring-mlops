"""Simple tests for the validation and quality-gate functions."""
import pandas as pd

from src.quality_gate import decide
from src.validate_data import validate

COLUMNS = ["age", "amount", "credit_risk"]


def test_validation_passes_for_good_data():
    df = pd.DataFrame({"age": [25, 40], "amount": [1000, 2000], "credit_risk": [1, 0]})
    report = validate(df, COLUMNS)
    assert report["status"] == "PASSED"


def test_validation_fails_for_missing_column():
    df = pd.DataFrame({"age": [25, 40], "credit_risk": [1, 0]})
    report = validate(df, COLUMNS)
    assert report["status"] == "FAILED"


def test_validation_fails_for_duplicates():
    df = pd.DataFrame({"age": [25, 25], "amount": [1000, 1000], "credit_risk": [1, 1]})
    report = validate(df, COLUMNS)
    assert report["duplicate_rows"] == 1
    assert report["status"] == "FAILED"


def test_validation_fails_for_missing_target():
    df = pd.DataFrame({"age": [25, 40], "amount": [1000, 2000], "credit_risk": [1, None]})
    report = validate(df, COLUMNS)
    assert report["status"] == "FAILED"


def test_gate_promotes_better_candidate():
    champion = {"f1_score": 0.65, "recall": 0.80}
    candidate = {"f1_score": 0.70, "recall": 0.80}
    result = decide(champion, candidate)
    assert result["eligible_for_promotion"] is True
    assert result["champion_overwritten"] is False


def test_gate_keeps_champion_when_candidate_is_worse():
    champion = {"f1_score": 0.65, "recall": 0.80}
    candidate = {"f1_score": 0.60, "recall": 0.80}
    assert decide(champion, candidate)["eligible_for_promotion"] is False


def test_gate_keeps_champion_when_recall_drops_too_much():
    champion = {"f1_score": 0.65, "recall": 0.80}
    candidate = {"f1_score": 0.70, "recall": 0.60}   # F1 better but recall dropped 0.20
    assert decide(champion, candidate)["eligible_for_promotion"] is False
