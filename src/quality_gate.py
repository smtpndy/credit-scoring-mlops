"""Task D: quality gate - compare champion and candidate and record a decision.

Selected metric: F1-score (on the hold-out test set).
Why F1?  - Only ~30% of customers are bad credit, so accuracy can look good
           even when the model misses bad customers.
         - F1 balances Precision and Recall of the bad-credit class.
Extra safety rule: Recall must not drop by more than 0.05, because
approving a customer who will default is the costly mistake for a bank.

The script only RECORDS the decision. It never replaces the champion.
"""
import mlflow
import pandas as pd

import src.model_utils  # noqa: F401  (just sets the MLflow tracking URI)
from src.common import REPORT_DIR, load_json, save_json

MIN_F1_GAIN = 0.0        # candidate F1 must be better than champion F1 by more than this
MAX_RECALL_DROP = 0.05   # candidate recall may be at most this much lower


def decide(champion, candidate):
    """champion / candidate = metric dictionaries. Returns the decision dictionary."""
    f1_gain = round(candidate["f1_score"] - champion["f1_score"], 4)
    recall_change = round(candidate["recall"] - champion["recall"], 4)

    rule1 = f1_gain > MIN_F1_GAIN
    rule2 = recall_change >= -MAX_RECALL_DROP

    eligible = rule1 and rule2
    if eligible:
        action = "Candidate is ELIGIBLE for promotion (manual approval needed)"
    else:
        action = "KEEP existing champion"

    return {
        "selected_metric": "f1_score",
        "rule_1": f"candidate F1 must be higher than champion F1 by more than {MIN_F1_GAIN}",
        "rule_1_passed": rule1,
        "rule_2": f"candidate recall must not drop by more than {MAX_RECALL_DROP}",
        "rule_2_passed": rule2,
        "f1_gain": f1_gain,
        "recall_change": recall_change,
        "eligible_for_promotion": eligible,
        "action": action,
        "champion_overwritten": False,   # we never overwrite automatically
    }


def main():
    champ = load_json(REPORT_DIR / "champion_evaluation_report.json")["holdout_metrics"]
    cand_report = load_json(REPORT_DIR / "candidate_training_report.json")
    cand = cand_report["holdout_metrics"]

    decision = decide(champ, cand)
    decision["champion_metrics"] = champ
    decision["candidate_metrics"] = cand
    decision["evaluated_on"] = f"hold-out test set ({champ['rows']} rows)"
    decision["candidate_mlflow_run_id"] = cand_report["mlflow_run_id"]
    save_json(decision, REPORT_DIR / "quality_gate_decision.json")

    # comparison table (also saved as csv)
    table = pd.DataFrame({"champion": champ, "candidate": cand}).loc[
        ["accuracy", "precision", "recall", "f1_score"]
    ]
    table["difference"] = (table["candidate"] - table["champion"]).round(4)
    table.to_csv(REPORT_DIR / "model_comparison.csv")
    save_json(table.to_dict(), REPORT_DIR / "model_comparison.json")

    print("\nMODEL COMPARISON (hold-out test set)")
    print(table.to_string())
    print("\nRule 1 (F1 improved)        :", "PASS" if decision["rule_1_passed"] else "FAIL")
    print("Rule 2 (recall not dropped) :", "PASS" if decision["rule_2_passed"] else "FAIL")
    print("DECISION:", decision["action"])

    # record the decision in MLflow too
    mlflow.set_experiment("credit-scoring")
    with mlflow.start_run(run_name="quality-gate-decision"):
        mlflow.log_metrics({
            "champion_f1": champ["f1_score"],
            "candidate_f1": cand["f1_score"],
            "f1_gain": decision["f1_gain"],
            "eligible_for_promotion": int(decision["eligible_for_promotion"]),
        })
        mlflow.set_tag("decision", decision["action"])
        mlflow.log_artifact(str(REPORT_DIR / "quality_gate_decision.json"))


if __name__ == "__main__":
    main()
