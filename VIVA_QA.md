# Viva Questions and Answers

**Q1. Explain the complete workflow in simple words.**
New labelled data arrives -> we validate it -> we test our existing (champion) model on it -> we retrain a new (candidate) model on old + new data -> we compare both models -> a quality gate decides whether the candidate is good enough. The decision is saved in a JSON file. Everything runs with `python -m src.run_workflow`.

**Q2. Why is retraining needed?**
A model learns from old data. Over time customers change (income, loan amounts, economy), so the model can slowly become less accurate (data drift / model decay). Retraining with new labelled data helps it stay up to date. But a retrained model is not always better, so we compare it before using it.

**Q3. What is a champion and what is a candidate?**
Champion = the model currently in use (`artifacts/credit.joblib`, alias `champion` in MLflow). Candidate = the newly retrained model, saved separately in `artifacts/candidate/`. It only replaces the champion if it passes the quality gate and someone approves.

**Q4. What does data validation check and why?**
Required columns present, missing values, duplicate rows/ids, number of rows, and target distribution. Bad data gives a bad model, so we stop the workflow if the data fails (e.g. missing target, duplicates).

**Q5. Why do you evaluate on the hold-out test set and not only on the new batch?**
The new batch has only 8 rows, so its metrics change a lot with one wrong prediction. The old test set (192 rows after removing the 8 new rows) is bigger, so the comparison is more reliable. The 8 new rows are removed from it so the candidate is not tested on rows it trained on.

**Q6. Which metrics did you use and what do they mean?**
Positive class = bad credit risk. Accuracy = correct predictions / all. Precision = of the customers predicted bad, how many were really bad. Recall = of the really bad customers, how many we caught. F1 = balance of precision and recall.

**Q7. Why did you select F1-score for the quality gate?**
Only about 30% of customers are bad credit, so accuracy can look high even if the model misses them. F1 uses both precision and recall of the bad-credit class. Our original `train_credit.py` also selects the best model by F1, so this is consistent.

**Q8. How does the quality gate work?**
Two rules: (1) candidate F1 must be higher than champion F1; (2) candidate recall must not fall by more than 0.05 (missing a bad customer is costly). If both pass -> "eligible for promotion", else -> "keep existing champion". The code is the `decide()` function in `src/quality_gate.py`.

**Q9. Why does the gate not replace the champion automatically?**
Replacing a live model is risky, so a human should approve it. The gate only records the decision (`champion_overwritten: false`). We also compare file hashes to prove the champion file did not change.

**Q10. How do you reuse the existing preprocessing pipeline?**
`clone(champion)` makes an untrained copy of the same Pipeline (imputer + scaler + one-hot encoder + classifier with same settings). We call `.fit()` on old + new data, so the preprocessing is identical.

**Q11. How do you get the old training data?**
`load_original_split()` in `src/common.py` repeats the same `train_test_split` (test_size=0.2, random_state=42, stratify) as `train_credit.py`, so we get the same 800 training rows.

**Q12. How does MLflow track the experiments?**
MLflow stores runs in `mlflow.db` (SQLite). Each run has parameters (rows used), metrics (holdout accuracy, precision, recall, F1) and the saved model. Our workflow adds two runs to the existing `credit-scoring` experiment: `retrain-candidate` and `quality-gate-decision` (tagged with the decision). View with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

**Q13. What is the MLflow Model Registry and alias?**
It stores versions of a model by name. The alias `champion` points to the version used in production (`models:/credit-scoring-model@champion`). `src/model_utils.py` loads this and falls back to the local joblib file if the registry is empty.

**Q14. What did your actual result show?**
See the report: the candidate F1 on the hold-out set was slightly lower than the champion's, so the gate kept the champion. This is a valid outcome; it shows the gate protects against a worse model. Only 8 new rows were added to 800, so a big change was not expected.

**Q15. What is added to GitHub Actions?**
One step `python -m src.run_workflow` after model training, and `pytest` now also runs `tests/test_monitoring.py` (tests for validation and the quality gate).

**Q16. What are the limitations?**
Very small new batch (8 rows); the batch is taken from the same dataset so real drift is not shown; promotion is manual; only one metric split is used.
