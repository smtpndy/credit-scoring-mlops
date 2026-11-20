# Credit Scoring & Fraud Detection — Two-Dashboard MLOps Starter

This repository contains **two separate dashboards** backed by one FastAPI service:
- Credit Scoring dashboard: estimates probability of loan default and returns a risk band.
- Fraud Detection dashboard: estimates transaction fraud probability and returns a review decision.

The starter is runnable locally and follows the engineering pattern of your Iris MLOps project:
training script → saved model → FastAPI model serving → pytest → Docker → GitHub Actions.

## Important data note
The included training scripts generate **synthetic demonstration data** with scikit-learn. The models are useful for testing the end-to-end pipeline and dashboard, **not for real lending or fraud decisions**. Replace the synthetic generators with appropriately licensed, documented datasets and perform fairness, privacy, security, calibration and domain validation before any real use.

## 1. Requirements
- Python 3.11+ (3.13 recommended)
- Node.js 20+
- Git
- Docker Desktop (optional)

## 2. Run locally on Windows

### Backend
Open a terminal in the project root:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
python -m src.train_credit
python -m src.train_fraud
uvicorn backend.main:app --reload
```

API: http://127.0.0.1:8000  
Swagger docs: http://127.0.0.1:8000/docs

### Frontend
Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

Use the left navigation to switch between the **Credit Scoring Dashboard** and the **Fraud Detection Dashboard**. They are separate dashboard views, not a combined analytics page.

## 3. Train again
From the project root, with the virtual environment activated:

```powershell
python -m src.train_credit
python -m src.train_fraud
```

Models and evaluation JSON files are written to `artifacts/`. MLflow tracking is enabled locally under `mlruns/` (the training scripts log parameters, metrics and model artifacts).

To view MLflow:

```powershell
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Then open http://127.0.0.1:5000.

## 4. Run tests

```powershell
python -m pytest -v
```

Tests use fake models and do not require the MLflow server to be running.

## 5. Docker

The Docker setup builds the backend and frontend separately:

```powershell
docker compose up --build
```

Frontend: http://localhost:5173  
API docs: http://localhost:8000/docs

For Docker, train the models locally first so the `artifacts/` folder exists:

```powershell
python -m src.train_credit
python -m src.train_fraud
docker compose up --build
```

## 6. Project structure

```text
credit-fraud-two-dashboards/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   └── styles.css
│   ├── Dockerfile
│   ├── package.json
│   └── vite.config.js
├── src/
│   ├── train_credit.py
│   ├── train_fraud.py
│   └── model_utils.py
├── tests/
│   └── test_api.py
├── artifacts/                 # created by training scripts
├── .github/workflows/ci.yml
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## Next milestones (not yet implemented)
1. Replace synthetic data with real, properly licensed datasets.
2. Add a database for prediction history (current API history is in-memory).
3. Add model registry promotion/versioning and champion alias like your Iris project.
4. Add drift/performance monitoring with Evidently or a similar tool.
5. Add authentication, role-based access, audit logging and deployment to AWS.
6. Add a separate Fraud Detection frontend deployment later if you want each dashboard hosted independently.

## API endpoints
- `GET /health`
- `GET /api/credit/metrics`
- `POST /api/credit/predict`
- `GET /api/fraud/metrics`
- `POST /api/fraud/predict`
- `GET /api/history?domain=credit|fraud`

---

## Model Monitoring and Retraining Workflow (MLOps Practical Assessment)

This extends the existing Credit Scoring model (German Credit data). Nothing old was changed.

**Run everything with ONE command (from the project root):**

```bash
python -m src.run_workflow
```

| Step | File | What it does |
|------|------|--------------|
| 1 | `src/create_new_data.py` | Creates a small incoming batch `datasets/new/new_batch.csv` (8 rows) |
| 2 | `src/validate_data.py` | Checks columns, missing values, duplicates, size, target distribution |
| 3 | `src/evaluate_model.py` | Evaluates the existing champion model (Accuracy, Precision, Recall, F1) |
| 4 | `src/retrain_model.py` | Old training data + new batch -> candidate model (saved separately, tracked in MLflow) |
| 5 | `src/quality_gate.py` | Compares champion vs candidate and records the decision |
| - | `src/common.py` | Shared paths, train/test split and metric function |

Outputs: `reports/*.json`, `reports/model_comparison.csv`, `artifacts/candidate/credit_candidate.joblib`.
The champion (`artifacts/credit.joblib`) is never overwritten. Tests: `python -m pytest -v`.
See `VIVA_QA.md` for viva questions and answers.
