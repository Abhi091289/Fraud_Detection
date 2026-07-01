# FraudGuard — Real-Time Credit Card Fraud Detection

A production-grade fraud detection system trained on **real credit card transaction data** with a live streaming dashboard.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | ULB Machine Learning Group (OpenML / Kaggle) |
| Transactions | 284,807 |
| Fraud | 492 (0.172%) |
| Features | V1–V28 (PCA-anonymised) + Amount + Time |
| Download | Free — no account or credentials needed |

Real European credit card transactions from September 2013. Features V1–V28 are PCA-transformed to protect cardholder privacy.

---

## Architecture

```
[creditcard.csv]
      ↓
[Kafka Producer]  ──→  [Kafka: transactions.raw]
                                ↓
                       [Kafka Consumer]
                                ↓
                   [FastAPI /predict endpoint]
                                ↓
                     [XGBoost + RobustScaler]
                                ↓
             ┌── is_fraud=true ──→ [Kafka: fraud.alerts]
             └── SSE broadcast ──→ [Browser Dashboard]

[MLflow Docker]  ← experiment tracking + model registry
```

---

## Run Order (Step by Step)

### Step 1 — Setup virtual environment

```powershell
cd <project-folder>
py -3.14 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### Step 2 — Download real dataset (~143 MB, once only)

```powershell
python data/download_dataset.py
```

Expected:
```
Dataset saved → data/creditcard.csv
    Rows        : 284,807
    Fraud (1)   : 492  (0.173%)
    Legit (0)   : 284,315
```

---

### Step 3 — Start Docker MLflow (keeps running in background)

```powershell
docker compose up mlflow --build -d
```

Wait 30 seconds, then verify:
```powershell
curl http://localhost:5000/health
```

Open **http://localhost:5000** to see the MLflow UI.

> MLflow runs inside Docker using Python 3.11.
> Do NOT run `mlflow server ...` in your venv — it crashes on Python 3.14.

---

### Step 4 — Train the model (~3–5 min)

```powershell
venv\Scripts\activate
python training/train.py
```

Expected:
```
Training RobustScaler + XGBoost (300 trees) ...
  Results at threshold 0.42:
    ROC-AUC  : 0.9989
    PR-AUC   : 0.8541
    Precision: 0.8571
    Recall   : 0.8980
    F1       : 0.8771
  Production model ready: mlartifacts/production/
  Next step: docker compose up --build
```

---

### Step 5 — Start full stack

```powershell
docker compose up --build
```

Wait ~60 seconds for Kafka to be ready. You will see:
```
fg_kafka     | [KafkaServer] started
fg_api       | Model loaded and ready
fg_producer  | Streaming to 'transactions.raw' at 10/sec
fg_consumer  | Consuming 'transactions.raw'
```

---

### Step 6 — Open the Dashboard

```
http://localhost:8000
```

---

## Dashboard Features

| Feature | Description |
|---------|-------------|
| Live Feed | Every real transaction scored in real-time |
| Fraud Alerts | Instant alert panel with amount and probability |
| True Labels | TP / TN / FP / FN from real dataset labels |
| Precision / Recall | Live accuracy metrics updated per transaction |
| Test Panel | 6 pre-built fraud and legit scenarios |
| SSE Streaming | Browser gets instant push — no polling |

---

## Test Scenarios (Dashboard → Test Panel)

| Scenario | Expected | Why |
|----------|----------|-----|
| Midnight ATM | FRAUD | Late-night + V14 anomaly |
| Card Cloning | FRAUD | Stolen card PCA pattern |
| Grocery Store | SAFE | Normal spend |
| Coffee Shop | SAFE | Small daily amount |
| Overseas Fraud | FRAUD | Very high risk V features |
| Salary Day | SAFE | Normal large transfer |

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Live dashboard |
| `GET /health` | Model status |
| `POST /predict?txn_id=X&true_label=Y` | Score + broadcast to dashboard |
| `POST /api/test` | Manual test (no broadcast) |
| `GET /api/stream` | SSE event stream |
| `GET /docs` | Swagger UI |

---

## Retrain with a new model

```powershell
# 1. Make sure Docker MLflow is running
docker compose up mlflow -d

# 2. Retrain
venv\Scripts\activate
python training/train.py

# 3. Restart API to load new model
docker compose restart api
```

---

## Common Errors

| Error | Fix |
|-------|-----|
| `Dataset not found` | Run `python data/download_dataset.py` |
| `model_loaded: false` | Run `python training/train.py` first |
| `Traversable ImportError` | Do NOT run `mlflow server` in venv — use Docker MLflow |
| `WinError 10022` | Fixed — `--workers 1` in mlflow/Dockerfile |
| `Invalid Host header` | Fixed — API loads model from file, not MLflow registry |
| Kafka not ready | Wait 60s after `docker compose up --build` |

---

## Stop Everything

```powershell
docker compose down
```
