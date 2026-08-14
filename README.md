# FraudStream — Real-Time Fraud Detection Platform

> A portfolio-level end-to-end real-time fraud detection system built with Kafka, XGBoost, Redis, PostgreSQL, and Grafana.

---

## Problem Statement

Financial fraud costs billions annually. Traditional batch processing detects fraud hours or days after it occurs. FraudStream detects potentially fraudulent transactions in real time — within milliseconds of them occurring.

---

## System Architecture

```text
Transaction Generator
 ↓
 Apache Kafka
 ↓
 Python Consumer
 ↓         ↓
 Redis      XGBoost Model
 (state)    (prediction)
 ↓
 PostgreSQL
 ↓
 Grafana Dashboard
```

---

## Technology Stack

| Technology     | Purpose                                             |
| -------------- | --------------------------------------------------- |
| Python         | Core programming language                           |
| XGBoost        | Fraud classification model                          |
| Scikit-learn   | Preprocessing + Logistic Regression + Random Forest |
| Apache Kafka   | Real-time transaction streaming                     |
| Redis          | Low-latency customer state storage                  |
| PostgreSQL     | Persistent storage of transactions and predictions  |
| MLflow         | Experiment tracking and model management            |
| Grafana        | Real-time monitoring dashboard                      |
| Docker Compose | Container orchestration                             |

---

## ML Results

| Model               | Precision  | Recall     | F1         | PR-AUC     |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.0564     | 0.8737     | 0.1059     | 0.6719     |
| Random Forest       | 0.8889     | 0.7579     | 0.8182     | 0.7869     |
| **XGBoost**         | **0.9494** | **0.7895** | **0.8621** | **0.8166** |

XGBoost selected as best model based on PR-AUC score.

---

## Dataset

* **Source:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Size:** 284,807 transactions
* **Fraud cases:** 492 (0.17%)
* **Features:** V1–V28 (PCA anonymized), Amount, Time
* **Class imbalance:** 578:1

---

## Project Structure

```text
FraudStream/
 ├── data/
 │   ├── raw/              ← creditcard.csv (not committed)
 │   └── processed/        ← cleaned data + plots
 ├── notebooks/            ← EDA and model comparison
 ├── src/
 │   ├── data_inspect.py       ← dataset validation
 │   ├── eda.py                ← exploratory data analysis
 │   ├── preprocessing.py      ← cleaning + scaling + splitting
 │   ├── train_models.py       ← LR + RF + XGBoost training
 │   ├── evaluate.py           ← metrics + ROC + PR curves
 │   ├── producer.py           ← Kafka transaction generator
 │   ├── stream_consumer.py    ← real-time prediction pipeline
 │   ├── redis_state.py         ← customer behavioral state
 │   └── db.py                  ← PostgreSQL writer
 ├── models/               ← saved model files
 ├── sql/                  ← database schema
 ├── grafana/              ← dashboard JSON
 ├── docker-compose.yml    ← all services
 └── requirements.txt
```

---

## How to Run

### 1. Prerequisites

* Python 3.13
* Docker Desktop

### 2. Clone the repository

```bash
git clone git@github-personal:mansichotaliya22/FraudStream.git
cd FraudStream
```

### 3. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Start Docker services

```bash
docker compose up -d
```

### 5. Initialize database

```bash
Get-Content sql/create_tables.sql | docker exec -i fraudstream-postgres psql -U fraud_user -d fraudstream
```

### 6. Train the model

```bash
python src/preprocessing.py
python src/train_models.py
```

### 7. Start the streaming pipeline

```bash
# Terminal 1
python src/stream_consumer.py

# Terminal 2
python src/producer.py --rate 5 --mode mixed
```

### 8. View dashboards

* Grafana: http://localhost:3000 (admin/admin)
* MLflow: http://localhost:5000

---

## Live Fraud Demonstration

```bash
# Inject fraud scenarios
python src/producer.py --rate 5 --mode fraud
```

Watch Grafana dashboard show fraud rate jumping in real time.

---

## Performance

* **Model:** XGBoost (PR-AUC: 0.8166)
* **Processing latency:** ~5ms per transaction
* **Throughput:** 5–20 transactions/second
* **False positives:** 4 out of 56,746 test transactions

---

## Fraud Scenarios Simulated

| Scenario     | Description                             |
| ------------ | --------------------------------------- |
| high_amount  | Transaction 10–50x customer average     |
| new_device   | Device ID never seen before             |
| new_location | Foreign location (London, Dubai, Tokyo) |
| unusual_time | Transaction at 2–4 AM                   |
| combined     | Multiple red flags simultaneously       |

---

## Limitations

* V1–V28 features are PCA anonymized — not interpretable
* Synthetic transactions use behavioral signals, not real PCA features
* Single Kafka broker — not production scale
* Model trained on European credit card data — may not generalize globally

---

## Future Improvements

* Add Apache Spark for distributed processing
* Add FastAPI prediction endpoint
* Add model retraining pipeline
* Add alerting via Grafana alerts
* Deploy on cloud (AWS/GCP)

---

## Author

Mansi Chotaliya — [GitHub](https://github.com/mansichotaliya22)

> This is a research/portfolio prototype. Not connected to any real banking system.
