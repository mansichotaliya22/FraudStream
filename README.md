# FraudStream — Real-Time Fraud Detection Platform

> A portfolio-level end-to-end real-time fraud detection system built with **Apache Kafka, XGBoost, Redis, PostgreSQL, MLflow, Grafana, and Docker**.

FraudStream is a real-time fraud detection platform that processes financial transactions as they occur, performs ML-based fraud prediction within milliseconds, maintains customer behavioral state, persists predictions, and visualizes fraud activity through a live Grafana dashboard.

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Problem Statement](#-problem-statement)
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [Technology Stack](#-technology-stack)
* [Machine Learning](#-machine-learning)
* [Dataset](#-dataset)
* [Fraud Scenarios](#-fraud-scenarios)
* [Project Structure](#-project-structure)
* [Prerequisites](#-prerequisites)
* [Installation](#-installation)
* [Running the Project](#-running-the-project)
* [Live Fraud Demonstration](#-live-fraud-demonstration)
* [Monitoring](#-monitoring)
* [Performance](#-performance)
* [Database](#-database)
* [MLflow](#-mlflow)
* [Limitations](#-limitations)
* [Future Improvements](#-future-improvements)
* [Use Cases](#-use-cases)
* [Author](#-author)
* [Disclaimer](#-disclaimer)

---

## 🔎 Overview

**FraudStream** demonstrates how a modern real-time fraud detection pipeline can be designed using streaming, machine learning, low-latency state management, persistent storage, and monitoring technologies.

A transaction is generated and published to **Apache Kafka**. A Python consumer receives the transaction, retrieves relevant customer behavioral information from **Redis**, and passes the transaction through a trained **XGBoost** model.

The prediction and transaction details are then stored in **PostgreSQL**, while **Grafana** provides real-time visualization of transaction activity and fraud detection results.

### End-to-end flow

```text
Transaction Generator
        │
        ▼
   Apache Kafka
        │
        ▼
 Python Stream Consumer
        │
        ├──────────────► Redis
        │               Customer State
        │
        ▼
  XGBoost Model
        │
        ▼
 Fraud Prediction
        │
        ▼
   PostgreSQL
        │
        ▼
 Grafana Dashboard
```

---

## 🎯 Problem Statement

Financial fraud can cause significant financial losses and damage customer trust.

Traditional batch-based fraud detection systems may process transactions periodically, meaning suspicious transactions can be identified hours or even days after they occur.

FraudStream addresses this problem by demonstrating a **real-time fraud detection architecture** capable of:

* Receiving transactions continuously
* Processing transactions through Kafka
* Maintaining customer behavioral state
* Running ML-based fraud predictions
* Persisting transactions and predictions
* Monitoring fraud activity in real time

The primary goal is to demonstrate the architecture and engineering principles behind a real-time fraud detection system rather than connect to an actual banking infrastructure.

---

## ✨ Key Features

* ⚡ Real-time transaction streaming using Apache Kafka
* 🤖 XGBoost-based fraud classification
* 📊 Comparison of Logistic Regression, Random Forest, and XGBoost
* 🚀 Low-latency customer state retrieval using Redis
* 🗄️ Persistent transaction storage using PostgreSQL
* 📈 Real-time Grafana monitoring dashboard
* 🧪 ML experiment tracking using MLflow
* 🐳 Docker Compose-based infrastructure
* 🎭 Simulated fraud scenarios
* 📉 Precision, Recall, F1-score, ROC-AUC and PR-AUC evaluation
* 🔄 Configurable transaction generation rate
* 🧩 Modular Python project structure

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │ Transaction Generator│
                         │      producer.py     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Apache Kafka      │
                         │   Transaction Topic  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Python Consumer    │
                         │  stream_consumer.py  │
                         └──────┬─────────┬─────┘
                                │         │
                    ┌───────────┘         └───────────┐
                    ▼                                 ▼
          ┌──────────────────┐              ┌──────────────────┐
          │      Redis       │              │   XGBoost Model  │
          │ Customer State   │              │ Fraud Prediction │
          └────────┬─────────┘              └────────┬─────────┘
                   │                                 │
                   └──────────────┬──────────────────┘
                                  ▼
                         ┌──────────────────┐
                         │   PostgreSQL     │
                         │ Transactions +   │
                         │   Predictions    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     Grafana      │
                         │ Live Dashboard   │
                         └──────────────────┘
```

---

# 🛠️ Technology Stack

| Technology         | Purpose                                       |
| ------------------ | --------------------------------------------- |
| **Python**         | Core programming language                     |
| **XGBoost**        | Primary fraud classification model            |
| **Scikit-learn**   | Preprocessing and baseline ML models          |
| **Apache Kafka**   | Real-time transaction streaming               |
| **Redis**          | Low-latency customer behavioral state         |
| **PostgreSQL**     | Persistent transaction and prediction storage |
| **MLflow**         | Experiment tracking and model management      |
| **Grafana**        | Real-time monitoring and visualization        |
| **Docker Compose** | Infrastructure and service orchestration      |

---

# 🤖 Machine Learning

FraudStream uses supervised machine learning to classify transactions as legitimate or fraudulent.

Three models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

Because fraud detection datasets are highly imbalanced, **PR-AUC** is used as an important model-selection metric alongside Precision, Recall, and F1-score.

## Model Comparison

| Model               |  Precision |     Recall |         F1 |     PR-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     0.0564 |     0.8737 |     0.1059 |     0.6719 |
| Random Forest       |     0.8889 |     0.7579 |     0.8182 |     0.7869 |
| **XGBoost**         | **0.9494** | **0.7895** | **0.8621** | **0.8166** |

### Selected Model

**XGBoost** was selected as the final model because it achieved the highest PR-AUC:

```text
XGBoost PR-AUC = 0.8166
```

It also achieved:

```text
Precision = 0.9494
Recall    = 0.7895
F1-score  = 0.8621
```

---

# 📊 Dataset

FraudStream uses the publicly available **Credit Card Fraud Detection** dataset from Kaggle.

**Dataset:** Credit Card Fraud Detection

* Transactions: **284,807**
* Fraudulent transactions: **492**
* Fraud rate: approximately **0.17%**
* Features: **V1–V28, Time, Amount**
* V1–V28: PCA-anonymized features
* Severe class imbalance: approximately **578:1**

Dataset source:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> The dataset is used only for research and portfolio demonstration purposes.

---

# 🚨 Fraud Scenarios

FraudStream includes simulated behavioral fraud scenarios to demonstrate how suspicious activity can be detected in a streaming environment.

| Scenario       | Description                                                                        |
| -------------- | ---------------------------------------------------------------------------------- |
| `high_amount`  | Transaction is significantly higher than the customer's typical transaction amount |
| `new_device`   | Transaction originates from a device that has not previously been observed         |
| `new_location` | Transaction occurs from an unusual or foreign location                             |
| `unusual_time` | Transaction occurs during an unusual period, such as 2–4 AM                        |
| `combined`     | Multiple suspicious behavioral signals occur simultaneously                        |

### Example

A transaction may be flagged because:

```text
High transaction amount
        +
New device
        +
Unusual location
        +
Unusual transaction time
        ↓
Potential Fraud
```

---

# 📁 Project Structure

```text
FraudStream/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   │
│   └── processed/
│       └── cleaned data + plots
│
├── notebooks/
│   └── EDA and model comparison
│
├── src/
│   ├── data_inspect.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── producer.py
│   ├── stream_consumer.py
│   ├── redis_state.py
│   └── db.py
│
├── models/
│   └── saved model files
│
├── sql/
│   └── create_tables.sql
│
├── grafana/
│   └── dashboard JSON
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# ⚙️ Prerequisites

Before running FraudStream, install:

* **Python 3.13**
* **Docker Desktop**
* **Git**
* Windows PowerShell or a compatible terminal

Verify Python:

```powershell
python --version
```

Verify Docker:

```powershell
docker --version
docker compose version
```

---

# 🚀 Installation

## 1. Clone the Repository

```powershell
git clone git@github-personal:mansichotaliya22/FraudStream.git
cd FraudStream
```

If you are using HTTPS instead:

```powershell
git clone https://github.com/mansichotaliya22/FraudStream.git
cd FraudStream
```

---

## 2. Create a Virtual Environment

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

You should see something similar to:

```text
(venv) PS C:\...\FraudStream>
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# 🐳 Start Infrastructure

Start all Docker services:

```powershell
docker compose up -d
```

Check running containers:

```powershell
docker compose ps
```

You should see the required infrastructure services running, including Kafka, Redis, PostgreSQL, Grafana, and MLflow where configured in `docker-compose.yml`.

To view service logs:

```powershell
docker compose logs -f
```

To stop the services:

```powershell
docker compose down
```

---

# 🗄️ Initialize PostgreSQL

Run the database schema:

```powershell
Get-Content sql/create_tables.sql | docker exec -i fraudstream-postgres psql -U fraud_user -d fraudstream
```

This creates the required tables for storing transactions and fraud predictions.

---

# 🧠 Train the ML Models

Run preprocessing:

```powershell
python src/preprocessing.py
```

Then train the models:

```powershell
python src/train_models.py
```

The training pipeline evaluates:

* Logistic Regression
* Random Forest
* XGBoost

The best-performing model is selected based on the evaluation results.

---

# 📡 Start the Streaming Pipeline

FraudStream uses two processes:

### Terminal 1 — Start Consumer

```powershell
python src/stream_consumer.py
```

The consumer:

1. Connects to Kafka
2. Reads transactions
3. Retrieves customer state from Redis
4. Performs fraud prediction
5. Updates customer state
6. Stores transaction and prediction information in PostgreSQL

---

### Terminal 2 — Start Producer

```powershell
python src/producer.py --rate 5 --mode mixed
```

The producer generates transactions and publishes them to Kafka.

The `--rate` argument controls the approximate number of transactions generated per second.

Example:

```powershell
python src/producer.py --rate 10 --mode mixed
```

---

# 🚨 Live Fraud Demonstration

FraudStream includes a dedicated fraud simulation mode.

Run:

```powershell
python src/producer.py --rate 5 --mode fraud
```

This generates transactions containing simulated suspicious behaviors.

Open the Grafana dashboard while the producer is running.

You should be able to observe changes in:

* Total transactions
* Fraudulent transactions
* Fraud rate
* Transaction volume
* Prediction results
* Transaction activity over time

---

# 📈 Monitoring with Grafana

Grafana provides a real-time visualization layer for the FraudStream pipeline.

Open:

```text
http://localhost:3000
```

Default credentials:

```text
Username: admin
Password: admin
```

> Change default credentials before using the project in any publicly accessible or production environment.

The dashboard can be used to monitor:

* Transaction volume
* Fraud count
* Fraud percentage
* Legitimate vs fraudulent transactions
* Recent predictions
* Transaction trends

---

# 🧪 MLflow

MLflow is used for experiment tracking and model management.

Open:

```text
http://localhost:5000
```

Depending on the configured MLflow setup, experiments can contain:

* Model parameters
* Evaluation metrics
* Model versions
* Training runs
* Experiment comparisons

This makes it easier to track changes between different ML experiments.

---

# ⚡ Performance

The current portfolio prototype achieves approximately:

| Metric                   |                    Result |
| ------------------------ | ------------------------: |
| Model                    |                   XGBoost |
| PR-AUC                   |                **0.8166** |
| Processing latency       |   **~5 ms / transaction** |
| Throughput               | **5–20 transactions/sec** |
| Test transactions        |                **56,746** |
| Reported false positives |                     **4** |

> Performance values are environment-dependent and should be treated as measurements from the development/portfolio environment rather than production guarantees.

---

# 🗃️ Data Flow

The complete transaction lifecycle is:

```text
1. Transaction generated
          ↓
2. Transaction published to Kafka
          ↓
3. Consumer receives transaction
          ↓
4. Customer state retrieved from Redis
          ↓
5. Transaction passed to ML model
          ↓
6. Fraud probability/prediction generated
          ↓
7. Customer behavioral state updated
          ↓
8. Transaction + prediction stored in PostgreSQL
          ↓
9. Grafana reads stored data
          ↓
10. Real-time fraud metrics displayed
```

---

# 🔴 Redis Customer State

Redis provides fast access to customer behavioral information.

Example state information may include:

```text
Customer ID
Average transaction amount
Transaction count
Known devices
Known locations
Recent transaction timestamps
```

This allows the system to compare incoming transactions against recent customer behavior without repeatedly querying the primary database.

For example:

```text
Customer average = ₹2,000
Incoming transaction = ₹35,000

Transaction amount is significantly above normal
                ↓
Potential risk signal
```

---

# 🐘 PostgreSQL

PostgreSQL acts as the persistent storage layer.

The database stores information such as:

* Transaction details
* Customer information
* Fraud predictions
* Prediction scores
* Transaction timestamps
* Fraud indicators

Unlike Redis, PostgreSQL is intended for durable storage and historical analysis.

---

# 📨 Apache Kafka

Kafka provides the event-streaming layer.

The basic flow is:

```text
Producer
   ↓
Kafka Topic
   ↓
Consumer
```

Kafka decouples transaction generation from transaction processing.

This makes the architecture easier to extend later with additional consumers, analytics services, alerting systems, or downstream processing components.

---

# 🔬 Model Evaluation

Because the dataset is extremely imbalanced, accuracy alone is not sufficient for evaluating the fraud detection model.

FraudStream focuses on:

### Precision

Measures how many transactions predicted as fraud were actually fraudulent.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual fraudulent transactions were detected.

```text
Recall = TP / (TP + FN)
```

### F1 Score

Balances precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### PR-AUC

The **Precision-Recall Area Under the Curve** is particularly useful for highly imbalanced classification problems such as fraud detection.

---

# ⚠️ Limitations

FraudStream is a research and portfolio prototype and has several limitations.

### 1. Anonymized Dataset

The V1–V28 features are PCA-anonymized, making them difficult to interpret from a business perspective.

### 2. Synthetic Streaming Data

The real-time transaction generator creates simulated transactions. The generated behavioral signals are not equivalent to real-world banking transaction features.

### 3. Single Kafka Broker

The current architecture uses a single Kafka broker and is not designed for production-scale fault tolerance.

### 4. Dataset Generalization

The model is trained on European credit card transaction data and may not generalize to different countries, financial institutions, customer populations, or transaction environments.

### 5. No Production Alerting

The current system focuses primarily on detection and visualization rather than operational alert delivery.

### 6. No Automated Retraining

Model retraining is currently a manual process.

---

# 🔮 Future Improvements

Several improvements can extend FraudStream toward a more production-oriented architecture.

## Distributed Stream Processing

Integrate **Apache Spark Structured Streaming** for larger-scale stream processing.

```text
Kafka
  ↓
Spark Structured Streaming
  ↓
ML Inference
  ↓
PostgreSQL / Data Lake
```

## FastAPI Prediction Service

Expose the trained model through a REST API:

```text
Client
  ↓
FastAPI
  ↓
XGBoost
  ↓
Prediction
```

## Automated Model Retraining

Create an automated pipeline that:

1. Collects new transaction data
2. Validates the data
3. Retrains models
4. Evaluates model performance
5. Registers the best model
6. Deploys the updated model

## Grafana Alerting

Configure alerts when:

* Fraud rate exceeds a threshold
* Fraud volume suddenly increases
* High-risk transactions are detected
* Kafka processing latency increases
* Consumer failures occur

## Cloud Deployment

The platform could eventually be deployed using cloud infrastructure such as:

* AWS
* Google Cloud
* Microsoft Azure

---

# 💼 Potential Use Cases

The architecture demonstrated by FraudStream can be adapted for:

* Credit card fraud detection
* Digital payment monitoring
* Banking transaction monitoring
* E-commerce fraud detection
* Account takeover detection
* Insurance claim fraud detection
* Real-time risk scoring
* Suspicious activity monitoring

---

# 🔐 Security Considerations

For a production implementation, additional security controls would be required, including:

* Kafka authentication and encryption
* TLS for service communication
* PostgreSQL access control
* Redis authentication
* Secrets management
* API authentication
* Role-based Grafana access
* Audit logging
* Network isolation
* Secure Docker configuration

**Never commit real credentials, API keys, passwords, or production database connection strings to GitHub.**

Use environment variables or a secrets-management system instead.

---

# 🧹 Git & Data Hygiene

The raw dataset should not be committed to the repository.

Recommended `.gitignore` entries:

```gitignore
# Virtual environment
venv/
.venv/

# Python
__pycache__/
*.py[cod]

# Data
data/raw/
*.csv

# Environment variables
.env

# ML artifacts
mlruns/
*.pkl
*.joblib

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

Adjust the ML artifact rules if trained models are intentionally versioned in the repository.

---

# 🧪 Example Workflow

A typical local development session looks like this:

```powershell
# Activate environment
venv\Scripts\activate

# Start infrastructure
docker compose up -d

# Initialize database
Get-Content sql/create_tables.sql | docker exec -i fraudstream-postgres psql -U fraud_user -d fraudstream

# Train models
python src/preprocessing.py
python src/train_models.py

# Terminal 1
python src/stream_consumer.py

# Terminal 2
python src/producer.py --rate 5 --mode mixed

# Fraud demonstration
python src/producer.py --rate 5 --mode fraud
```

Then open:

```text
Grafana → http://localhost:3000
MLflow  → http://localhost:5000
```

---

# 📌 Project Highlights

FraudStream demonstrates several important concepts in modern data engineering and machine learning:

```text
Machine Learning
       +
Real-Time Streaming
       +
Feature/State Management
       +
Persistent Data Storage
       +
Experiment Tracking
       +
Observability
       =
End-to-End Fraud Detection Platform
```

The project combines **ML engineering, data engineering, backend processing, distributed systems concepts, and monitoring** into a single portfolio project.

---

# 👩‍💻 Author

**Mansi Chotaliya**

Computer Engineering Student
Aspiring Data & Business Analyst

GitHub:
https://github.com/mansichotaliya22

---

# 📄 License

This project is intended for educational, research, and portfolio purposes.

If you add a specific open-source license to the repository, replace this section with the corresponding license information.

---

# ⚠️ Disclaimer

FraudStream is a **research/portfolio prototype**.

It is **not connected to any real banking, financial, payment, or customer system** and should not be used for real financial decision-making without substantial additional validation, security, compliance, monitoring, and production engineering.

The fraud scenarios and streaming transactions are simulated for demonstration purposes.
