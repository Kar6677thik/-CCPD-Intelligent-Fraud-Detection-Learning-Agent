# FraudShield AI - Intelligent Fraud Detection Learning Agent

A production-grade credit card fraud detection system featuring a hybrid ensemble of three machine learning models, a modern React dashboard, real-time SHAP explainability, and an adaptive learning loop that improves detection accuracy over time through human feedback.

Built as a project for the AI Specialist course (6th Semester).

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [ML Models](#ml-models)
5. [Usage Guide](#usage-guide)
6. [Adaptive Learning](#adaptive-learning)
7. [API Reference](#api-reference)
8. [Features](#features)

---

## Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- The `creditcard.csv` dataset (Kaggle Credit Card Fraud Detection) placed in the project root

---

## Quick Start

### Option 1: Local Development (Recommended)

**Backend** (Terminal 1):

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend** (Terminal 2):

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

### Option 2: Docker

```bash
docker-compose up --build
```

---

## Architecture

```
CreditCard3/
├── backend/
│   ├── main.py             # FastAPI REST and WebSocket endpoints
│   ├── ml_pipeline.py      # Isolation Forest, Autoencoder, XGBoost, Ensemble
│   ├── database.py         # SQLite CRUD for predictions, feedback, model versions
│   └── output_manager.py   # Timestamped output folders with plots and metrics
├── frontend/
│   └── src/
│       ├── App.tsx          # Dashboard page components
│       ├── api.ts           # Typed API client layer
│       └── index.css        # Design system (dark fintech theme)
├── models/                  # Saved model files (.pkl, .keras)
├── output/                  # Auto-generated: YYYY-MM-DD_HH-MM-SS/
│   └── <timestamp>/
│       ├── confusion_matrix_*.png
│       ├── roc_curves.png
│       ├── precision_recall_curves.png
│       ├── feature_importance.png
│       ├── score_distributions.png
│       └── metrics.json
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## ML Models

| Model              | Type            | Strength                                            |
|---------------------|-----------------|-----------------------------------------------------|
| Isolation Forest    | Unsupervised    | Detects novel anomalies without requiring labels     |
| Autoencoder         | Deep Learning   | Learns normal transaction patterns via reconstruction error |
| XGBoost             | Supervised      | Highest precision with labeled data and SMOTE balancing |
| Ensemble            | Weighted Voting | Combines all three models for robust predictions     |

The ensemble uses weighted voting (Isolation Forest: 25%, Autoencoder: 25%, XGBoost: 50%) and applies per-model thresholds optimized by maximizing the F1 score on the precision-recall curve.

---

## Usage Guide

### Step 1: Train Models

Navigate to the **Dashboard** page and click **Train Models**. This trains all three models on the `creditcard.csv` dataset. Training progress is displayed in real time via WebSocket.

### Step 2: Upload Data for Prediction

Go to the **Detection** page and upload a CSV file containing transaction data. The system will return fraud predictions with ensemble scores, individual model scores, and SHAP-based explanations for each transaction.

### Step 3: Review Transactions

Open the **Transaction Explorer** to browse predicted transactions. Each row displays the transaction ID, ensemble fraud score, predicted status, and a feedback column.

### Step 4: Submit Feedback

For each transaction in the Explorer, click the **Fraud** or **Legit** button in the Feedback column to label it with the correct classification. A badge will confirm your label. You can label as many transactions as needed; the more feedback you provide, the better the retraining results.

### Step 5: Retrain with Feedback (Adaptive Learning)

Go to the **Adaptive Learning** page and click **Retrain with Feedback**. The system will use all labeled transactions to perform incremental retraining of the XGBoost model. After retraining, the page displays a learning progress chart and a history of all retraining cycles with F1 score improvements.

---

## Adaptive Learning

The adaptive learning loop allows the system to improve over time based on human expertise:

```
Upload CSV           Review              Label              Retrain
(Detection) ──────>  Transactions ─────> Transactions ────> with Feedback
                     (Explorer)          (Fraud/Legit)      (Learning Page)
                                                                  │
                                                                  v
                                                           XGBoost does
                                                           incremental
                                                           retraining
                                                                  │
                                                                  v
                                                           Metrics compared
                                                           (before vs after)
```

**How it works internally:**

1. When you click Fraud or Legit on a transaction, the frontend calls `POST /api/feedback` which updates the `feedback_label` column in the predictions table.
2. When you click Retrain with Feedback, the backend fetches all predictions that have a non-null `feedback_label`, extracts their feature vectors, and passes them to `XGBoostModel.incremental_train()`.
3. XGBoost performs incremental training by continuing from its existing booster with the new feedback samples.
4. The system records F1 score before and after retraining in the training history table, allowing you to track model improvement over successive retraining cycles.

---

## API Reference

| Method | Endpoint                     | Description                          |
|--------|------------------------------|--------------------------------------|
| POST   | `/api/train`                 | Train all three models               |
| POST   | `/api/predict`               | Upload CSV and get predictions       |
| GET    | `/api/model-stats`           | Current model performance metrics    |
| POST   | `/api/feedback`              | Submit fraud/legit labels            |
| POST   | `/api/retrain`               | Incremental learning with feedback   |
| GET    | `/api/feature-importance`    | SHAP feature importance values       |
| GET    | `/api/transactions`          | Paginated transaction history        |
| POST   | `/api/generate-samples`      | Generate synthetic demo transactions |
| GET    | `/api/alerts`                | Fraud alerts                         |
| GET    | `/api/export/csv`            | Download predictions as CSV          |
| WS     | `/ws/training`               | Real-time training progress          |

---

## Features

- **Hybrid Ensemble Detection**: Three complementary models combined via weighted voting for robust fraud identification.
- **SHAP Explainability**: Human-readable explanations for why each transaction was flagged, powered by SHAP TreeExplainer.
- **Adaptive Learning Loop**: Submit feedback on transactions and retrain the model to improve accuracy over time.
- **Drift Detection**: KS-test based monitoring that alerts when incoming data distribution shifts from training data.
- **Active Learning**: Identifies the most uncertain predictions for priority manual review.
- **Dark Fintech Dashboard**: Navy and teal color scheme with glassmorphic cards and smooth animations.
- **Professor Mode**: Toggle to hide technical details and show only business-level summaries.
- **Auto-generated Output**: Each prediction run creates a timestamped folder containing confusion matrices, ROC curves, precision-recall curves, feature importance charts, score distributions, and a metrics.json file.
- **Real-time Training Progress**: WebSocket connection streams training progress updates to the dashboard.
- **CSV Export**: Download all prediction results as a CSV file.
