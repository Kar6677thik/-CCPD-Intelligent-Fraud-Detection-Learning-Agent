# 🛡️ FraudShield AI — Intelligent Fraud Detection Learning Agent

A production-grade fraud detection system with a modern React dashboard and FastAPI backend, featuring real-time fraud detection, model comparison, SHAP explainability, and adaptive learning.

## 🚀 Quick Start

### Option 1: Local Development (Recommended)

**Backend:**
```bash
# Activate venv and install dependencies
venv\Scripts\activate
pip install -r requirements.txt

# Start FastAPI server
venv\Scripts\uvicorn.exe backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend** (in a new terminal):
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

## 📋 First-Time Setup

1. **Train Models**: Click "Train Models" on the Dashboard or call `POST /api/train`
2. **Upload Data**: Go to Detection → upload `creditcard.csv` for predictions
3. **Explore Results**: View transactions, model comparison, and SHAP explanations
4. **Adaptive Learning**: Submit feedback → click "Retrain with Feedback"

---

## 🏗️ Architecture

```
CreditCard3/
├── backend/                # FastAPI + ML Pipeline
│   ├── main.py             # REST API + WebSocket endpoints
│   ├── ml_pipeline.py      # IF, Autoencoder, XGBoost, Ensemble
│   ├── database.py         # SQLite CRUD operations
│   └── output_manager.py   # Timestamped output folders with plots
├── frontend/               # React + TypeScript + Vite
│   └── src/
│       ├── App.tsx          # All dashboard pages
│       ├── api.ts           # API service layer
│       └── index.css        # Design system
├── output/                  # Auto-generated: YYYY-MM-DD_HH-MM-SS/
│   └── <timestamp>/
│       ├── confusion_matrix_*.png
│       ├── roc_curves.png
│       ├── precision_recall_curves.png
│       ├── feature_importance.png
│       ├── score_distributions.png
│       └── metrics.json
├── models/                  # Saved model files
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 🎯 ML Models

| Model | Type | Strength |
|-------|------|----------|
| **Isolation Forest** | Unsupervised | Catches novel anomalies without labels |
| **Autoencoder** | Deep Learning | Learns normal patterns via reconstruction error |
| **XGBoost** | Supervised | Best precision with labeled data + SMOTE |
| **Ensemble** | Weighted Voting | Combines all three for robust predictions |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/train` | Train all three models |
| POST | `/api/predict` | Upload CSV → get predictions |
| GET | `/api/model-stats` | Current model metrics |
| POST | `/api/retrain` | Incremental learning with feedback |
| GET | `/api/feature-importance` | SHAP feature importance |
| GET | `/api/transactions` | Paginated transaction history |
| POST | `/api/feedback` | Submit fraud/legit labels |
| POST | `/api/generate-samples` | Generate demo transactions |
| GET | `/api/alerts` | Fraud alerts |
| GET | `/api/export/csv` | Download predictions CSV |
| WS | `/ws/training` | Real-time training progress |

---

## 🎬 5-Minute Demo Script

1. **[0:00]** Open dashboard → Show clean fintech UI, explain model status
2. **[1:00]** Click "Train Models" → Show WebSocket training progress
3. **[2:00]** Go to Detection → Upload `creditcard.csv` → Show predictions with fraud scores
4. **[2:30]** Explore SHAP explanations → "V14 increases fraud risk"
5. **[3:00]** Go to Model Comparison → Show radar chart, per-model metrics
6. **[3:30]** Go to Transaction Explorer → Filter fraud-only, search by ID
7. **[4:00]** Toggle Professor Mode → Show simplified business view
8. **[4:30]** Show `output/` folder → Open metrics.json, confusion matrices, ROC curves
9. **[5:00]** Explain adaptive learning → Feedback loop → Retrain → Compare improvement

---

## 🎨 Features

- **Dark Fintech Theme**: Navy/teal color scheme with glassmorphic cards
- **Professor Mode**: Hide technical details, show business value
- **SHAP Explainability**: Human-readable reasons why transactions are flagged
- **Drift Detection**: KS-test alerts when data distribution changes
- **Active Learning**: Identifies uncertain transactions for manual review
- **Auto-generated Output**: Each prediction creates timestamped folder with plots + metrics.json
- **Export**: Download predictions as CSV
