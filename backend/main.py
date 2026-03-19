"""
FastAPI main application for the Fraud Detection Learning Agent.
"""
import os
import io
import csv
import json
import uuid
import asyncio
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from backend.database import (
    init_db, save_predictions, get_predictions, save_feedback,
    get_feedback_data, save_model_version, get_model_versions,
    save_training_history, get_training_history, save_alert,
    get_alerts, acknowledge_alert, save_dataset_info, get_datasets,
    get_dashboard_stats
)
from backend.ml_pipeline import FraudDetectionPipeline
from backend.output_manager import generate_all_outputs

# ─── App Setup ────────────────────────────────────────────────
app = FastAPI(title="Fraud Detection Learning Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────
pipeline = FraudDetectionPipeline()
training_in_progress = False
connected_websockets: List[WebSocket] = []
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── Pydantic Models ─────────────────────────────────────────
class FeedbackRequest(BaseModel):
    transaction_ids: List[str]
    labels: List[int]

class ThresholdConfig(BaseModel):
    isolation_forest_weight: float = 0.25
    autoencoder_weight: float = 0.25
    xgboost_weight: float = 0.50
    alert_threshold: float = 0.7

class SampleGeneratorRequest(BaseModel):
    num_transactions: int = 100
    fraud_ratio: float = 0.05

# ─── Events ──────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_db()
    # Try to load pre-trained models
    if pipeline.load_models():
        print("✅ Pre-trained models loaded successfully")
    else:
        print("⚠️  No pre-trained models found. Please train models first.")

# ─── WebSocket ───────────────────────────────────────────────
@app.websocket("/ws/training")
async def training_ws(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)

async def broadcast_progress(model: str, message: str, progress: float):
    data = json.dumps({
        "model": model,
        "message": message,
        "progress": progress,
        "timestamp": datetime.now().isoformat()
    })
    dead = []
    for ws in connected_websockets:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_websockets.remove(ws)

# ─── Dashboard ───────────────────────────────────────────────
@app.get("/api/dashboard")
async def dashboard():
    stats = get_dashboard_stats()
    stats["models_trained"] = pipeline.is_trained
    stats["training_in_progress"] = training_in_progress
    return stats

# ─── Model Stats ─────────────────────────────────────────────
@app.get("/api/model-stats")
async def model_stats():
    if not pipeline.is_trained:
        return {"error": "Models not trained yet", "models_trained": False}
    metrics = pipeline.get_all_metrics()
    metrics["models_trained"] = True
    return metrics

# ─── Training ────────────────────────────────────────────────
@app.post("/api/train")
async def train_models(dataset_path: Optional[str] = None):
    global training_in_progress

    if training_in_progress:
        raise HTTPException(400, "Training already in progress")

    training_in_progress = True

    try:
        csv_path = dataset_path or os.path.join(BASE_DIR, "creditcard.csv")
        if not os.path.exists(csv_path):
            raise HTTPException(404, f"Dataset not found: {csv_path}")

        loop = asyncio.get_event_loop()

        def progress_callback(model, message, pct):
            asyncio.run_coroutine_threadsafe(
                broadcast_progress(model, message, pct), loop
            )

        X, y, feature_names = pipeline.load_and_prepare_data(csv_path)

        await broadcast_progress("system", "Starting training pipeline...", 0.0)

        # Run training in thread pool to not block event loop
        results = await loop.run_in_executor(
            None, lambda: pipeline.train_all(X, y, progress_callback)
        )

        # Generate output plots and metrics
        predictions = pipeline.predict(X)
        importance = pipeline.get_feature_importance()
        output_folder, files, metrics = generate_all_outputs(
            y, predictions, importance,
            {"dataset": csv_path, "training_samples": len(X)}
        )

        # Save model versions to DB
        for model_name, model_metrics in results.items():
            version = getattr(getattr(pipeline, model_name), "version", 1)
            save_model_version(model_name, version, model_metrics, len(X))

        # Save dataset info
        save_dataset_info(
            os.path.basename(csv_path), csv_path, len(X),
            X.shape[1], float(np.mean(y))
        )

        await broadcast_progress("system", "Training complete! ✅", 1.0)

        return {
            "status": "success",
            "results": results,
            "output_folder": output_folder,
            "generated_files": [os.path.basename(f) for f in files]
        }

    except Exception as e:
        await broadcast_progress("system", f"Training failed: {str(e)}", -1)
        raise HTTPException(500, str(e))
    finally:
        training_in_progress = False

# ─── Prediction ──────────────────────────────────────────────
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not pipeline.is_trained:
        raise HTTPException(400, "Models not trained. Please train first via POST /api/train")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Extract features and optional labels
        y_true = None
        if "Class" in df.columns:
            y_true = df["Class"].values
            drop_cols = ["Class"]
            if "Time" in df.columns:
                drop_cols.append("Time")
            X = df.drop(columns=drop_cols).values
            feature_names = [c for c in df.columns if c not in drop_cols]
        else:
            X = df.values
            feature_names = list(df.columns)

        # Predict
        results = pipeline.predict(X)
        batch_id = str(uuid.uuid4())[:8]

        # Get SHAP explanations for first 100
        shap_results = pipeline.get_shap_explanations(X, max_samples=min(100, len(X)))

        # Build transaction-level results
        transactions = []
        for i in range(len(X)):
            tid = f"TXN-{batch_id}-{i:06d}"
            explanation = {}
            if "per_transaction" in shap_results and i < len(shap_results["per_transaction"]):
                explanation = shap_results["per_transaction"][i]

            txn = {
                "transaction_id": tid,
                "features": {feature_names[j]: float(X[i, j]) for j in range(len(feature_names))},
                "isolation_forest_score": float(results["isolation_forest_scores"][i]),
                "autoencoder_score": float(results["autoencoder_scores"][i]),
                "xgboost_score": float(results["xgboost_scores"][i]),
                "ensemble_score": float(results["ensemble_scores"][i]),
                "is_fraud": int(results["predictions"][i]),
                "confidence": float(max(
                    results["ensemble_scores"][i],
                    1 - results["ensemble_scores"][i]
                )),
                "explanation": explanation,
            }
            transactions.append(txn)

            # Generate alerts for high-risk transactions
            if results["ensemble_scores"][i] >= 0.7:
                save_alert(tid, float(results["ensemble_scores"][i]), "high_risk")

        # Save predictions to DB
        save_predictions(transactions, batch_id)

        # Generate output folder with plots if we have labels
        output_folder = None
        generated_files = []
        metrics = {}
        if y_true is not None:
            importance = pipeline.get_feature_importance()
            output_folder, generated_files, metrics = generate_all_outputs(
                y_true, results, importance,
                {"batch_id": batch_id, "dataset": file.filename}
            )

        # Drift detection
        drift = pipeline.drift_detector.detect_drift(X)

        # Active learning — uncertain samples
        uncertain_indices = pipeline.active_learner.get_uncertain_samples(
            results["ensemble_scores"], n_samples=min(20, len(X))
        )

        summary = {
            "batch_id": batch_id,
            "total_transactions": len(X),
            "flagged_fraud": int(np.sum(results["predictions"])),
            "avg_ensemble_score": float(np.mean(results["ensemble_scores"])),
            "max_ensemble_score": float(np.max(results["ensemble_scores"])),
            "model_scores": {
                "isolation_forest": {
                    "flagged": int(np.sum(results["isolation_forest_preds"])),
                    "avg_score": float(np.mean(results["isolation_forest_scores"]))
                },
                "autoencoder": {
                    "flagged": int(np.sum(results["autoencoder_preds"])),
                    "avg_score": float(np.mean(results["autoencoder_scores"]))
                },
                "xgboost": {
                    "flagged": int(np.sum(results["xgboost_preds"])),
                    "avg_score": float(np.mean(results["xgboost_scores"]))
                }
            },
            "drift_detected": drift.get("overall_drift", False),
            "drift_score": drift.get("drift_score", 0),
            "uncertain_transaction_count": len(uncertain_indices),
            "output_folder": output_folder,
            "metrics": metrics
        }

        return {
            "summary": summary,
            "transactions": transactions[:200],  # Limit response size
            "shap_global": shap_results.get("global_importance", {}),
            "drift": drift,
            "uncertain_indices": uncertain_indices.tolist()
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Prediction failed: {str(e)}")

# ─── Upload Dataset ──────────────────────────────────────────
@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(content)

        df = pd.read_csv(filepath)
        fraud_col = "Class" if "Class" in df.columns else df.columns[-1]
        fraud_ratio = float(df[fraud_col].mean()) if fraud_col in df.columns else 0

        save_dataset_info(
            file.filename, filepath, len(df),
            len(df.columns), fraud_ratio
        )

        return {
            "status": "success",
            "filename": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "fraud_ratio": fraud_ratio,
            "message": f"Dataset uploaded ({len(df):,} rows). Use POST /api/train to retrain."
        }
    except Exception as e:
        raise HTTPException(500, str(e))

# ─── Transactions ────────────────────────────────────────────
@app.get("/api/transactions")
async def transactions(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    fraud_only: bool = False,
    min_score: float = 0.0,
    search: str = ""
):
    return get_predictions(page, per_page, fraud_only, min_score, search)

# ─── Feature Importance ─────────────────────────────────────
@app.get("/api/feature-importance")
async def feature_importance():
    if not pipeline.is_trained:
        raise HTTPException(400, "Models not trained")
    return pipeline.get_feature_importance()

# ─── Feedback & Retrain ─────────────────────────────────────
@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    save_feedback(req.transaction_ids, req.labels)
    return {"status": "success", "count": len(req.transaction_ids)}

@app.post("/api/retrain")
async def retrain_with_feedback():
    global training_in_progress
    if training_in_progress:
        raise HTTPException(400, "Training already in progress")

    feedback_data = get_feedback_data()
    if not feedback_data:
        raise HTTPException(400, "No feedback data available for retraining")

    training_in_progress = True
    try:
        loop = asyncio.get_event_loop()

        # Get current metrics before retraining
        metrics_before = pipeline.get_all_metrics()

        # Prepare feedback data
        features_list = []
        labels = []
        for item in feedback_data:
            if item.get("features") and isinstance(item["features"], dict):
                features_list.append(list(item["features"].values()))
                labels.append(item["feedback_label"])

        if not features_list:
            raise HTTPException(400, "No valid feedback data")

        X_feedback = np.array(features_list)
        y_feedback = np.array(labels)

        def progress_callback(model, message, pct):
            asyncio.run_coroutine_threadsafe(
                broadcast_progress(model, message, pct), loop
            )

        # Incremental training for XGBoost
        await broadcast_progress("xgboost", "Starting incremental retraining...", 0.1)
        xgb_metrics = await loop.run_in_executor(
            None, lambda: pipeline.xgboost.incremental_train(X_feedback, y_feedback)
        )
        pipeline.xgboost.save()

        metrics_after = pipeline.get_all_metrics()

        # Save training history
        save_training_history("xgboost", metrics_before.get("xgboost", {}),
                              xgb_metrics, len(feedback_data))

        await broadcast_progress("system", "Retraining complete! ✅", 1.0)

        return {
            "status": "success",
            "feedback_samples": len(feedback_data),
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
        }
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        training_in_progress = False

# ─── Training History ────────────────────────────────────────
@app.get("/api/training-history")
async def training_history():
    return get_training_history()

# ─── Model Versions ─────────────────────────────────────────
@app.get("/api/model-versions")
async def model_versions(model_name: Optional[str] = None):
    return get_model_versions(model_name)

# ─── Drift Detection ────────────────────────────────────────
@app.get("/api/drift-status")
async def drift_status():
    return {
        "reference_set": pipeline.drift_detector.reference_stats is not None,
        "message": "Upload data via /api/predict to check for drift"
    }

# ─── Active Learning ────────────────────────────────────────
@app.get("/api/active-learning")
async def active_learning_samples():
    preds = get_predictions(page=1, per_page=200)
    items = preds.get("items", [])
    if not items:
        return {"samples": [], "message": "No predictions available"}

    # Find most uncertain
    uncertain = sorted(items, key=lambda x: abs(x.get("ensemble_score", 0) - 0.5))[:20]
    return {"samples": uncertain}

# ─── Alerts ──────────────────────────────────────────────────
@app.get("/api/alerts")
async def get_all_alerts(acknowledged: Optional[bool] = None):
    return get_alerts(acknowledged)

@app.post("/api/alerts/{alert_id}/acknowledge")
async def ack_alert(alert_id: int):
    acknowledge_alert(alert_id)
    return {"status": "acknowledged"}

# ─── Datasets ───────────────────────────────────────────────
@app.get("/api/datasets")
async def list_datasets():
    return get_datasets()

# ─── Sample Generator ───────────────────────────────────────
@app.post("/api/generate-samples")
async def generate_samples(req: SampleGeneratorRequest):
    """Generate synthetic transactions for demos."""
    np.random.seed(int(datetime.now().timestamp()))
    n = req.num_transactions
    n_fraud = max(1, int(n * req.fraud_ratio))
    n_legit = n - n_fraud

    feature_names = pipeline.feature_names if pipeline.feature_names else [f"V{i}" for i in range(1, 29)] + ["Amount"]
    n_features = len(feature_names)

    # Generate normal transactions
    legit_data = np.random.randn(n_legit, n_features) * 0.5
    if "Amount" in feature_names:
        amt_idx = feature_names.index("Amount")
        legit_data[:, amt_idx] = np.abs(np.random.exponential(50, n_legit))

    # Generate fraudulent transactions (more extreme values)
    fraud_data = np.random.randn(n_fraud, n_features) * 2.5
    if "Amount" in feature_names:
        fraud_data[:, amt_idx] = np.abs(np.random.exponential(500, n_fraud))

    X = np.vstack([legit_data, fraud_data])
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    transactions = []
    for i in range(len(X)):
        txn = {"Class": int(y[i])}
        for j, fname in enumerate(feature_names):
            txn[fname] = round(float(X[i, j]), 6)
        transactions.append(txn)

    return {
        "transactions": transactions,
        "total": len(transactions),
        "fraud_count": int(np.sum(y)),
        "feature_names": feature_names
    }

# ─── Export ──────────────────────────────────────────────────
@app.get("/api/export/csv")
async def export_csv():
    preds = get_predictions(page=1, per_page=10000)
    items = preds.get("items", [])
    if not items:
        raise HTTPException(404, "No predictions to export")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Transaction ID", "Timestamp", "Ensemble Score",
        "Isolation Forest", "Autoencoder", "XGBoost",
        "Is Fraud", "Confidence", "Feedback"
    ])
    for item in items:
        writer.writerow([
            item.get("transaction_id", ""),
            item.get("timestamp", ""),
            item.get("ensemble_score", ""),
            item.get("isolation_forest_score", ""),
            item.get("autoencoder_score", ""),
            item.get("xgboost_score", ""),
            item.get("is_fraud", ""),
            item.get("confidence", ""),
            item.get("feedback_label", "")
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions_export.csv"}
    )

# ─── Health ──────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "models_trained": pipeline.is_trained,
        "training_in_progress": training_in_progress,
        "timestamp": datetime.now().isoformat()
    }
