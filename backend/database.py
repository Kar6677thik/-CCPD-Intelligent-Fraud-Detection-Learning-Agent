"""
Database module for storing predictions, feedback, and model versions.
Uses SQLite with async support via aiosqlite.
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fraud_detection.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            features TEXT,
            isolation_forest_score REAL,
            autoencoder_score REAL,
            xgboost_score REAL,
            ensemble_score REAL,
            is_fraud INTEGER,
            confidence REAL,
            explanation TEXT,
            batch_id TEXT,
            feedback_label INTEGER DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            version INTEGER,
            timestamp TEXT DEFAULT (datetime('now')),
            metrics TEXT,
            training_samples INTEGER,
            is_active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            filename TEXT,
            upload_time TEXT DEFAULT (datetime('now')),
            num_rows INTEGER,
            num_features INTEGER,
            fraud_ratio REAL,
            is_active INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now')),
            model_name TEXT,
            metrics_before TEXT,
            metrics_after TEXT,
            num_feedback_samples INTEGER,
            improvement REAL
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now')),
            transaction_id TEXT,
            fraud_score REAL,
            alert_type TEXT,
            acknowledged INTEGER DEFAULT 0
        );
    """)
    conn.commit()
    conn.close()


def save_predictions(predictions: list, batch_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    for pred in predictions:
        cursor.execute("""
            INSERT INTO predictions 
            (transaction_id, features, isolation_forest_score, autoencoder_score,
             xgboost_score, ensemble_score, is_fraud, confidence, explanation, batch_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred.get("transaction_id", ""),
            json.dumps(pred.get("features", {})),
            pred.get("isolation_forest_score", 0),
            pred.get("autoencoder_score", 0),
            pred.get("xgboost_score", 0),
            pred.get("ensemble_score", 0),
            pred.get("is_fraud", 0),
            pred.get("confidence", 0),
            json.dumps(pred.get("explanation", {})),
            batch_id
        ))
    conn.commit()
    conn.close()


def get_predictions(page: int = 1, per_page: int = 50, fraud_only: bool = False,
                    min_score: float = 0.0, search: str = ""):
    conn = get_connection()
    cursor = conn.cursor()
    where_clauses = []
    params = []

    if fraud_only:
        where_clauses.append("is_fraud = 1")
    if min_score > 0:
        where_clauses.append("ensemble_score >= ?")
        params.append(min_score)
    if search:
        where_clauses.append("transaction_id LIKE ?")
        params.append(f"%{search}%")

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    offset = (page - 1) * per_page

    count_sql = f"SELECT COUNT(*) FROM predictions {where_sql}"
    cursor.execute(count_sql, params)
    total = cursor.fetchone()[0]

    query = f"""
        SELECT * FROM predictions {where_sql}
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """
    cursor.execute(query, params + [per_page, offset])
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    for row in rows:
        if row.get("features"):
            try:
                row["features"] = json.loads(row["features"])
            except Exception:
                pass
        if row.get("explanation"):
            try:
                row["explanation"] = json.loads(row["explanation"])
            except Exception:
                pass

    return {"items": rows, "total": total, "page": page, "per_page": per_page}


def save_feedback(transaction_ids: list, labels: list):
    conn = get_connection()
    cursor = conn.cursor()
    for tid, label in zip(transaction_ids, labels):
        cursor.execute(
            "UPDATE predictions SET feedback_label = ? WHERE transaction_id = ?",
            (label, tid)
        )
    conn.commit()
    conn.close()


def get_feedback_data():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT features, feedback_label FROM predictions 
        WHERE feedback_label IS NOT NULL
    """)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    for row in rows:
        if row.get("features"):
            try:
                row["features"] = json.loads(row["features"])
            except Exception:
                pass
    return rows


def save_model_version(model_name: str, version: int, metrics: dict,
                       training_samples: int):
    conn = get_connection()
    cursor = conn.cursor()
    # Deactivate previous versions
    cursor.execute(
        "UPDATE model_versions SET is_active = 0 WHERE model_name = ?",
        (model_name,)
    )
    cursor.execute("""
        INSERT INTO model_versions (model_name, version, metrics, training_samples, is_active)
        VALUES (?, ?, ?, ?, 1)
    """, (model_name, version, json.dumps(metrics), training_samples))
    conn.commit()
    conn.close()


def get_model_versions(model_name: Optional[str] = None):
    conn = get_connection()
    cursor = conn.cursor()
    if model_name:
        cursor.execute(
            "SELECT * FROM model_versions WHERE model_name = ? ORDER BY version DESC",
            (model_name,)
        )
    else:
        cursor.execute("SELECT * FROM model_versions ORDER BY timestamp DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    for row in rows:
        if row.get("metrics"):
            try:
                row["metrics"] = json.loads(row["metrics"])
            except Exception:
                pass
    return rows


def save_training_history(model_name: str, metrics_before: dict,
                          metrics_after: dict, num_feedback: int):
    improvement = (metrics_after.get("f1", 0) - metrics_before.get("f1", 0))
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO training_history 
        (model_name, metrics_before, metrics_after, num_feedback_samples, improvement)
        VALUES (?, ?, ?, ?, ?)
    """, (
        model_name,
        json.dumps(metrics_before),
        json.dumps(metrics_after),
        num_feedback,
        improvement
    ))
    conn.commit()
    conn.close()


def get_training_history():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM training_history ORDER BY timestamp DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    for row in rows:
        for key in ["metrics_before", "metrics_after"]:
            if row.get(key):
                try:
                    row[key] = json.loads(row[key])
                except Exception:
                    pass
    return rows


def save_alert(transaction_id: str, fraud_score: float, alert_type: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO alerts (transaction_id, fraud_score, alert_type)
        VALUES (?, ?, ?)
    """, (transaction_id, fraud_score, alert_type))
    conn.commit()
    conn.close()


def get_alerts(acknowledged: Optional[bool] = None):
    conn = get_connection()
    cursor = conn.cursor()
    if acknowledged is not None:
        cursor.execute(
            "SELECT * FROM alerts WHERE acknowledged = ? ORDER BY timestamp DESC",
            (1 if acknowledged else 0,)
        )
    else:
        cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def acknowledge_alert(alert_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()


def save_dataset_info(name: str, filename: str, num_rows: int,
                      num_features: int, fraud_ratio: float):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE datasets SET is_active = 0"
    )
    cursor.execute("""
        INSERT INTO datasets (name, filename, num_rows, num_features, fraud_ratio, is_active)
        VALUES (?, ?, ?, ?, ?, 1)
    """, (name, filename, num_rows, num_features, fraud_ratio))
    conn.commit()
    conn.close()


def get_datasets():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets ORDER BY upload_time DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_dashboard_stats():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE is_fraud = 1")
    total_fraud = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(ensemble_score) FROM predictions WHERE is_fraud = 1")
    row = cursor.fetchone()
    avg_fraud_score = row[0] if row[0] else 0

    cursor.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged = 0")
    pending_alerts = cursor.fetchone()[0]

    cursor.execute("""
        SELECT * FROM model_versions WHERE is_active = 1
        ORDER BY timestamp DESC
    """)
    active_models = [dict(row) for row in cursor.fetchall()]
    for m in active_models:
        if m.get("metrics"):
            try:
                m["metrics"] = json.loads(m["metrics"])
            except Exception:
                pass

    conn.close()

    return {
        "total_predictions": total_predictions,
        "total_fraud": total_fraud,
        "fraud_rate": (total_fraud / total_predictions * 100) if total_predictions > 0 else 0,
        "avg_fraud_confidence": round(avg_fraud_score, 4),
        "pending_alerts": pending_alerts,
        "active_models": active_models
    }
