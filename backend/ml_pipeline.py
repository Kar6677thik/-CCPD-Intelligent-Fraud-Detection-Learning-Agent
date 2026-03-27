"""
ML Pipeline: Isolation Forest, Autoencoder, XGBoost with ensemble voting,
SHAP explainability, drift detection, and active learning.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from scipy import stats

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, precision_score,
    recall_score, accuracy_score, average_precision_score
)

import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


class IsolationForestModel:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.is_trained = False
        self.metrics = {}
        self.version = 0
        self.threshold = 0.5

    def train(self, X: np.ndarray, y: np.ndarray = None, progress_callback=None):
        if progress_callback:
            progress_callback("Isolation Forest: Scaling features...", 0.1)

        X_scaled = self.scaler.fit_transform(X)

        if progress_callback:
            progress_callback("Isolation Forest: Training model...", 0.3)

        self.model = IsolationForest(
            n_estimators=200,
            max_samples='auto',
            contamination=0.00173,  # Approx fraud rate in creditcard.csv
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled)

        if progress_callback:
            progress_callback("Isolation Forest: Computing metrics...", 0.8)

        # Compute scores
        scores = self.model.decision_function(X_scaled)
        # Normalize scores to [0, 1] where higher = more anomalous
        self.min_score = scores.min()
        self.max_score = scores.max()
        normalized = 1 - (scores - self.min_score) / (self.max_score - self.min_score + 1e-10)

        if y is not None:
            from sklearn.metrics import precision_recall_curve as prc
            precisions, recalls, thresholds = prc(y, normalized)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            best_idx = np.argmax(f1_scores)
            self.threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            predictions = (normalized >= self.threshold).astype(int)
            self.metrics = self._compute_metrics(y, predictions, normalized)

        self.is_trained = True
        self.version += 1

        if progress_callback:
            progress_callback("Isolation Forest: Training complete!", 1.0)

        return self.metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise ValueError("IsolationForest model not trained. Call train() first.")
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        normalized = 1 - (scores - self.min_score) / (self.max_score - self.min_score + 1e-10)
        normalized = np.clip(normalized, 0, 1)
        predictions = (normalized >= self.threshold).astype(int)
        return predictions, normalized

    def _compute_metrics(self, y_true, y_pred, y_scores):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, y_scores)),
            "avg_precision": float(average_precision_score(y_true, y_scores)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

    def save(self):
        path = os.path.join(MODEL_DIR, "isolation_forest.pkl")
        joblib.dump({
            "model": self.model, "scaler": self.scaler,
            "threshold": self.threshold, "metrics": self.metrics,
            "version": self.version,
            "min_score": self.min_score, "max_score": self.max_score
        }, path)

    def load(self):
        path = os.path.join(MODEL_DIR, "isolation_forest.pkl")
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.threshold = data["threshold"]
            self.metrics = data["metrics"]
            self.version = data["version"]
            self.min_score = data["min_score"]
            self.max_score = data["max_score"]
            self.is_trained = True
            return True
        return False


class AutoencoderModel:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.is_trained = False
        self.metrics = {}
        self.version = 0
        self.threshold = 0.5

    def _build_model(self, input_dim: int):
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from tensorflow import keras

        encoder = keras.Sequential([
            keras.layers.Dense(input_dim, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(max(1, input_dim // 2), activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(max(1, input_dim // 4), activation='relu'),
            keras.layers.Dense(max(1, input_dim // 8), activation='tanh'),
        ])

        decoder = keras.Sequential([
            keras.layers.Dense(max(1, input_dim // 4), activation='relu', input_shape=(max(1, input_dim // 8),)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(max(1, input_dim // 2), activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(input_dim, activation='linear'),
        ])

        autoencoder_input = keras.layers.Input(shape=(input_dim,))
        encoded = encoder(autoencoder_input)
        decoded = decoder(encoded)
        autoencoder = keras.Model(autoencoder_input, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def train(self, X: np.ndarray, y: np.ndarray = None, progress_callback=None):
        if progress_callback:
            progress_callback("Autoencoder: Scaling features...", 0.05)

        X_scaled = self.scaler.fit_transform(X)

        # Train only on normal transactions
        if y is not None:
            X_normal = X_scaled[y == 0]
        else:
            X_normal = X_scaled

        if progress_callback:
            progress_callback("Autoencoder: Building model...", 0.1)

        self.model = self._build_model(X_scaled.shape[1])

        if progress_callback:
            progress_callback("Autoencoder: Training (this may take a minute)...", 0.15)

        class ProgressCallback:
            def __init__(self, total_epochs, cb):
                self.total = total_epochs
                self.cb = cb

            def on_epoch_end(self, epoch, logs=None):
                progress = 0.15 + (epoch / self.total) * 0.65
                if self.cb and epoch % 5 == 0:
                    self.cb(
                        f"Autoencoder: Epoch {epoch}/{self.total}, loss={logs.get('loss', 0):.6f}",
                        progress
                    )

        import tensorflow as tf
        epochs = 30
        pcb = ProgressCallback(epochs, progress_callback)
        tf_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=pcb.on_epoch_end
        )

        self.model.fit(
            X_normal, X_normal,
            epochs=epochs,
            batch_size=256,
            shuffle=True,
            validation_split=0.1,
            verbose=0,
            callbacks=[tf_callback]
        )

        if progress_callback:
            progress_callback("Autoencoder: Computing reconstruction errors...", 0.85)

        # Compute reconstruction error
        reconstructed = self.model.predict(X_scaled, verbose=0)
        mse = np.mean((X_scaled - reconstructed) ** 2, axis=1)

        # Normalize to [0, 1]
        self.min_error = mse.min()
        self.max_error = np.percentile(mse, 99.9)
        normalized = (mse - self.min_error) / (self.max_error - self.min_error + 1e-10)
        normalized = np.clip(normalized, 0, 1)

        if y is not None:
            from sklearn.metrics import precision_recall_curve as prc
            precisions, recalls, thresholds = prc(y, normalized)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            best_idx = np.argmax(f1_scores)
            self.threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            predictions = (normalized >= self.threshold).astype(int)
            self.metrics = self._compute_metrics(y, predictions, normalized)

        self.is_trained = True
        self.version += 1

        if progress_callback:
            progress_callback("Autoencoder: Training complete!", 1.0)

        return self.metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise ValueError("Autoencoder model not trained. Call train() first.")
        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled, verbose=0)
        mse = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        normalized = (mse - self.min_error) / (self.max_error - self.min_error + 1e-10)
        normalized = np.clip(normalized, 0, 1)
        predictions = (normalized >= self.threshold).astype(int)
        return predictions, normalized

    def _compute_metrics(self, y_true, y_pred, y_scores):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, y_scores)),
            "avg_precision": float(average_precision_score(y_true, y_scores)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

    def save(self):
        model_path = os.path.join(MODEL_DIR, "autoencoder_model.keras")
        meta_path = os.path.join(MODEL_DIR, "autoencoder_meta.pkl")
        self.model.save(model_path)
        joblib.dump({
            "scaler": self.scaler, "threshold": self.threshold,
            "metrics": self.metrics, "version": self.version,
            "min_error": self.min_error, "max_error": self.max_error
        }, meta_path)

    def load(self):
        model_path = os.path.join(MODEL_DIR, "autoencoder_model.keras")
        meta_path = os.path.join(MODEL_DIR, "autoencoder_meta.pkl")
        if os.path.exists(model_path) and os.path.exists(meta_path):
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            self.model = tf.keras.models.load_model(model_path)
            meta = joblib.load(meta_path)
            self.scaler = meta["scaler"]
            self.threshold = meta["threshold"]
            self.metrics = meta["metrics"]
            self.version = meta["version"]
            self.min_error = meta["min_error"]
            self.max_error = meta["max_error"]
            self.is_trained = True
            return True
        return False


class XGBoostModel:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.is_trained = False
        self.metrics = {}
        self.version = 0
        self.threshold = 0.5

    def train(self, X: np.ndarray, y: np.ndarray, progress_callback=None):
        if progress_callback:
            progress_callback("XGBoost: Scaling features...", 0.05)

        X_scaled = self.scaler.fit_transform(X)

        if progress_callback:
            progress_callback("XGBoost: Applying SMOTE for class balancing...", 0.1)

        # Apply SMOTE for class imbalance
        smote = SMOTE(random_state=42, sampling_strategy=0.5)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )

        if progress_callback:
            progress_callback("XGBoost: Training model...", 0.2)

        scale_pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)

        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        if progress_callback:
            progress_callback("XGBoost: Computing metrics...", 0.85)

        # Evaluate on original (non-SMOTE) data
        y_proba = self.model.predict_proba(X_scaled)[:, 1]

        from sklearn.metrics import precision_recall_curve as prc
        precisions, recalls, thresholds = prc(y, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        self.threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        predictions = (y_proba >= self.threshold).astype(int)
        self.metrics = self._compute_metrics(y, predictions, y_proba)

        self.is_trained = True
        self.version += 1

        if progress_callback:
            progress_callback("XGBoost: Training complete!", 1.0)

        return self.metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise ValueError("XGBoost model not trained. Call train() first.")
        X_scaled = self.scaler.transform(X)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        predictions = (y_proba >= self.threshold).astype(int)
        return predictions, y_proba

    def incremental_train(self, X: np.ndarray, y: np.ndarray, progress_callback=None):
        """Retrain with additional feedback data."""
        if progress_callback:
            progress_callback("XGBoost: Incremental training...", 0.2)

        X_scaled = self.scaler.transform(X)

        self.model.fit(
            X_scaled, y,
            xgb_model=self.model.get_booster(),
            eval_set=[(X_scaled, y)],
            verbose=False
        )

        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        predictions = (y_proba >= self.threshold).astype(int)
        self.metrics = self._compute_metrics(y, predictions, y_proba)
        self.version += 1

        if progress_callback:
            progress_callback("XGBoost: Incremental training complete!", 1.0)

        return self.metrics

    def _compute_metrics(self, y_true, y_pred, y_scores):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, y_scores)),
            "avg_precision": float(average_precision_score(y_true, y_scores)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

    def save(self):
        path = os.path.join(MODEL_DIR, "xgboost.pkl")
        joblib.dump({
            "model": self.model, "scaler": self.scaler,
            "threshold": self.threshold, "metrics": self.metrics,
            "version": self.version
        }, path)

    def load(self):
        path = os.path.join(MODEL_DIR, "xgboost.pkl")
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.threshold = data["threshold"]
            self.metrics = data["metrics"]
            self.version = data["version"]
            self.is_trained = True
            return True
        return False


class EnsembleVoter:
    """Weighted voting across all three models."""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "isolation_forest": 0.25,
            "autoencoder": 0.25,
            "xgboost": 0.50
        }

    def predict(self, scores: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        weighted_sum = np.zeros_like(list(scores.values())[0])
        for model_name, model_scores in scores.items():
            weight = self.weights.get(model_name, 1.0 / len(scores))
            weighted_sum += weight * model_scores

        total_weight = sum(self.weights.get(k, 1.0 / len(scores)) for k in scores)
        ensemble_scores = weighted_sum / total_weight
        predictions = (ensemble_scores >= 0.5).astype(int)
        return predictions, ensemble_scores


class DriftDetector:
    """Detects distribution drift between training and new data."""

    def __init__(self):
        self.reference_stats = None
        self.feature_names = None

    def set_reference(self, X: np.ndarray, feature_names: list = None):
        self.feature_names = feature_names or [f"V{i}" for i in range(X.shape[1])]
        self.reference_stats = {
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0),
            "percentiles": {
                "25": np.percentile(X, 25, axis=0),
                "50": np.percentile(X, 50, axis=0),
                "75": np.percentile(X, 75, axis=0)
            }
        }
        self._reference_data = X[:5000] if len(X) > 5000 else X  # Keep sample

    def detect_drift(self, X_new: np.ndarray) -> Dict:
        if self.reference_stats is None:
            return {"drifted": False, "message": "No reference data set"}

        results = {"features": {}, "overall_drift": False, "drift_score": 0.0}
        drift_count = 0

        for i in range(min(X_new.shape[1], self._reference_data.shape[1])):
            stat, p_value = stats.ks_2samp(self._reference_data[:, i], X_new[:, i])
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            is_drifted = p_value < 0.01
            if is_drifted:
                drift_count += 1
            results["features"][feature_name] = {
                "ks_statistic": float(stat),
                "p_value": float(p_value),
                "is_drifted": is_drifted
            }

        results["drift_score"] = drift_count / X_new.shape[1]
        results["overall_drift"] = results["drift_score"] > 0.3
        results["num_drifted_features"] = drift_count
        results["total_features"] = X_new.shape[1]
        return results


class ActiveLearner:
    """Identifies most uncertain predictions for manual review."""

    @staticmethod
    def get_uncertain_samples(scores: np.ndarray, n_samples: int = 20) -> np.ndarray:
        uncertainty = np.abs(scores - 0.5)
        indices = np.argsort(uncertainty)[:n_samples]
        return indices

    @staticmethod
    def get_disagreement_samples(
        model_predictions: Dict[str, np.ndarray], n_samples: int = 20
    ) -> np.ndarray:
        preds = np.stack(list(model_predictions.values()), axis=0)
        disagreement = np.std(preds, axis=0)
        indices = np.argsort(disagreement)[::-1][:n_samples]
        return indices


class FraudDetectionPipeline:
    """Main pipeline orchestrating all models and components."""

    def __init__(self):
        self.isolation_forest = IsolationForestModel()
        self.autoencoder = AutoencoderModel()
        self.xgboost = XGBoostModel()
        self.ensemble = EnsembleVoter()
        self.drift_detector = DriftDetector()
        self.active_learner = ActiveLearner()
        self.feature_names = []
        self.is_trained = False
        self.training_data_shape = None

    def load_and_prepare_data(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
        df = pd.read_csv(csv_path)

        # Handle creditcard.csv format
        if "Class" in df.columns:
            y = df["Class"].values
            # Drop non-feature columns
            drop_cols = ["Class"]
            if "Time" in df.columns:
                drop_cols.append("Time")
            X = df.drop(columns=drop_cols).values
            feature_names = [c for c in df.columns if c not in drop_cols]
        else:
            # Assume last column is label
            y = df.iloc[:, -1].values
            X = df.iloc[:, :-1].values
            feature_names = list(df.columns[:-1])

        self.feature_names = feature_names
        self.training_data_shape = X.shape
        return X, y, feature_names

    def train_all(self, X: np.ndarray, y: np.ndarray, progress_callback=None):
        results = {}

        def if_progress(msg, pct):
            if progress_callback:
                progress_callback("isolation_forest", msg, pct)

        results["isolation_forest"] = self.isolation_forest.train(X, y, if_progress)

        def ae_progress(msg, pct):
            if progress_callback:
                progress_callback("autoencoder", msg, pct)

        results["autoencoder"] = self.autoencoder.train(X, y, ae_progress)

        def xgb_progress(msg, pct):
            if progress_callback:
                progress_callback("xgboost", msg, pct)

        results["xgboost"] = self.xgboost.train(X, y, xgb_progress)

        # Set drift reference
        self.drift_detector.set_reference(X, self.feature_names)

        # Save all models
        self.isolation_forest.save()
        self.autoencoder.save()
        self.xgboost.save()

        self.is_trained = True
        return results

    def predict(self, X: np.ndarray) -> Dict:
        if_preds, if_scores = self.isolation_forest.predict(X)
        ae_preds, ae_scores = self.autoencoder.predict(X)
        xgb_preds, xgb_scores = self.xgboost.predict(X)

        scores = {
            "isolation_forest": if_scores,
            "autoencoder": ae_scores,
            "xgboost": xgb_scores
        }

        ensemble_preds, ensemble_scores = self.ensemble.predict(scores)

        return {
            "predictions": ensemble_preds,
            "ensemble_scores": ensemble_scores,
            "isolation_forest_scores": if_scores,
            "autoencoder_scores": ae_scores,
            "xgboost_scores": xgb_scores,
            "isolation_forest_preds": if_preds,
            "autoencoder_preds": ae_preds,
            "xgboost_preds": xgb_preds
        }

    def get_feature_importance(self) -> Dict:
        importance = {}

        # XGBoost feature importance
        if self.xgboost.is_trained:
            xgb_importance = self.xgboost.model.feature_importances_
            # Use min to prevent IndexError
            num_features = min(len(self.feature_names), len(xgb_importance))
            importance["xgboost"] = {
                self.feature_names[i]: float(xgb_importance[i])
                for i in range(num_features)
            }
            importance["xgboost"] = dict(
                sorted(importance["xgboost"].items(), key=lambda x: x[1], reverse=True)
            )

        return importance

    def get_shap_explanations(self, X: np.ndarray, max_samples: int = 100) -> Dict:
        """Generate SHAP explanations for XGBoost predictions."""
        try:
            import shap
            if not self.xgboost.is_trained:
                return {"error": "XGBoost model not trained"}

            X_scaled = self.xgboost.scaler.transform(X[:max_samples])
            explainer = shap.TreeExplainer(self.xgboost.model)
            shap_values = explainer.shap_values(X_scaled)

            # Get top features per prediction
            explanations = []
            for i in range(len(X_scaled)):
                feature_impacts = {}
                for j, fname in enumerate(self.feature_names):
                    feature_impacts[fname] = float(shap_values[i, j])

                sorted_impacts = sorted(
                    feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True
                )[:10]

                explanation_text = []
                for fname, impact in sorted_impacts:
                    direction = "increases" if impact > 0 else "decreases"
                    explanation_text.append(
                        f"{fname} {direction} fraud risk (impact: {impact:+.4f})"
                    )

                explanations.append({
                    "top_features": dict(sorted_impacts),
                    "explanation": "; ".join(explanation_text[:5]),
                    "base_value": float(explainer.expected_value)
                        if isinstance(explainer.expected_value, (int, float))
                        else float(explainer.expected_value[0])
                })

            # Global feature importance from SHAP
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            global_importance = {
                self.feature_names[i]: float(mean_abs_shap[i])
                for i in range(len(self.feature_names))
            }
            global_importance = dict(
                sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
            )

            return {
                "per_transaction": explanations,
                "global_importance": global_importance
            }
        except Exception as e:
            return {"error": str(e)}

    def load_models(self) -> bool:
        loaded = []
        if self.isolation_forest.load():
            loaded.append("isolation_forest")
        if self.autoencoder.load():
            loaded.append("autoencoder")
        if self.xgboost.load():
            loaded.append("xgboost")

        self.is_trained = len(loaded) == 3
        return self.is_trained

    def get_all_metrics(self) -> Dict:
        return {
            "isolation_forest": self.isolation_forest.metrics,
            "autoencoder": self.autoencoder.metrics,
            "xgboost": self.xgboost.metrics,
            "models_trained": self.is_trained,
            "versions": {
                "isolation_forest": self.isolation_forest.version,
                "autoencoder": self.autoencoder.version,
                "xgboost": self.xgboost.version
            }
        }
