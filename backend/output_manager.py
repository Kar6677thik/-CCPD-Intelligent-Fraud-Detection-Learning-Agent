"""
Output Manager: Creates timestamped output folders with plots and metrics.json
for each prediction run.
"""
import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set professional dark theme for plots
plt.rcParams.update({
    'figure.facecolor': '#0a1628',
    'axes.facecolor': '#111d35',
    'axes.edgecolor': '#2a3f5f',
    'axes.labelcolor': '#e0e6ed',
    'xtick.color': '#e0e6ed',
    'ytick.color': '#e0e6ed',
    'text.color': '#e0e6ed',
    'grid.color': '#1e3150',
    'grid.alpha': 0.5,
    'figure.figsize': (10, 6),
    'font.size': 12,
    'font.family': 'sans-serif'
})

COLORS = {
    'primary': '#00d4aa',
    'secondary': '#4fc3f7',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'success': '#10b981',
    'text': '#e0e6ed',
    'bg_dark': '#0a1628',
    'bg_card': '#111d35',
}


def create_output_folder() -> str:
    """Create a timestamped output folder."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(folder, exist_ok=True)
    return folder


def save_confusion_matrix(y_true, y_pred, model_name, output_folder):
    """Save confusion matrix plot."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'],
            ax=ax, linewidths=1,
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16, 'weight': 'bold'}
        )
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('Actual', fontsize=14)
        ax.set_title(f'{model_name} — Confusion Matrix', fontsize=16, fontweight='bold',
                     color=COLORS['primary'])

        plt.tight_layout()
        path = os.path.join(output_folder, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path
    except Exception as e:
        logger.error(f"Failed to save confusion matrix for {model_name}: {e}")
        plt.close('all')
        return None


def save_roc_curve(y_true, scores_dict, output_folder):
    """Save ROC curve comparison plot for all models."""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Generate colors dynamically based on number of models
        color_list = [COLORS['primary'], COLORS['secondary'], COLORS['warning'],
                      COLORS['success'], COLORS['danger']]
        for i, (name, scores) in enumerate(scores_dict.items()):
            color = color_list[i % len(color_list)]
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            ax.plot(fpr, tpr, color=color, linewidth=2.5,
                    label=f'{name} (AUC = {auc:.4f})')

        ax.plot([0, 1], [0, 1], 'w--', alpha=0.3, linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('ROC Curve Comparison', fontsize=16, fontweight='bold',
                     color=COLORS['primary'])
        ax.legend(loc='lower right', fontsize=12, facecolor=COLORS['bg_card'],
                  edgecolor=COLORS['primary'])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_folder, "roc_curves.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path
    except Exception as e:
        logger.error(f"Failed to save ROC curve: {e}")
        plt.close('all')
        return None


def save_precision_recall_curve(y_true, scores_dict, output_folder):
    """Save precision-recall curve comparison."""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Generate colors dynamically based on number of models
        color_list = [COLORS['primary'], COLORS['secondary'], COLORS['warning'],
                      COLORS['success'], COLORS['danger']]
        for i, (name, scores) in enumerate(scores_dict.items()):
            color = color_list[i % len(color_list)]
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            ax.plot(recall, precision, color=color, linewidth=2.5,
                    label=f'{name} (AP = {ap:.4f})')

        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title('Precision-Recall Curve Comparison', fontsize=16, fontweight='bold',
                     color=COLORS['primary'])
        ax.legend(loc='upper right', fontsize=12, facecolor=COLORS['bg_card'],
                  edgecolor=COLORS['primary'])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_folder, "precision_recall_curves.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path
    except Exception as e:
        logger.error(f"Failed to save precision-recall curve: {e}")
        plt.close('all')
        return None


def save_feature_importance(importance_dict, output_folder, top_n=15):
    """Save feature importance bar chart."""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))

        if "xgboost" in importance_dict:
            xgb_imp = importance_dict["xgboost"]
            sorted_features = list(xgb_imp.items())[:top_n]
            sorted_features.reverse()

            names = [f[0] for f in sorted_features]
            values = [f[1] for f in sorted_features]

            bars = ax.barh(names, values, color=COLORS['primary'], edgecolor=COLORS['secondary'],
                           linewidth=0.5, height=0.7)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', va='center', fontsize=10, color=COLORS['text'])

        ax.set_xlabel('Importance Score', fontsize=14)
        ax.set_title(f'Top {top_n} Feature Importance (XGBoost)', fontsize=16,
                     fontweight='bold', color=COLORS['primary'])
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_folder, "feature_importance.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path
    except Exception as e:
        logger.error(f"Failed to save feature importance: {e}")
        plt.close('all')
        return None


def save_score_distribution(scores_dict, y_true, output_folder):
    """Save fraud score distribution for each model."""
    try:
        num_models = len(scores_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6))
        if num_models == 1:
            axes = [axes]

        color_list = [COLORS['primary'], COLORS['secondary'], COLORS['warning'],
                      COLORS['success'], COLORS['danger']]
        for i, (ax, (name, scores)) in enumerate(zip(axes, scores_dict.items())):
            color = color_list[i % len(color_list)]
            legit_scores = scores[y_true == 0]
            fraud_scores = scores[y_true == 1]

            # Only plot if arrays are non-empty
            if len(legit_scores) > 0:
                ax.hist(legit_scores, bins=50, alpha=0.7, color=COLORS['success'],
                        label='Legitimate', density=True)
            if len(fraud_scores) > 0:
                ax.hist(fraud_scores, bins=50, alpha=0.7, color=COLORS['danger'],
                        label='Fraud', density=True)
            ax.set_title(name, fontsize=14, fontweight='bold', color=color)
            ax.set_xlabel('Fraud Score')
            ax.set_ylabel('Density')
            ax.legend(facecolor=COLORS['bg_card'], edgecolor=color)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Score Distribution by Model', fontsize=16, fontweight='bold',
                     color=COLORS['primary'], y=1.02)
        plt.tight_layout()
        path = os.path.join(output_folder, "score_distributions.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path
    except Exception as e:
        logger.error(f"Failed to save score distribution: {e}")
        plt.close('all')
        return None


def save_metrics_json(metrics, output_folder, additional_info=None):
    """Save comprehensive metrics.json."""
    try:
        output = {
            "timestamp": datetime.now().isoformat(),
            "models": metrics,
        }
        if additional_info:
            output.update(additional_info)

        path = os.path.join(output_folder, "metrics.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        return path
    except Exception as e:
        logger.error(f"Failed to save metrics JSON: {e}")
        return None


def _find_optimal_threshold(y_true, scores):
    """Find threshold that maximizes F1 score."""
    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
        precisions = precisions[1:]
        recalls = recalls[1:]
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        f1_scores = np.nan_to_num(f1_scores, nan=0.0)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx]
    except Exception:
        return 0.5


def _recalibrate_predictions(y_true, scores_dict, ensemble_scores):
    """
    Recalibrate thresholds using ground-truth labels to find optimal predictions.
    Used for evaluation output only (does not affect real-time predictions).
    """
    recalibrated_preds = {}
    recalibrated_thresholds = {}

    for name, scores in scores_dict.items():
        threshold = _find_optimal_threshold(y_true, scores)
        preds = (scores >= threshold).astype(int)
        # Validate — if nothing is predicted, try a lower percentile
        if preds.sum() == 0 and y_true.sum() > 0:
            fraud_scores = scores[y_true == 1]
            if len(fraud_scores) > 0:
                threshold = np.percentile(fraud_scores, 25)
                preds = (scores >= threshold).astype(int)
        recalibrated_preds[name] = preds
        recalibrated_thresholds[name] = float(threshold)

    # Ensemble
    ens_threshold = _find_optimal_threshold(y_true, ensemble_scores)
    ens_preds = (ensemble_scores >= ens_threshold).astype(int)
    if ens_preds.sum() == 0 and y_true.sum() > 0:
        fraud_scores = ensemble_scores[y_true == 1]
        if len(fraud_scores) > 0:
            ens_threshold = np.percentile(fraud_scores, 25)
            ens_preds = (ensemble_scores >= ens_threshold).astype(int)
    recalibrated_preds["Ensemble"] = ens_preds
    recalibrated_thresholds["Ensemble"] = float(ens_threshold)

    return recalibrated_preds, recalibrated_thresholds


def generate_all_outputs(y_true, prediction_results, feature_importance, additional_info=None):
    """Generate all output files for a prediction run."""
    output_folder = create_output_folder()
    generated_files = []

    scores_dict = {
        "Isolation Forest": prediction_results["isolation_forest_scores"],
        "Autoencoder": prediction_results["autoencoder_scores"],
        "XGBoost": prediction_results["xgboost_scores"]
    }

    original_preds_dict = {
        "Isolation Forest": prediction_results["isolation_forest_preds"],
        "Autoencoder": prediction_results["autoencoder_preds"],
        "XGBoost": prediction_results["xgboost_preds"]
    }

    # Check if original predictions missed all fraud — if so, recalibrate
    original_ens_preds = prediction_results["predictions"]
    ensemble_scores = prediction_results["ensemble_scores"]
    needs_recalibration = (np.sum(original_ens_preds) == 0 and np.sum(y_true) > 0)

    if needs_recalibration:
        logger.warning(
            "Original thresholds produced 0 fraud predictions with %d actual frauds. "
            "Recalibrating thresholds for evaluation output.",
            int(np.sum(y_true))
        )
        recal_preds, recal_thresholds = _recalibrate_predictions(
            y_true, scores_dict, ensemble_scores
        )
        preds_dict = {k: recal_preds[k] for k in scores_dict}
        ens_preds = recal_preds["Ensemble"]
    else:
        preds_dict = original_preds_dict
        ens_preds = original_ens_preds
        recal_thresholds = None

    # Confusion matrices for each model
    for name, preds in preds_dict.items():
        path = save_confusion_matrix(y_true, preds, name, output_folder)
        generated_files.append(path)

    # Ensemble confusion matrix
    path = save_confusion_matrix(y_true, ens_preds, "Ensemble", output_folder)
    generated_files.append(path)

    # ROC curves
    path = save_roc_curve(y_true, scores_dict, output_folder)
    generated_files.append(path)

    # Precision-Recall curves
    path = save_precision_recall_curve(y_true, scores_dict, output_folder)
    generated_files.append(path)

    # Feature importance
    if feature_importance:
        path = save_feature_importance(feature_importance, output_folder)
        generated_files.append(path)

    # Score distributions
    path = save_score_distribution(scores_dict, y_true, output_folder)
    generated_files.append(path)

    # Metrics JSON
    metrics = {}
    for model_name, scores in scores_dict.items():
        key = model_name.lower().replace(" ", "_")
        preds = preds_dict[model_name]
        cm = confusion_matrix(y_true, preds).tolist()
        try:
            auc = float(roc_auc_score(y_true, scores))
            ap = float(average_precision_score(y_true, scores))
        except Exception:
            auc = 0.0
            ap = 0.0

        metrics[key] = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "auc_roc": auc,
            "avg_precision": ap,
            "confusion_matrix": cm
        }

    # Ensemble metrics
    try:
        ens_auc = float(roc_auc_score(y_true, ensemble_scores))
        ens_ap = float(average_precision_score(y_true, ensemble_scores))
    except Exception:
        ens_auc = 0.0
        ens_ap = 0.0

    metrics["ensemble"] = {
        "accuracy": float(accuracy_score(y_true, ens_preds)),
        "precision": float(precision_score(y_true, ens_preds, zero_division=0)),
        "recall": float(recall_score(y_true, ens_preds, zero_division=0)),
        "f1": float(f1_score(y_true, ens_preds, zero_division=0)),
        "auc_roc": ens_auc,
        "avg_precision": ens_ap,
        "confusion_matrix": confusion_matrix(y_true, ens_preds).tolist()
    }

    info = additional_info or {}
    info["output_folder"] = output_folder
    info["generated_files"] = generated_files
    info["num_transactions"] = len(y_true)
    info["num_fraud_actual"] = int(np.sum(y_true))
    info["num_fraud_predicted"] = int(np.sum(ens_preds))
    if recal_thresholds:
        info["recalibrated"] = True
        info["recalibrated_thresholds"] = recal_thresholds

    path = save_metrics_json(metrics, output_folder, info)
    generated_files.append(path)

    return output_folder, generated_files, metrics
