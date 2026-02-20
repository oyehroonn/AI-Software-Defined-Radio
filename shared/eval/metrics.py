"""Evaluation metrics for anomaly detection models."""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
        confusion_matrix,
        classification_report,
        f1_score,
        precision_score,
        recall_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def evaluate_detector(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    detection_times: Optional[np.ndarray] = None,
    target_far: float = 0.01
) -> dict:
    """
    Compute comprehensive detector performance metrics.
    
    Args:
        y_true: Ground truth labels (0=normal, 1=anomaly)
        y_scores: Anomaly scores from detector
        detection_times: Optional time-to-detect values
        target_far: Target false alarm rate for threshold selection
        
    Returns:
        Dictionary of performance metrics
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available, returning basic metrics")
        return {"error": "sklearn required for metrics"}
        
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    metrics = {}
    
    # ROC metrics
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_scores))
    except ValueError as e:
        logger.warning(f"Could not compute AUC-ROC: {e}")
        metrics["auc_roc"] = None
        
    # Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    
    # Find threshold for target FAR
    idx_far = np.argmin(np.abs(fpr - target_far))
    threshold_at_far = roc_thresholds[idx_far] if idx_far < len(roc_thresholds) else 0.5
    
    metrics["threshold_at_target_far"] = float(threshold_at_far)
    metrics["actual_far_at_threshold"] = float(fpr[idx_far])
    metrics["tpr_at_target_far"] = float(tpr[idx_far])
    
    # Find corresponding precision/recall
    pr_idx = np.argmin(np.abs(thresholds - threshold_at_far)) if len(thresholds) > 0 else 0
    metrics["precision_at_target_far"] = float(precisions[pr_idx]) if pr_idx < len(precisions) else 0
    metrics["recall_at_target_far"] = float(recalls[pr_idx]) if pr_idx < len(recalls) else 0
    
    # Binary predictions at threshold
    y_pred = (y_scores >= threshold_at_far).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    
    # Derived metrics
    metrics["accuracy"] = float((tp + tn) / (tp + tn + fp + fn))
    metrics["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0
    metrics["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
    metrics["f1_score"] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
    
    # Time-to-detect metrics
    if detection_times is not None:
        detection_times = np.asarray(detection_times)
        detected_mask = (y_true == 1) & (y_scores >= threshold_at_far)
        
        if np.sum(detected_mask) > 0:
            detected_times = detection_times[detected_mask]
            metrics["mean_time_to_detect"] = float(np.mean(detected_times))
            metrics["median_time_to_detect"] = float(np.median(detected_times))
            metrics["p95_time_to_detect"] = float(np.percentile(detected_times, 95))
            metrics["min_time_to_detect"] = float(np.min(detected_times))
            metrics["max_time_to_detect"] = float(np.max(detected_times))
        else:
            metrics["mean_time_to_detect"] = None
            
    # Operating point metrics at different thresholds
    metrics["operating_points"] = []
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        y_pred_t = (y_scores >= threshold).astype(int)
        metrics["operating_points"].append({
            "threshold": threshold,
            "precision": float(precision_score(y_true, y_pred_t, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred_t, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred_t, zero_division=0))
        })
        
    return metrics


def evaluate_by_attack_type(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    attack_types: np.ndarray
) -> dict:
    """
    Evaluate detector performance broken down by attack type.
    
    Args:
        y_true: Ground truth labels
        y_scores: Anomaly scores
        attack_types: Array of attack type labels
        
    Returns:
        Dictionary with metrics per attack type
    """
    results = {}
    
    unique_types = np.unique(attack_types)
    
    for attack_type in unique_types:
        if attack_type in ["none", "legitimate"]:
            continue
            
        mask = attack_types == attack_type
        if np.sum(mask) == 0:
            continue
            
        type_scores = y_scores[mask]
        type_labels = y_true[mask]
        
        # For attack samples, compute detection rate at various thresholds
        results[attack_type] = {
            "count": int(np.sum(mask)),
            "detection_rate_0.3": float(np.mean(type_scores >= 0.3)),
            "detection_rate_0.5": float(np.mean(type_scores >= 0.5)),
            "detection_rate_0.7": float(np.mean(type_scores >= 0.7)),
            "mean_score": float(np.mean(type_scores)),
            "std_score": float(np.std(type_scores)),
            "min_score": float(np.min(type_scores)),
            "max_score": float(np.max(type_scores))
        }
        
    return results


def compute_alarm_rate(
    y_scores: np.ndarray,
    threshold: float,
    window_size: int = 100
) -> np.ndarray:
    """
    Compute rolling alarm rate.
    
    Args:
        y_scores: Anomaly scores
        threshold: Alarm threshold
        window_size: Rolling window size
        
    Returns:
        Array of rolling alarm rates
    """
    alarms = (y_scores >= threshold).astype(float)
    
    if len(alarms) < window_size:
        return np.array([np.mean(alarms)])
        
    # Rolling mean
    kernel = np.ones(window_size) / window_size
    rolling_rate = np.convolve(alarms, kernel, mode='valid')
    
    return rolling_rate


class EvaluationReport:
    """Generate formatted evaluation reports."""
    
    def __init__(self, metrics: dict):
        self.metrics = metrics
        
    def to_text(self) -> str:
        """Generate text report."""
        lines = [
            "=" * 50,
            "ANOMALY DETECTION EVALUATION REPORT",
            "=" * 50,
            "",
            "Overall Performance:",
            f"  AUC-ROC: {self.metrics.get('auc_roc', 'N/A'):.4f}" if self.metrics.get('auc_roc') else "  AUC-ROC: N/A",
            f"  Accuracy: {self.metrics.get('accuracy', 0):.4f}",
            f"  Precision: {self.metrics.get('precision', 0):.4f}",
            f"  Recall: {self.metrics.get('recall', 0):.4f}",
            f"  F1 Score: {self.metrics.get('f1_score', 0):.4f}",
            "",
            "Confusion Matrix:",
            f"  True Positives: {self.metrics.get('true_positives', 0)}",
            f"  True Negatives: {self.metrics.get('true_negatives', 0)}",
            f"  False Positives: {self.metrics.get('false_positives', 0)}",
            f"  False Negatives: {self.metrics.get('false_negatives', 0)}",
            "",
            f"At Target FAR ({self.metrics.get('actual_far_at_threshold', 0):.2%}):",
            f"  Threshold: {self.metrics.get('threshold_at_target_far', 0):.4f}",
            f"  True Positive Rate: {self.metrics.get('tpr_at_target_far', 0):.4f}",
            ""
        ]
        
        if self.metrics.get('mean_time_to_detect'):
            lines.extend([
                "Time-to-Detect:",
                f"  Mean: {self.metrics['mean_time_to_detect']:.2f}s",
                f"  Median: {self.metrics.get('median_time_to_detect', 0):.2f}s",
                f"  95th Percentile: {self.metrics.get('p95_time_to_detect', 0):.2f}s",
                ""
            ])
            
        lines.append("=" * 50)
        
        return "\n".join(lines)
        
    def to_dict(self) -> dict:
        """Return metrics as dictionary."""
        return self.metrics
