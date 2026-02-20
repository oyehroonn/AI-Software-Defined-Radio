"""Machine learning based anomaly detection for ADS-B tracks."""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import joblib

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrackAnomalyDetector:
    """Unsupervised anomaly detector using Isolation Forest."""
    
    FEATURE_COLUMNS = [
        "velocity_mean",
        "velocity_std", 
        "velocity_max_delta",
        "altitude_std",
        "altitude_range",
        "max_climb_rate",
        "max_descent_rate",
        "max_turn_rate",
        "max_implied_speed",
        "speed_consistency_error",
        "msg_rate",
        "msg_interval_std",
        "max_acceleration"
    ]
    
    def __init__(
        self,
        contamination: float = 0.01,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for TrackAnomalyDetector")
            
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples='auto',
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        
    def _extract_feature_vector(self, features: dict) -> np.ndarray:
        """Extract feature vector from features dictionary."""
        return np.array([
            features.get(col, 0) or 0
            for col in self.FEATURE_COLUMNS
        ]).reshape(1, -1)
        
    def fit(self, features_list: list[dict]):
        """Fit the anomaly detector on training data."""
        if len(features_list) < 10:
            logger.warning("Insufficient training data for anomaly detector")
            return
            
        # Build feature matrix
        X = np.vstack([
            self._extract_feature_vector(f)
            for f in features_list
        ])
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        logger.info(f"Anomaly detector fitted on {len(features_list)} samples")
        
    def predict(self, features: dict) -> tuple[float, bool]:
        """
        Predict anomaly score for a single track.
        
        Returns:
            (anomaly_score, is_anomaly) - score is higher for more anomalous
        """
        if not self.is_fitted:
            logger.warning("Anomaly detector not fitted, returning default score")
            return 0.0, False
            
        X = self._extract_feature_vector(features)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly score (negative of decision function, higher = more anomalous)
        score = -self.model.score_samples(X_scaled)[0]
        
        # Get binary prediction (-1 for anomaly, 1 for normal)
        is_anomaly = self.model.predict(X_scaled)[0] == -1
        
        return float(score), bool(is_anomaly)
        
    def predict_batch(self, features_list: list[dict]) -> list[tuple[str, float, bool]]:
        """Predict anomaly scores for multiple tracks."""
        results = []
        
        for features in features_list:
            icao24 = features.get("icao24", "unknown")
            score, is_anomaly = self.predict(features)
            results.append((icao24, score, is_anomaly))
            
        return results
        
    def save(self, path: str):
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted, cannot save")
            
        save_data = {
            'scaler': self.scaler,
            'model': self.model,
            'feature_columns': self.FEATURE_COLUMNS,
            'contamination': self.contamination
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_data, path)
        logger.info(f"Model saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'TrackAnomalyDetector':
        """Load a fitted model from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        save_data = joblib.load(path)
        
        detector = cls(contamination=save_data['contamination'])
        detector.scaler = save_data['scaler']
        detector.model = save_data['model']
        detector.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        return detector


class EnsembleAnomalyDetector:
    """Ensemble combining rule-based and ML-based detection."""
    
    def __init__(
        self,
        ml_detector: Optional[TrackAnomalyDetector] = None,
        rule_weight: float = 0.4,
        ml_weight: float = 0.6
    ):
        self.ml_detector = ml_detector
        self.rule_weight = rule_weight
        self.ml_weight = ml_weight
        
    def score(
        self,
        features: dict,
        rule_triggers: list
    ) -> tuple[float, str]:
        """
        Compute ensemble anomaly score.
        
        Returns:
            (combined_score, confidence_level)
        """
        # Rule-based score (0-1 based on severity and count)
        if rule_triggers:
            severity_scores = {
                "low": 0.25,
                "medium": 0.5,
                "high": 0.75,
                "critical": 1.0
            }
            
            rule_score = max(
                severity_scores.get(t.severity.value, 0)
                for t in rule_triggers
            )
            # Boost score if multiple rules triggered
            rule_score = min(1.0, rule_score + 0.1 * (len(rule_triggers) - 1))
        else:
            rule_score = 0.0
            
        # ML-based score
        if self.ml_detector and self.ml_detector.is_fitted:
            ml_score, _ = self.ml_detector.predict(features)
            # Normalize ML score to 0-1 range (typical IF scores are -0.5 to 0.5)
            ml_score = min(1.0, max(0.0, (ml_score + 0.5)))
        else:
            ml_score = 0.0
            self.rule_weight = 1.0
            self.ml_weight = 0.0
            
        # Combine scores
        combined = (self.rule_weight * rule_score + 
                   self.ml_weight * ml_score)
        
        # Determine confidence level
        if rule_triggers and ml_score > 0.5:
            confidence = "high"
        elif rule_triggers or ml_score > 0.5:
            confidence = "medium"
        elif ml_score > 0.3:
            confidence = "low"
        else:
            confidence = "normal"
            
        return combined, confidence
