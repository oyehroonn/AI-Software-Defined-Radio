"""Integration tests for anomaly detection system."""

import pytest
import numpy as np
from datetime import datetime, timedelta, UTC


class TestRuleBasedDetection:
    """Tests for rule-based anomaly detection."""
    
    @pytest.fixture
    def rule_engine(self):
        """Create rule engine instance."""
        from edge.edge_inference.rules import RuleEngine
        return RuleEngine()
        
    def test_impossible_speed_detection(self, rule_engine):
        """Test detection of impossible speeds."""
        features = {
            "max_velocity": 1500,  # > 900 knots threshold
            "max_climb_rate": 2000,
            "max_turn_rate": 2,
            "max_jump": 0.5,
            "message_rate": 1.0,
            "track_duration": 60,
            "avg_altitude": 35000,
        }
        
        triggers = rule_engine.evaluate(features, "test123")
        
        assert any(t.rule_id == "impossible_speed" for t in triggers)
        
    def test_teleport_detection(self, rule_engine):
        """Test detection of position teleportation."""
        features = {
            "max_velocity": 450,
            "max_climb_rate": 2000,
            "max_turn_rate": 2,
            "max_jump": 15.0,  # > 10 nm threshold
            "message_rate": 1.0,
            "track_duration": 60,
            "avg_altitude": 35000,
        }
        
        triggers = rule_engine.evaluate(features, "test123")
        
        assert any(t.rule_id == "teleport_detected" for t in triggers)
        
    def test_extreme_climb_detection(self, rule_engine):
        """Test detection of extreme climb rates."""
        features = {
            "max_velocity": 450,
            "max_climb_rate": 12000,  # > 10000 ft/min threshold
            "max_turn_rate": 2,
            "max_jump": 0.5,
            "message_rate": 1.0,
            "track_duration": 60,
            "avg_altitude": 35000,
        }
        
        triggers = rule_engine.evaluate(features, "test123")
        
        assert any(t.rule_id == "extreme_climb" for t in triggers)
        
    def test_impossible_turn_detection(self, rule_engine):
        """Test detection of impossible turn rates."""
        features = {
            "max_velocity": 450,
            "max_climb_rate": 2000,
            "max_turn_rate": 8,  # > 5 deg/s threshold
            "max_jump": 0.5,
            "message_rate": 1.0,
            "track_duration": 60,
            "avg_altitude": 35000,
        }
        
        triggers = rule_engine.evaluate(features, "test123")
        
        assert any(t.rule_id == "impossible_turn" for t in triggers)
        
    def test_message_burst_detection(self, rule_engine):
        """Test detection of message bursts."""
        features = {
            "max_velocity": 450,
            "max_climb_rate": 2000,
            "max_turn_rate": 2,
            "max_jump": 0.5,
            "message_rate": 15.0,  # > 10 Hz threshold
            "track_duration": 60,
            "avg_altitude": 35000,
        }
        
        triggers = rule_engine.evaluate(features, "test123")
        
        assert any(t.rule_id == "message_burst" for t in triggers)
        
    def test_impossible_altitude_detection(self, rule_engine):
        """Test detection of impossible altitudes."""
        features = {
            "max_velocity": 450,
            "max_climb_rate": 2000,
            "max_turn_rate": 2,
            "max_jump": 0.5,
            "message_rate": 1.0,
            "track_duration": 60,
            "avg_altitude": 70000,  # > 60000 ft threshold
        }
        
        triggers = rule_engine.evaluate(features, "test123")
        
        assert any(t.rule_id == "impossible_altitude" for t in triggers)
        
    def test_normal_track_no_alerts(self, rule_engine):
        """Test that normal tracks don't trigger alerts."""
        features = {
            "max_velocity": 450,
            "max_climb_rate": 2000,
            "max_turn_rate": 2,
            "max_jump": 0.5,
            "message_rate": 1.0,
            "track_duration": 120,
            "avg_altitude": 35000,
        }
        
        triggers = rule_engine.evaluate(features, "test123")
        
        # Normal track should have no high-severity triggers
        high_severity = [t for t in triggers if t.severity.value >= 3]
        assert len(high_severity) == 0


class TestMLAnomalyDetection:
    """Tests for ML-based anomaly detection."""
    
    @pytest.fixture
    def detector(self):
        """Create anomaly detector instance."""
        from edge.edge_inference.anomaly_model import TrackAnomalyDetector
        return TrackAnomalyDetector(contamination=0.1)
        
    def test_detector_training(self, detector):
        """Test detector can be trained."""
        # Generate training data
        np.random.seed(42)
        n_samples = 100
        
        training_data = []
        for _ in range(n_samples):
            features = {
                "avg_velocity": np.random.normal(400, 50),
                "max_velocity": np.random.normal(450, 50),
                "min_velocity": np.random.normal(350, 50),
                "velocity_std": np.random.normal(30, 10),
                "avg_altitude": np.random.normal(35000, 5000),
                "max_altitude": np.random.normal(37000, 5000),
                "min_altitude": np.random.normal(33000, 5000),
                "altitude_std": np.random.normal(1000, 300),
                "avg_climb_rate": np.random.normal(0, 500),
                "max_climb_rate": np.random.normal(2000, 500),
                "avg_turn_rate": np.random.normal(0, 0.5),
                "max_turn_rate": np.random.normal(2, 0.5),
                "total_distance": np.random.normal(50, 20),
                "max_jump": np.random.normal(1, 0.5),
                "position_variance": np.random.normal(0.1, 0.05),
                "message_rate": np.random.normal(1, 0.2),
                "rate_variance": np.random.normal(0.1, 0.05),
                "track_duration": np.random.normal(300, 100),
                "point_count": np.random.randint(50, 200),
            }
            training_data.append(features)
            
        detector.fit(training_data)
        
        assert detector.is_fitted
        
    def test_anomaly_scoring(self, detector):
        """Test anomaly scoring after training."""
        # Train first
        np.random.seed(42)
        training_data = []
        for _ in range(100):
            features = {
                "avg_velocity": np.random.normal(400, 50),
                "max_velocity": np.random.normal(450, 50),
                "min_velocity": np.random.normal(350, 50),
                "velocity_std": np.random.normal(30, 10),
                "avg_altitude": np.random.normal(35000, 5000),
                "max_altitude": np.random.normal(37000, 5000),
                "min_altitude": np.random.normal(33000, 5000),
                "altitude_std": np.random.normal(1000, 300),
                "avg_climb_rate": np.random.normal(0, 500),
                "max_climb_rate": np.random.normal(2000, 500),
                "avg_turn_rate": np.random.normal(0, 0.5),
                "max_turn_rate": np.random.normal(2, 0.5),
                "total_distance": np.random.normal(50, 20),
                "max_jump": np.random.normal(1, 0.5),
                "position_variance": np.random.normal(0.1, 0.05),
                "message_rate": np.random.normal(1, 0.2),
                "rate_variance": np.random.normal(0.1, 0.05),
                "track_duration": np.random.normal(300, 100),
                "point_count": np.random.randint(50, 200),
            }
            training_data.append(features)
            
        detector.fit(training_data)
        
        # Test normal sample
        normal_features = {
            "avg_velocity": 400,
            "max_velocity": 450,
            "min_velocity": 350,
            "velocity_std": 30,
            "avg_altitude": 35000,
            "max_altitude": 37000,
            "min_altitude": 33000,
            "altitude_std": 1000,
            "avg_climb_rate": 0,
            "max_climb_rate": 2000,
            "avg_turn_rate": 0,
            "max_turn_rate": 2,
            "total_distance": 50,
            "max_jump": 1,
            "position_variance": 0.1,
            "message_rate": 1,
            "rate_variance": 0.1,
            "track_duration": 300,
            "point_count": 100,
        }
        
        normal_score = detector.predict(normal_features)
        
        # Test anomalous sample
        anomaly_features = {
            "avg_velocity": 1500,  # Very high
            "max_velocity": 2000,
            "min_velocity": 1000,
            "velocity_std": 300,
            "avg_altitude": 70000,  # Very high
            "max_altitude": 75000,
            "min_altitude": 65000,
            "altitude_std": 3000,
            "avg_climb_rate": 5000,
            "max_climb_rate": 15000,  # Very high
            "avg_turn_rate": 5,
            "max_turn_rate": 10,  # Very high
            "total_distance": 500,
            "max_jump": 20,  # Very high
            "position_variance": 5,
            "message_rate": 20,  # Very high
            "rate_variance": 5,
            "track_duration": 300,
            "point_count": 100,
        }
        
        anomaly_score = detector.predict(anomaly_features)
        
        # Anomaly should have higher score
        assert anomaly_score > normal_score


class TestEnsembleDetection:
    """Tests for ensemble anomaly detection."""
    
    def test_ensemble_detector_import(self):
        """Test ensemble detector can be imported."""
        from edge.edge_inference.anomaly_model import EnsembleAnomalyDetector
        
    def test_ensemble_combines_scores(self):
        """Test ensemble combines rule and ML scores."""
        from edge.edge_inference.anomaly_model import EnsembleAnomalyDetector
        
        ensemble = EnsembleAnomalyDetector(rule_weight=0.5, ml_weight=0.5)
        
        # Test with mock scores
        combined = ensemble.combine_scores(
            rule_score=0.8,
            ml_score=0.6,
            phy_score=None
        )
        
        # Should be weighted average
        expected = 0.5 * 0.8 + 0.5 * 0.6
        assert abs(combined - expected) < 0.01


class TestSyntheticAttacks:
    """Tests for synthetic attack generation."""
    
    def test_attack_generation_import(self):
        """Test attack generation can be imported."""
        from shared.eval.synthetic_attacks import (
            inject_spoofed_track,
            inject_replay_attack,
            inject_ghost_aircraft,
            create_attack_dataset
        )
        
    def test_spoofed_track_generation(self):
        """Test spoofed track generation."""
        from shared.eval.synthetic_attacks import inject_spoofed_track
        
        # Create base track
        base_track = []
        for i in range(10):
            base_track.append({
                "timestamp": (datetime.now(UTC) + timedelta(seconds=i)).isoformat(),
                "icao24": "abc123",
                "latitude": 37.7749 + i * 0.01,
                "longitude": -122.4194 + i * 0.01,
                "altitude": 35000,
                "velocity": 450,
                "heading": 270,
            })
            
        spoofed = inject_spoofed_track(base_track, deviation_nm=5.0)
        
        # Should have same length
        assert len(spoofed) == len(base_track)
        
        # Should have modified positions
        has_deviation = False
        for orig, spoof in zip(base_track, spoofed):
            if abs(orig["latitude"] - spoof["latitude"]) > 0.01:
                has_deviation = True
                break
                
        assert has_deviation
        
    def test_ghost_aircraft_generation(self):
        """Test ghost aircraft injection."""
        from shared.eval.synthetic_attacks import inject_ghost_aircraft
        
        ghost = inject_ghost_aircraft(
            center_lat=37.7749,
            center_lon=-122.4194,
            duration_seconds=60,
            message_rate=1.0
        )
        
        assert len(ghost) > 0
        assert all("icao24" in msg for msg in ghost)
        
    def test_attack_dataset_creation(self):
        """Test attack dataset creation."""
        from shared.eval.synthetic_attacks import create_attack_dataset
        
        dataset = create_attack_dataset(
            n_normal=10,
            n_spoofed=5,
            n_replay=3,
            n_ghost=2
        )
        
        assert "tracks" in dataset
        assert "labels" in dataset
        assert len(dataset["tracks"]) == len(dataset["labels"])


class TestEvaluationMetrics:
    """Tests for evaluation metrics."""
    
    def test_metrics_import(self):
        """Test metrics can be imported."""
        from shared.eval.metrics import evaluate_detector, EvaluationReport
        
    def test_basic_evaluation(self):
        """Test basic detector evaluation."""
        from shared.eval.metrics import evaluate_detector
        
        # Mock predictions and labels
        predictions = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.15, 0.85]
        labels = [0, 0, 1, 1, 0, 1, 0, 1]
        
        results = evaluate_detector(predictions, labels, threshold=0.5)
        
        assert "precision" in results
        assert "recall" in results
        assert "auc_roc" in results
        assert 0 <= results["precision"] <= 1
        assert 0 <= results["recall"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
