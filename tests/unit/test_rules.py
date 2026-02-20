"""Unit tests for anomaly detection rules."""

import pytest
import sys
sys.path.insert(0, '.')

from edge.edge_inference.rules import RuleEngine, AlertSeverity


class TestRuleEngine:
    """Tests for RuleEngine."""
    
    def setup_method(self):
        self.engine = RuleEngine()
        
    def test_impossible_speed_triggers(self):
        """Test that impossible speed rule triggers correctly."""
        features = {"velocity_mean": 1500, "icao24": "TEST01"}
        triggers = self.engine.evaluate(features)
        
        rule_ids = [t.rule_id for t in triggers]
        assert "impossible_speed" in rule_ids
        
    def test_impossible_speed_no_trigger(self):
        """Test that normal speed doesn't trigger."""
        features = {"velocity_mean": 450, "icao24": "TEST01"}
        triggers = self.engine.evaluate(features)
        
        rule_ids = [t.rule_id for t in triggers]
        assert "impossible_speed" not in rule_ids
        
    def test_teleport_detection(self):
        """Test teleportation detection."""
        features = {"max_implied_speed": 3000, "icao24": "TEST01"}
        triggers = self.engine.evaluate(features)
        
        rule_ids = [t.rule_id for t in triggers]
        assert "teleport_detected" in rule_ids
        
        # Check severity
        teleport_trigger = next(t for t in triggers if t.rule_id == "teleport_detected")
        assert teleport_trigger.severity == AlertSeverity.CRITICAL
        
    def test_extreme_climb(self):
        """Test extreme climb rate detection."""
        features = {"max_climb_rate": 20000, "icao24": "TEST01"}
        triggers = self.engine.evaluate(features)
        
        rule_ids = [t.rule_id for t in triggers]
        assert "extreme_climb" in rule_ids
        
    def test_extreme_descent(self):
        """Test extreme descent rate detection."""
        features = {"max_descent_rate": -18000, "icao24": "TEST01"}
        triggers = self.engine.evaluate(features)
        
        rule_ids = [t.rule_id for t in triggers]
        assert "extreme_climb" in rule_ids
        
    def test_impossible_turn(self):
        """Test impossible turn rate detection."""
        features = {"max_turn_rate": 15, "icao24": "TEST01"}
        triggers = self.engine.evaluate(features)
        
        rule_ids = [t.rule_id for t in triggers]
        assert "impossible_turn" in rule_ids
        
    def test_message_burst(self):
        """Test message burst detection."""
        features = {"msg_rate": 10, "icao24": "TEST01"}
        triggers = self.engine.evaluate(features)
        
        rule_ids = [t.rule_id for t in triggers]
        assert "message_burst" in rule_ids
        
    def test_sparse_track(self):
        """Test sparse track detection."""
        features = {"msg_rate": 0.05, "icao24": "TEST01"}
        triggers = self.engine.evaluate(features)
        
        rule_ids = [t.rule_id for t in triggers]
        assert "sparse_track" in rule_ids
        
    def test_normal_track_no_triggers(self):
        """Test that normal track doesn't trigger any rules."""
        features = {
            "velocity_mean": 450,
            "max_implied_speed": 500,
            "max_climb_rate": 2000,
            "max_descent_rate": -2000,
            "max_turn_rate": 3,
            "msg_rate": 1.5,
            "icao24": "TEST01"
        }
        triggers = self.engine.evaluate(features)
        
        assert len(triggers) == 0
        
    def test_multiple_triggers(self):
        """Test multiple rules triggering simultaneously."""
        features = {
            "velocity_mean": 1500,
            "max_implied_speed": 3000,
            "max_turn_rate": 15,
            "icao24": "TEST01"
        }
        triggers = self.engine.evaluate(features)
        
        rule_ids = [t.rule_id for t in triggers]
        assert "impossible_speed" in rule_ids
        assert "teleport_detected" in rule_ids
        assert "impossible_turn" in rule_ids
        
    def test_disable_rule(self):
        """Test disabling a rule."""
        self.engine.disable_rule("impossible_speed")
        
        features = {"velocity_mean": 1500, "icao24": "TEST01"}
        triggers = self.engine.evaluate(features)
        
        rule_ids = [t.rule_id for t in triggers]
        assert "impossible_speed" not in rule_ids
        
    def test_evidence_extraction(self):
        """Test that evidence is correctly extracted."""
        features = {
            "velocity_mean": 1500,
            "velocity_max": 1600,
            "icao24": "TEST01"
        }
        triggers = self.engine.evaluate(features)
        
        speed_trigger = next(t for t in triggers if t.rule_id == "impossible_speed")
        assert "velocity_mean" in speed_trigger.evidence
        assert speed_trigger.evidence["velocity_mean"] == 1500


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""
    
    def test_severity_values(self):
        """Test severity level values."""
        assert AlertSeverity.LOW.value == "low"
        assert AlertSeverity.MEDIUM.value == "medium"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.CRITICAL.value == "critical"
