"""Rule-based anomaly detection for ADS-B tracks."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RuleDefinition:
    """Definition of an anomaly detection rule."""
    rule_id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: Callable[[dict], bool]
    evidence_keys: list[str]


@dataclass
class RuleTrigger:
    """Result of a triggered rule."""
    rule_id: str
    name: str
    severity: AlertSeverity
    description: str
    evidence: dict


# Rule definitions
ANOMALY_RULES: list[RuleDefinition] = [
    RuleDefinition(
        rule_id="impossible_speed",
        name="Impossible Speed",
        description="Velocity exceeds physically possible limits (>Mach 1.5)",
        severity=AlertSeverity.HIGH,
        condition=lambda f: f.get("velocity_mean", 0) > 1200,
        evidence_keys=["velocity_mean", "velocity_max"]
    ),
    
    RuleDefinition(
        rule_id="teleport_detected",
        name="Teleportation Detected",
        description="Position jump implies impossible movement (implied speed >2000 knots)",
        severity=AlertSeverity.CRITICAL,
        condition=lambda f: f.get("max_implied_speed", 0) > 2000,
        evidence_keys=["max_implied_speed", "position_gap_count"]
    ),
    
    RuleDefinition(
        rule_id="extreme_climb",
        name="Extreme Climb Rate",
        description="Climb/descent rate exceeds aircraft performance limits (>15000 ft/min)",
        severity=AlertSeverity.MEDIUM,
        condition=lambda f: abs(f.get("max_climb_rate", 0)) > 15000 or 
                           abs(f.get("max_descent_rate", 0)) > 15000,
        evidence_keys=["max_climb_rate", "max_descent_rate"]
    ),
    
    RuleDefinition(
        rule_id="impossible_turn",
        name="Impossible Turn Rate",
        description="Turn rate exceeds aircraft structural limits (>10 deg/sec)",
        severity=AlertSeverity.HIGH,
        condition=lambda f: f.get("max_turn_rate", 0) > 10,
        evidence_keys=["max_turn_rate", "heading_std"]
    ),
    
    RuleDefinition(
        rule_id="sparse_track",
        name="Sparse Track",
        description="Unusually sparse message rate (<0.1 msg/sec)",
        severity=AlertSeverity.LOW,
        condition=lambda f: 0 < f.get("msg_rate", 1) < 0.1,
        evidence_keys=["msg_rate", "msg_count", "track_duration"]
    ),
    
    RuleDefinition(
        rule_id="speed_inconsistency",
        name="Speed Inconsistency",
        description="Large discrepancy between reported and implied speed (>200 knots)",
        severity=AlertSeverity.MEDIUM,
        condition=lambda f: f.get("max_speed_discrepancy", 0) > 200,
        evidence_keys=["max_speed_discrepancy", "speed_consistency_error"]
    ),
    
    RuleDefinition(
        rule_id="erratic_altitude",
        name="Erratic Altitude",
        description="Excessive altitude variation in short window (>5000 ft std dev)",
        severity=AlertSeverity.MEDIUM,
        condition=lambda f: f.get("altitude_std", 0) > 5000,
        evidence_keys=["altitude_std", "altitude_range", "altitude_min", "altitude_max"]
    ),
    
    RuleDefinition(
        rule_id="ground_clutter",
        name="Potential Ground Clutter",
        description="Low altitude with no movement - may be ground reflection",
        severity=AlertSeverity.LOW,
        condition=lambda f: (
            f.get("altitude_mean", 10000) < 500 and 
            f.get("velocity_mean", 100) < 10 and
            f.get("altitude_std", 100) < 50
        ),
        evidence_keys=["altitude_mean", "velocity_mean", "altitude_std"]
    ),
    
    RuleDefinition(
        rule_id="message_burst",
        name="Message Burst",
        description="Abnormally high message rate (>5 msg/sec) - potential flooding",
        severity=AlertSeverity.MEDIUM,
        condition=lambda f: f.get("msg_rate", 0) > 5,
        evidence_keys=["msg_rate", "msg_count"]
    ),
    
    RuleDefinition(
        rule_id="extreme_acceleration",
        name="Extreme Acceleration",
        description="Acceleration exceeds possible limits (>50 knots/sec)",
        severity=AlertSeverity.HIGH,
        condition=lambda f: f.get("max_acceleration", 0) > 50,
        evidence_keys=["max_acceleration", "velocity_max_delta"]
    ),
    
    RuleDefinition(
        rule_id="vertical_rate_mismatch",
        name="Vertical Rate Mismatch",
        description="Reported vertical rate inconsistent with altitude changes",
        severity=AlertSeverity.MEDIUM,
        condition=lambda f: (
            abs(f.get("vertical_rate_mean", 0)) < 100 and
            abs(f.get("max_climb_rate", 0)) > 3000
        ),
        evidence_keys=["vertical_rate_mean", "max_climb_rate", "max_descent_rate"]
    ),
]


class RuleEngine:
    """Engine for evaluating anomaly detection rules."""
    
    def __init__(self, rules: Optional[list[RuleDefinition]] = None):
        self.rules = rules or ANOMALY_RULES
        self.enabled_rules: set[str] = {r.rule_id for r in self.rules}
        
    def enable_rule(self, rule_id: str):
        """Enable a specific rule."""
        self.enabled_rules.add(rule_id)
        
    def disable_rule(self, rule_id: str):
        """Disable a specific rule."""
        self.enabled_rules.discard(rule_id)
        
    def evaluate(self, features: dict) -> list[RuleTrigger]:
        """Evaluate all enabled rules against features."""
        triggered = []
        
        for rule in self.rules:
            if rule.rule_id not in self.enabled_rules:
                continue
                
            try:
                if rule.condition(features):
                    evidence = {
                        key: features.get(key)
                        for key in rule.evidence_keys
                        if key in features
                    }
                    
                    triggered.append(RuleTrigger(
                        rule_id=rule.rule_id,
                        name=rule.name,
                        severity=rule.severity,
                        description=rule.description,
                        evidence=evidence
                    ))
                    
            except Exception as e:
                logger.warning(f"Rule {rule.rule_id} evaluation failed: {e}")
                
        return triggered
        
    def evaluate_batch(self, features_list: list[dict]) -> dict[str, list[RuleTrigger]]:
        """Evaluate rules for multiple tracks."""
        results = {}
        
        for features in features_list:
            icao24 = features.get("icao24")
            if icao24:
                triggers = self.evaluate(features)
                if triggers:
                    results[icao24] = triggers
                    
        return results


def get_max_severity(triggers: list[RuleTrigger]) -> Optional[AlertSeverity]:
    """Get the maximum severity from a list of triggers."""
    if not triggers:
        return None
        
    severity_order = [
        AlertSeverity.LOW,
        AlertSeverity.MEDIUM,
        AlertSeverity.HIGH,
        AlertSeverity.CRITICAL
    ]
    
    max_idx = -1
    for trigger in triggers:
        idx = severity_order.index(trigger.severity)
        max_idx = max(max_idx, idx)
        
    return severity_order[max_idx] if max_idx >= 0 else None
