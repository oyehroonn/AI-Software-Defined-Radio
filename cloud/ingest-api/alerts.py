"""Alert management and pipeline for AeroSentry AI."""

import uuid
import json
import logging
from datetime import datetime, UTC
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional
import asyncio

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of anomaly alerts."""
    SPOOFING_SUSPECTED = "spoofing_suspected"
    REPLAY_ATTACK = "replay_attack"
    KINEMATIC_ANOMALY = "kinematic_anomaly"
    GHOST_AIRCRAFT = "ghost_aircraft"
    MESSAGE_FLOOD = "message_flood"
    RF_INTERFERENCE = "rf_interference"
    TRACK_INCONSISTENCY = "track_inconsistency"
    PHY_ANOMALY = "phy_anomaly"


@dataclass
class AnomalyAlert:
    """Structured anomaly alert."""
    alert_id: str
    timestamp: datetime
    icao24: str
    callsign: Optional[str]
    sensor_id: str
    alert_type: AlertType
    severity: AlertSeverity
    anomaly_score: float
    rule_triggers: list[dict]
    evidence: dict
    position: Optional[tuple[float, float]]
    
    @classmethod
    def create(
        cls,
        icao24: str,
        sensor_id: str,
        alert_type: AlertType,
        severity: AlertSeverity,
        anomaly_score: float,
        rule_triggers: list[dict],
        evidence: dict,
        callsign: Optional[str] = None,
        position: Optional[tuple[float, float]] = None
    ) -> 'AnomalyAlert':
        """Create a new alert with auto-generated ID and timestamp."""
        return cls(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC),
            icao24=icao24,
            callsign=callsign,
            sensor_id=sensor_id,
            alert_type=alert_type,
            severity=severity,
            anomaly_score=anomaly_score,
            rule_triggers=rule_triggers,
            evidence=evidence,
            position=position
        )
        
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "icao24": self.icao24,
            "callsign": self.callsign,
            "sensor_id": self.sensor_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "anomaly_score": self.anomaly_score,
            "rule_triggers": self.rule_triggers,
            "evidence": self.evidence,
            "latitude": self.position[0] if self.position else None,
            "longitude": self.position[1] if self.position else None
        }
        
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AlertCooldown:
    """Manages alert cooldowns to prevent alert fatigue."""
    
    def __init__(self, cooldown_seconds: int = 60):
        self.cooldown_seconds = cooldown_seconds
        self.last_alert: dict[str, datetime] = {}  # key: icao24_alert_type
        
    def should_alert(self, icao24: str, alert_type: AlertType) -> bool:
        """Check if enough time has passed since last alert of this type."""
        key = f"{icao24}_{alert_type.value}"
        now = datetime.now(UTC)
        
        if key in self.last_alert:
            elapsed = (now - self.last_alert[key]).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False
                
        self.last_alert[key] = now
        return True
        
    def cleanup(self, max_age_seconds: int = 3600):
        """Remove old cooldown entries."""
        now = datetime.now(UTC)
        expired = [
            key for key, ts in self.last_alert.items()
            if (now - ts).total_seconds() > max_age_seconds
        ]
        for key in expired:
            del self.last_alert[key]


class AlertPipeline:
    """Pipeline for processing and routing alerts."""
    
    def __init__(
        self,
        cooldown_seconds: int = 60,
        min_severity: AlertSeverity = AlertSeverity.LOW
    ):
        self.cooldown = AlertCooldown(cooldown_seconds)
        self.min_severity = min_severity
        self.handlers: list[callable] = []
        self.alert_history: list[AnomalyAlert] = []
        self.max_history = 1000
        
    def add_handler(self, handler: callable):
        """Add an alert handler function."""
        self.handlers.append(handler)
        
    async def process(self, alert: AnomalyAlert) -> bool:
        """Process an alert through the pipeline."""
        # Check minimum severity
        severity_order = [
            AlertSeverity.LOW,
            AlertSeverity.MEDIUM,
            AlertSeverity.HIGH,
            AlertSeverity.CRITICAL
        ]
        if severity_order.index(alert.severity) < severity_order.index(self.min_severity):
            return False
            
        # Check cooldown
        if not self.cooldown.should_alert(alert.icao24, alert.alert_type):
            logger.debug(f"Alert suppressed by cooldown: {alert.alert_id}")
            return False
            
        # Store in history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
            
        # Call handlers
        for handler in self.handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
                
        logger.info(
            f"Alert processed: {alert.alert_id} - "
            f"{alert.alert_type.value} ({alert.severity.value}) "
            f"for {alert.icao24}"
        )
        return True
        
    def get_recent_alerts(
        self,
        icao24: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> list[AnomalyAlert]:
        """Get recent alerts with optional filters."""
        filtered = self.alert_history.copy()
        
        if icao24:
            filtered = [a for a in filtered if a.icao24 == icao24]
        if alert_type:
            filtered = [a for a in filtered if a.alert_type == alert_type]
        if severity:
            filtered = [a for a in filtered if a.severity == severity]
            
        return filtered[-limit:]


def determine_alert_type(rule_triggers: list) -> AlertType:
    """Determine overall alert type from triggered rules."""
    if not rule_triggers:
        return AlertType.KINEMATIC_ANOMALY
        
    rule_ids = {t.rule_id if hasattr(t, 'rule_id') else t.get('rule_id') for t in rule_triggers}
    
    if "teleport_detected" in rule_ids or "impossible_speed" in rule_ids:
        return AlertType.SPOOFING_SUSPECTED
    elif "message_burst" in rule_ids:
        return AlertType.MESSAGE_FLOOD
    elif "sparse_track" in rule_ids:
        return AlertType.GHOST_AIRCRAFT
    elif "speed_inconsistency" in rule_ids:
        return AlertType.TRACK_INCONSISTENCY
    else:
        return AlertType.KINEMATIC_ANOMALY


def determine_severity(
    rule_triggers: list,
    anomaly_score: float
) -> AlertSeverity:
    """Determine overall severity from rules and ML score."""
    # Get max severity from rules
    max_rule_severity = AlertSeverity.LOW
    
    severity_map = {
        "low": AlertSeverity.LOW,
        "medium": AlertSeverity.MEDIUM,
        "high": AlertSeverity.HIGH,
        "critical": AlertSeverity.CRITICAL
    }
    
    for trigger in rule_triggers:
        sev_str = trigger.severity.value if hasattr(trigger, 'severity') else trigger.get('severity', 'low')
        sev = severity_map.get(sev_str, AlertSeverity.LOW)
        if list(severity_map.values()).index(sev) > list(severity_map.values()).index(max_rule_severity):
            max_rule_severity = sev
            
    # Consider ML score
    if anomaly_score > 0.9:
        ml_severity = AlertSeverity.CRITICAL
    elif anomaly_score > 0.7:
        ml_severity = AlertSeverity.HIGH
    elif anomaly_score > 0.5:
        ml_severity = AlertSeverity.MEDIUM
    else:
        ml_severity = AlertSeverity.LOW
        
    # Return higher of the two
    severities = list(severity_map.values())
    return max(max_rule_severity, ml_severity, key=lambda s: severities.index(s))
