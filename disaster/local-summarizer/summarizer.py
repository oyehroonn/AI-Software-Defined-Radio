"""On-device airspace summarizer for offline/disaster mode."""

import logging
from datetime import datetime, timedelta, UTC
from collections import Counter
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AirspaceSummary:
    """Summary of airspace activity."""
    timestamp: datetime
    window_minutes: int
    aircraft_count: int
    unique_callsigns: list[str]
    emergency_squawks: list[dict]
    alert_summary: dict[str, int]
    anomaly_count: int
    message_rate: float
    coverage_estimate: str
    priority: str
    health: dict


class AirspaceSummarizer:
    """Generate local airspace summaries without cloud connectivity."""
    
    PRIORITY_LEVELS = {
        "critical": 5,  # Active emergency
        "high": 4,      # Multiple anomalies
        "medium": 3,    # Some alerts
        "low": 2,       # Normal operations
        "minimal": 1    # Very little activity
    }
    
    EMERGENCY_SQUAWKS = {
        "7500": "Hijacking",
        "7600": "Radio Failure",
        "7700": "General Emergency"
    }
    
    def __init__(self, local_store=None):
        self.store = local_store
        self.last_summary: Optional[AirspaceSummary] = None
        
    def generate_summary(
        self,
        window_minutes: int = 30,
        aircraft_data: Optional[list[dict]] = None,
        alert_data: Optional[list[dict]] = None
    ) -> AirspaceSummary:
        """
        Generate airspace summary for responders.
        
        Args:
            window_minutes: Time window to summarize
            aircraft_data: List of aircraft state dicts (if not using store)
            alert_data: List of alert dicts (if not using store)
            
        Returns:
            AirspaceSummary
        """
        cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)
        
        # Get data
        if aircraft_data is None:
            aircraft_data = self._get_recent_aircraft(cutoff)
        if alert_data is None:
            alert_data = self._get_recent_alerts(cutoff)
            
        # Count unique aircraft
        unique_icao = set(a.get("icao24", "") for a in aircraft_data if a.get("icao24"))
        
        # Get unique callsigns
        callsigns = list(set(
            a.get("callsign", "") 
            for a in aircraft_data 
            if a.get("callsign")
        ))
        
        # Find emergency squawks
        emergencies = []
        for a in aircraft_data:
            squawk = a.get("squawk", "")
            if squawk in self.EMERGENCY_SQUAWKS:
                emergencies.append({
                    "icao24": a.get("icao24"),
                    "callsign": a.get("callsign"),
                    "squawk": squawk,
                    "meaning": self.EMERGENCY_SQUAWKS[squawk],
                    "position": (a.get("latitude"), a.get("longitude")),
                    "altitude": a.get("altitude")
                })
                
        # Summarize alerts
        alert_types = Counter(a.get("alert_type", "unknown") for a in alert_data)
        
        # Calculate message rate
        if aircraft_data:
            total_time = window_minutes * 60
            message_rate = len(aircraft_data) / total_time
        else:
            message_rate = 0.0
            
        # Estimate coverage
        coverage = self._estimate_coverage(aircraft_data)
        
        # Determine priority
        priority = self._compute_priority(emergencies, alert_data, len(unique_icao))
        
        # Health status
        health = {
            "receiver_active": len(aircraft_data) > 0,
            "last_message_age_sec": self._get_last_message_age(aircraft_data),
            "alert_pipeline_active": True,
            "storage_available": True
        }
        
        summary = AirspaceSummary(
            timestamp=datetime.now(UTC),
            window_minutes=window_minutes,
            aircraft_count=len(unique_icao),
            unique_callsigns=callsigns[:20],  # Limit for mesh transmission
            emergency_squawks=emergencies,
            alert_summary=dict(alert_types),
            anomaly_count=len(alert_data),
            message_rate=message_rate,
            coverage_estimate=coverage,
            priority=priority,
            health=health
        )
        
        self.last_summary = summary
        return summary
        
    def _get_recent_aircraft(self, cutoff: datetime) -> list[dict]:
        """Get recent aircraft from store."""
        if self.store is None:
            return []
        # Implementation depends on local store interface
        return []
        
    def _get_recent_alerts(self, cutoff: datetime) -> list[dict]:
        """Get recent alerts from store."""
        if self.store is None:
            return []
        return []
        
    def _get_last_message_age(self, aircraft_data: list[dict]) -> float:
        """Get age of most recent message in seconds."""
        if not aircraft_data:
            return float('inf')
            
        latest = None
        for a in aircraft_data:
            ts = a.get("timestamp") or a.get("time")
            if ts:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                if latest is None or ts > latest:
                    latest = ts
                    
        if latest is None:
            return float('inf')
            
        age = (datetime.now(UTC) - latest).total_seconds()
        return max(0, age)
        
    def _estimate_coverage(self, aircraft_data: list[dict]) -> str:
        """Estimate coverage quality based on aircraft distribution."""
        if not aircraft_data:
            return "no_data"
            
        # Count aircraft with position data
        with_position = sum(
            1 for a in aircraft_data 
            if a.get("latitude") and a.get("longitude")
        )
        
        position_ratio = with_position / len(aircraft_data)
        
        if position_ratio > 0.8:
            return "excellent"
        elif position_ratio > 0.5:
            return "good"
        elif position_ratio > 0.2:
            return "degraded"
        else:
            return "poor"
            
    def _compute_priority(
        self,
        emergencies: list[dict],
        alerts: list[dict],
        aircraft_count: int
    ) -> str:
        """Compute summary priority for mesh relay."""
        # Critical if active emergency
        if emergencies:
            return "critical"
            
        # High if many alerts
        critical_alerts = sum(
            1 for a in alerts 
            if a.get("severity") in ["critical", "high"]
        )
        
        if critical_alerts >= 3:
            return "high"
        elif critical_alerts >= 1:
            return "medium"
        elif len(alerts) > 0:
            return "low"
        elif aircraft_count > 0:
            return "low"
        else:
            return "minimal"
            
    def to_compact_string(self, summary: Optional[AirspaceSummary] = None) -> str:
        """Convert summary to compact string for mesh transmission."""
        s = summary or self.last_summary
        if s is None:
            return "NO_DATA"
            
        parts = [
            f"T:{s.timestamp.strftime('%H%M')}",
            f"A:{s.aircraft_count}",
            f"P:{s.priority[0].upper()}"
        ]
        
        if s.emergency_squawks:
            for e in s.emergency_squawks[:2]:  # Max 2 emergencies
                parts.append(f"E:{e['squawk']}/{e.get('icao24', '?')[:4]}")
                
        if s.anomaly_count > 0:
            parts.append(f"X:{s.anomaly_count}")
            
        return "|".join(parts)
        
    def to_report(self, summary: Optional[AirspaceSummary] = None) -> str:
        """Generate human-readable report."""
        s = summary or self.last_summary
        if s is None:
            return "No summary available"
            
        lines = [
            "=" * 40,
            "AIRSPACE SUMMARY REPORT",
            "=" * 40,
            f"Time: {s.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Window: {s.window_minutes} minutes",
            f"Priority: {s.priority.upper()}",
            "",
            f"Aircraft Count: {s.aircraft_count}",
            f"Message Rate: {s.message_rate:.2f} msg/sec",
            f"Coverage: {s.coverage_estimate}",
            ""
        ]
        
        if s.emergency_squawks:
            lines.append("*** EMERGENCIES ***")
            for e in s.emergency_squawks:
                lines.append(
                    f"  {e['squawk']} ({e['meaning']}): "
                    f"{e.get('callsign') or e.get('icao24', 'Unknown')}"
                )
            lines.append("")
            
        if s.anomaly_count > 0:
            lines.append(f"Anomaly Alerts: {s.anomaly_count}")
            for alert_type, count in s.alert_summary.items():
                lines.append(f"  {alert_type}: {count}")
            lines.append("")
            
        lines.append("System Health:")
        lines.append(f"  Receiver: {'OK' if s.health['receiver_active'] else 'OFFLINE'}")
        lines.append(f"  Last Msg: {s.health['last_message_age_sec']:.0f}s ago")
        
        lines.append("=" * 40)
        
        return "\n".join(lines)
