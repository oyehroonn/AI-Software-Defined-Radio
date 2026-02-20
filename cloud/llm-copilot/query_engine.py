"""Natural language query engine for RF Copilot."""

import logging
import re
from typing import Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# SQL template patterns for common queries
SQL_TEMPLATES = {
    "recent_aircraft": """
        SELECT DISTINCT icao24, MAX(callsign) as callsign, 
               MAX(latitude) as lat, MAX(longitude) as lon,
               MAX(altitude) as altitude, MAX(velocity) as velocity,
               MAX(time) as last_seen
        FROM adsb_messages 
        WHERE time > NOW() - INTERVAL '{interval}'
        GROUP BY icao24
        ORDER BY last_seen DESC
        LIMIT {limit}
    """,
    
    "track_history": """
        SELECT time, icao24, callsign, latitude, longitude, 
               altitude, velocity, heading, vert_rate
        FROM adsb_messages
        WHERE icao24 = '{icao24}'
          AND time > NOW() - INTERVAL '{interval}'
        ORDER BY time DESC
        LIMIT {limit}
    """,
    
    "recent_alerts": """
        SELECT time, alert_id, icao24, callsign, alert_type, 
               severity, anomaly_score, evidence
        FROM anomaly_alerts
        WHERE time > NOW() - INTERVAL '{interval}'
        ORDER BY time DESC
        LIMIT {limit}
    """,
    
    "alerts_by_severity": """
        SELECT time, alert_id, icao24, callsign, alert_type, 
               severity, anomaly_score
        FROM anomaly_alerts
        WHERE severity = '{severity}'
          AND time > NOW() - INTERVAL '{interval}'
        ORDER BY time DESC
        LIMIT {limit}
    """,
    
    "aircraft_alerts": """
        SELECT time, alert_id, alert_type, severity, 
               anomaly_score, evidence
        FROM anomaly_alerts
        WHERE icao24 = '{icao24}'
        ORDER BY time DESC
        LIMIT {limit}
    """,
    
    "message_stats": """
        SELECT time_bucket('1 hour', time) as hour,
               COUNT(*) as msg_count,
               COUNT(DISTINCT icao24) as aircraft_count,
               AVG(signal_level) as avg_signal
        FROM adsb_messages
        WHERE time > NOW() - INTERVAL '{interval}'
        GROUP BY hour
        ORDER BY hour DESC
    """,
    
    "emergency_squawks": """
        SELECT time, icao24, callsign, squawk, latitude, longitude, altitude
        FROM adsb_messages
        WHERE squawk IN ('7500', '7600', '7700')
          AND time > NOW() - INTERVAL '{interval}'
        ORDER BY time DESC
    """,
    
    "anomaly_summary": """
        SELECT alert_type, severity, COUNT(*) as count,
               AVG(anomaly_score) as avg_score
        FROM anomaly_alerts
        WHERE time > NOW() - INTERVAL '{interval}'
        GROUP BY alert_type, severity
        ORDER BY count DESC
    """
}

# Query pattern matching
QUERY_PATTERNS = [
    (r"(?:show|list|get)\s+(?:recent\s+)?aircraft", "recent_aircraft"),
    (r"track(?:ing)?\s+(?:history\s+)?(?:for\s+)?([A-F0-9]{6})", "track_history"),
    (r"(?:recent\s+)?alerts", "recent_alerts"),
    (r"(critical|high|medium|low)\s+(?:severity\s+)?alerts", "alerts_by_severity"),
    (r"alerts?\s+(?:for\s+)?([A-F0-9]{6})", "aircraft_alerts"),
    (r"(?:message\s+)?stats|statistics", "message_stats"),
    (r"emergenc(?:y|ies)|squawk\s+7[567]00", "emergency_squawks"),
    (r"anomaly\s+summary|detection\s+summary", "anomaly_summary"),
]


@dataclass
class QueryResult:
    """Result of a query execution."""
    query: str
    sql: str
    results: list[dict]
    row_count: int
    error: Optional[str] = None


class QueryEngine:
    """Engine for converting natural language to SQL queries."""
    
    def __init__(
        self,
        db_connection: Optional[Any] = None,
        default_interval: str = "1 hour",
        default_limit: int = 100
    ):
        self.db = db_connection
        self.default_interval = default_interval
        self.default_limit = default_limit
        
    def parse_query(self, natural_query: str) -> tuple[str, dict]:
        """
        Parse natural language query into template and parameters.
        
        Returns:
            (template_name, parameters)
        """
        query_lower = natural_query.lower().strip()
        
        # Extract time intervals from query
        interval = self.default_interval
        interval_match = re.search(
            r"(?:last|past)\s+(\d+)\s+(hour|minute|day|week)s?",
            query_lower
        )
        if interval_match:
            num, unit = interval_match.groups()
            interval = f"{num} {unit}s"
            
        # Extract limit
        limit = self.default_limit
        limit_match = re.search(r"(?:top|limit)\s+(\d+)", query_lower)
        if limit_match:
            limit = int(limit_match.group(1))
            
        # Match query pattern
        for pattern, template_name in QUERY_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                params = {
                    "interval": interval,
                    "limit": limit
                }
                
                # Extract icao24 if present
                if match.groups():
                    group = match.group(1)
                    if re.match(r"[A-Fa-f0-9]{6}", group):
                        params["icao24"] = group.upper()
                    elif group in ["critical", "high", "medium", "low"]:
                        params["severity"] = group
                        
                return template_name, params
                
        # Default to recent aircraft
        return "recent_aircraft", {"interval": interval, "limit": limit}
        
    def generate_sql(self, template_name: str, params: dict) -> str:
        """Generate SQL from template and parameters."""
        if template_name not in SQL_TEMPLATES:
            raise ValueError(f"Unknown query template: {template_name}")
            
        template = SQL_TEMPLATES[template_name]
        
        # Sanitize parameters
        safe_params = {}
        for key, value in params.items():
            if isinstance(value, str):
                # Basic SQL injection prevention
                value = re.sub(r"[;'\"\-\-]", "", value)
            safe_params[key] = value
            
        return template.format(**safe_params).strip()
        
    def validate_sql(self, sql: str) -> bool:
        """Validate SQL for safety."""
        sql_lower = sql.lower()
        
        # Block dangerous operations
        dangerous = ["drop", "delete", "truncate", "update", "insert", "alter", "create"]
        for keyword in dangerous:
            if keyword in sql_lower:
                logger.warning(f"Blocked dangerous SQL keyword: {keyword}")
                return False
                
        # Must be a SELECT
        if not sql_lower.strip().startswith("select"):
            return False
            
        return True
        
    async def execute_query(self, natural_query: str) -> QueryResult:
        """
        Convert natural language to SQL and execute.
        
        Args:
            natural_query: Natural language query string
            
        Returns:
            QueryResult with SQL and results
        """
        try:
            template_name, params = self.parse_query(natural_query)
            sql = self.generate_sql(template_name, params)
            
            if not self.validate_sql(sql):
                return QueryResult(
                    query=natural_query,
                    sql=sql,
                    results=[],
                    row_count=0,
                    error="Query validation failed"
                )
                
            # Execute if database connection available
            if self.db:
                results = await self.db.fetch(sql)
                return QueryResult(
                    query=natural_query,
                    sql=sql,
                    results=[dict(r) for r in results],
                    row_count=len(results)
                )
            else:
                # Return SQL only (for testing)
                return QueryResult(
                    query=natural_query,
                    sql=sql,
                    results=[],
                    row_count=0,
                    error="No database connection"
                )
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return QueryResult(
                query=natural_query,
                sql="",
                results=[],
                row_count=0,
                error=str(e)
            )
            
    def get_available_queries(self) -> list[dict]:
        """Return list of supported query types."""
        return [
            {"name": "recent_aircraft", "description": "List recently seen aircraft"},
            {"name": "track_history", "description": "Get track history for specific aircraft"},
            {"name": "recent_alerts", "description": "List recent anomaly alerts"},
            {"name": "alerts_by_severity", "description": "Get alerts filtered by severity"},
            {"name": "aircraft_alerts", "description": "Get alerts for specific aircraft"},
            {"name": "message_stats", "description": "Get message statistics over time"},
            {"name": "emergency_squawks", "description": "Find emergency squawk codes"},
            {"name": "anomaly_summary", "description": "Summarize detected anomalies"},
        ]
