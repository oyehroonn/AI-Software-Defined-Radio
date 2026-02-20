"""Local-first storage for offline operation."""

import sqlite3
import json
import logging
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class OfflineStore:
    """Local-first storage with sync capability."""
    
    def __init__(self, db_path: str = "/var/lib/aerosentry/local.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        self._init_schema()
        
    def _init_schema(self):
        """Initialize database schema."""
        self.conn.executescript("""
            -- Pending sync queue
            CREATE TABLE IF NOT EXISTS pending_sync (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                record_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                synced INTEGER DEFAULT 0,
                sync_attempts INTEGER DEFAULT 0,
                last_error TEXT
            );
            
            -- Local alerts
            CREATE TABLE IF NOT EXISTS local_alerts (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                icao24 TEXT NOT NULL,
                callsign TEXT,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                anomaly_score REAL,
                evidence TEXT,
                latitude REAL,
                longitude REAL,
                synced INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Local aircraft states
            CREATE TABLE IF NOT EXISTS local_aircraft (
                icao24 TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                callsign TEXT,
                latitude REAL,
                longitude REAL,
                altitude INTEGER,
                velocity INTEGER,
                heading REAL,
                vert_rate INTEGER,
                squawk TEXT,
                signal_level REAL,
                PRIMARY KEY (icao24, timestamp)
            );
            
            -- Airspace summaries
            CREATE TABLE IF NOT EXISTS airspace_summaries (
                timestamp TIMESTAMP PRIMARY KEY,
                aircraft_count INTEGER,
                emergency_squawks TEXT,
                anomaly_count INTEGER,
                summary_json TEXT
            );
            
            -- Mesh messages log
            CREATE TABLE IF NOT EXISTS mesh_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                direction TEXT NOT NULL,
                from_id TEXT,
                to_id TEXT,
                text TEXT NOT NULL,
                is_alert INTEGER DEFAULT 0
            );
            
            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON local_alerts(timestamp);
            CREATE INDEX IF NOT EXISTS idx_alerts_synced ON local_alerts(synced);
            CREATE INDEX IF NOT EXISTS idx_aircraft_timestamp ON local_aircraft(timestamp);
            CREATE INDEX IF NOT EXISTS idx_pending_synced ON pending_sync(synced);
        """)
        self.conn.commit()
        
    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        try:
            yield self.conn
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
            
    def store_alert(self, alert: dict):
        """Store alert locally."""
        with self.transaction():
            self.conn.execute("""
                INSERT OR REPLACE INTO local_alerts 
                (id, timestamp, icao24, callsign, alert_type, severity, 
                 anomaly_score, evidence, latitude, longitude)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.get("alert_id", str(id(alert))),
                alert.get("timestamp", datetime.now(UTC).isoformat()),
                alert.get("icao24", ""),
                alert.get("callsign"),
                alert.get("alert_type", "unknown"),
                alert.get("severity", "unknown"),
                alert.get("anomaly_score"),
                json.dumps(alert.get("evidence", {})),
                alert.get("latitude"),
                alert.get("longitude")
            ))
            
        # Queue for sync
        self.store_for_sync("anomaly_alerts", alert)
        
    def store_aircraft(self, aircraft: dict):
        """Store aircraft state locally."""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO local_aircraft
                (icao24, timestamp, callsign, latitude, longitude,
                 altitude, velocity, heading, vert_rate, squawk, signal_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                aircraft.get("icao24", ""),
                aircraft.get("timestamp", datetime.now(UTC).isoformat()),
                aircraft.get("callsign"),
                aircraft.get("latitude"),
                aircraft.get("longitude"),
                aircraft.get("altitude"),
                aircraft.get("velocity"),
                aircraft.get("heading"),
                aircraft.get("vert_rate"),
                aircraft.get("squawk"),
                aircraft.get("signal_level")
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to store aircraft: {e}")
            
    def store_summary(self, summary: dict):
        """Store airspace summary."""
        with self.transaction():
            self.conn.execute("""
                INSERT OR REPLACE INTO airspace_summaries
                (timestamp, aircraft_count, emergency_squawks, 
                 anomaly_count, summary_json)
                VALUES (?, ?, ?, ?, ?)
            """, (
                summary.get("timestamp", datetime.now(UTC).isoformat()),
                summary.get("aircraft_count", 0),
                json.dumps(summary.get("emergency_squawks", [])),
                summary.get("anomaly_count", 0),
                json.dumps(summary)
            ))
            
    def store_mesh_message(
        self,
        text: str,
        direction: str,
        from_id: str = "",
        to_id: str = "",
        is_alert: bool = False
    ):
        """Log mesh message."""
        with self.transaction():
            self.conn.execute("""
                INSERT INTO mesh_messages
                (timestamp, direction, from_id, to_id, text, is_alert)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(UTC).isoformat(),
                direction,
                from_id,
                to_id,
                text,
                1 if is_alert else 0
            ))
            
    def store_for_sync(self, table: str, record: dict):
        """Store record for later sync to cloud."""
        with self.transaction():
            self.conn.execute("""
                INSERT INTO pending_sync (table_name, record_json)
                VALUES (?, ?)
            """, (table, json.dumps(record, default=str)))
            
    def get_recent_aircraft(
        self,
        minutes: int = 30,
        limit: int = 1000
    ) -> list[dict]:
        """Get recent aircraft states."""
        cutoff = datetime.now(UTC).isoformat()
        
        cursor = self.conn.execute("""
            SELECT * FROM local_aircraft
            WHERE timestamp > datetime('now', ?)
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"-{minutes} minutes", limit))
        
        return [dict(row) for row in cursor.fetchall()]
        
    def get_recent_alerts(
        self,
        minutes: int = 60,
        limit: int = 100
    ) -> list[dict]:
        """Get recent alerts."""
        cursor = self.conn.execute("""
            SELECT * FROM local_alerts
            WHERE timestamp > datetime('now', ?)
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"-{minutes} minutes", limit))
        
        results = []
        for row in cursor.fetchall():
            alert = dict(row)
            if alert.get("evidence"):
                alert["evidence"] = json.loads(alert["evidence"])
            results.append(alert)
            
        return results
        
    def get_pending_sync(self, limit: int = 100) -> list[dict]:
        """Get records pending sync."""
        cursor = self.conn.execute("""
            SELECT id, table_name, record_json 
            FROM pending_sync 
            WHERE synced = 0
            ORDER BY created_at
            LIMIT ?
        """, (limit,))
        
        return [
            {
                "id": row["id"],
                "table": row["table_name"],
                "record": json.loads(row["record_json"])
            }
            for row in cursor.fetchall()
        ]
        
    def mark_synced(self, sync_ids: list[int]):
        """Mark records as synced."""
        if not sync_ids:
            return
            
        placeholders = ",".join("?" * len(sync_ids))
        with self.transaction():
            self.conn.execute(f"""
                UPDATE pending_sync 
                SET synced = 1 
                WHERE id IN ({placeholders})
            """, sync_ids)
            
    def mark_sync_failed(self, sync_id: int, error: str):
        """Mark sync attempt as failed."""
        with self.transaction():
            self.conn.execute("""
                UPDATE pending_sync 
                SET sync_attempts = sync_attempts + 1,
                    last_error = ?
                WHERE id = ?
            """, (error, sync_id))
            
    def cleanup_old_data(self, days: int = 7):
        """Remove old data to save space."""
        with self.transaction():
            # Clean aircraft data
            self.conn.execute("""
                DELETE FROM local_aircraft
                WHERE timestamp < datetime('now', ?)
            """, (f"-{days} days",))
            
            # Clean synced pending records
            self.conn.execute("""
                DELETE FROM pending_sync
                WHERE synced = 1 AND created_at < datetime('now', '-1 day')
            """)
            
            # Clean old mesh messages
            self.conn.execute("""
                DELETE FROM mesh_messages
                WHERE timestamp < datetime('now', ?)
            """, (f"-{days} days",))
            
    def get_stats(self) -> dict:
        """Get storage statistics."""
        stats = {}
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM local_aircraft")
        stats["aircraft_records"] = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM local_alerts")
        stats["alert_records"] = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM pending_sync WHERE synced = 0")
        stats["pending_sync"] = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM mesh_messages")
        stats["mesh_messages"] = cursor.fetchone()[0]
        
        # Database file size
        if self.db_path.exists():
            stats["db_size_mb"] = self.db_path.stat().st_size / (1024 * 1024)
        else:
            stats["db_size_mb"] = 0
            
        return stats


class SyncManager:
    """Manages synchronization with cloud when connectivity is available."""
    
    def __init__(
        self,
        local_store: OfflineStore,
        remote_client: Optional[Any] = None
    ):
        self.store = local_store
        self.remote = remote_client
        
    async def sync_pending(self, batch_size: int = 50) -> dict:
        """Sync pending records to cloud."""
        if self.remote is None:
            return {"status": "no_remote", "synced": 0}
            
        pending = self.store.get_pending_sync(batch_size)
        
        if not pending:
            return {"status": "nothing_to_sync", "synced": 0}
            
        synced = 0
        failed = 0
        
        for item in pending:
            try:
                await self.remote.insert(item["table"], item["record"])
                self.store.mark_synced([item["id"]])
                synced += 1
            except Exception as e:
                self.store.mark_sync_failed(item["id"], str(e))
                failed += 1
                
        return {
            "status": "complete",
            "synced": synced,
            "failed": failed,
            "remaining": len(pending) - synced - failed
        }
