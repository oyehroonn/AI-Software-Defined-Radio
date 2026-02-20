"""Local Parquet storage manager for edge nodes."""

import logging
import sqlite3
import json
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logger.warning("PyArrow not available, Parquet storage disabled")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class ParquetPartition:
    """Metadata for a Parquet partition."""
    path: Path
    start_time: datetime
    end_time: datetime
    row_count: int
    size_bytes: int


class ParquetManager:
    """Manages Parquet file storage with time-based partitioning."""
    
    def __init__(
        self,
        base_path: str = "/var/lib/aerosentry/parquet",
        partition_hours: int = 1,
        max_partitions: int = 168  # 7 days at 1 hour each
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.partition_hours = partition_hours
        self.max_partitions = max_partitions
        
        self._buffer: list[dict] = []
        self._buffer_lock = threading.Lock()
        self._buffer_size = 10000
        
        # Schema for ADS-B messages
        if PYARROW_AVAILABLE:
            self.adsb_schema = pa.schema([
                ("timestamp", pa.timestamp("us", tz="UTC")),
                ("sensor_id", pa.string()),
                ("icao24", pa.string()),
                ("callsign", pa.string()),
                ("latitude", pa.float64()),
                ("longitude", pa.float64()),
                ("altitude", pa.int32()),
                ("velocity", pa.int32()),
                ("heading", pa.float32()),
                ("vert_rate", pa.int32()),
                ("squawk", pa.string()),
                ("signal_level", pa.float32()),
            ])
            
    def _get_partition_path(self, timestamp: datetime) -> Path:
        """Get partition path for timestamp."""
        partition_start = timestamp.replace(
            minute=0, second=0, microsecond=0,
            hour=(timestamp.hour // self.partition_hours) * self.partition_hours
        )
        partition_name = partition_start.strftime("%Y%m%d_%H0000")
        return self.base_path / f"adsb_{partition_name}.parquet"
        
    def add_message(self, message: dict):
        """Add message to buffer."""
        with self._buffer_lock:
            self._buffer.append(message)
            
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer()
                
    def _flush_buffer(self):
        """Flush buffer to Parquet files."""
        if not self._buffer or not PYARROW_AVAILABLE:
            return
            
        # Group by partition
        partitions: dict[Path, list[dict]] = {}
        
        for msg in self._buffer:
            ts = msg.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif ts is None:
                ts = datetime.now(UTC)
                
            partition_path = self._get_partition_path(ts)
            
            if partition_path not in partitions:
                partitions[partition_path] = []
            partitions[partition_path].append(msg)
            
        # Write each partition
        for path, messages in partitions.items():
            self._write_partition(path, messages)
            
        self._buffer.clear()
        
        # Cleanup old partitions
        self._cleanup_old_partitions()
        
    def _write_partition(self, path: Path, messages: list[dict]):
        """Write messages to Parquet partition."""
        try:
            # Convert to table
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(messages)
                
                # Ensure correct types
                for col in ["latitude", "longitude"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                for col in ["altitude", "velocity", "vert_rate"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")
                        
                table = pa.Table.from_pandas(df)
            else:
                # Build arrays manually
                arrays = []
                for field in self.adsb_schema:
                    values = [msg.get(field.name) for msg in messages]
                    arrays.append(pa.array(values, type=field.type))
                table = pa.table(dict(zip(self.adsb_schema.names, arrays)))
                
            # Write or append
            if path.exists():
                existing = pq.read_table(path)
                table = pa.concat_tables([existing, table])
                
            pq.write_table(table, path, compression="snappy")
            logger.debug(f"Wrote {len(messages)} messages to {path}")
            
        except Exception as e:
            logger.error(f"Failed to write Parquet: {e}")
            
    def _cleanup_old_partitions(self):
        """Remove partitions older than max_partitions."""
        try:
            partitions = sorted(self.base_path.glob("adsb_*.parquet"))
            
            while len(partitions) > self.max_partitions:
                oldest = partitions.pop(0)
                oldest.unlink()
                logger.info(f"Removed old partition: {oldest}")
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            
    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        icao24: Optional[str] = None
    ) -> Optional['pd.DataFrame']:
        """Query messages from Parquet files."""
        if not PYARROW_AVAILABLE or not PANDAS_AVAILABLE:
            return None
            
        try:
            partitions = sorted(self.base_path.glob("adsb_*.parquet"))
            
            if not partitions:
                return pd.DataFrame()
                
            tables = []
            for path in partitions:
                table = pq.read_table(path)
                tables.append(table)
                
            combined = pa.concat_tables(tables)
            df = combined.to_pandas()
            
            # Apply filters
            if start_time:
                df = df[df["timestamp"] >= start_time]
            if end_time:
                df = df[df["timestamp"] <= end_time]
            if icao24:
                df = df[df["icao24"] == icao24]
                
            return df
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None
            
    def get_stats(self) -> dict:
        """Get storage statistics."""
        partitions = list(self.base_path.glob("adsb_*.parquet"))
        
        total_size = sum(p.stat().st_size for p in partitions)
        
        return {
            "partition_count": len(partitions),
            "total_size_mb": total_size / (1024 * 1024),
            "buffer_size": len(self._buffer),
            "base_path": str(self.base_path)
        }
        
    def flush(self):
        """Force flush buffer."""
        with self._buffer_lock:
            self._flush_buffer()


class LocalDataStore:
    """Combined SQLite + Parquet local storage."""
    
    def __init__(
        self,
        db_path: str = "/var/lib/aerosentry/local.db",
        parquet_path: str = "/var/lib/aerosentry/parquet"
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        self.parquet = ParquetManager(parquet_path) if PYARROW_AVAILABLE else None
        
        self._init_schema()
        
    def _init_schema(self):
        """Initialize SQLite schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS adsb_recent (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                sensor_id TEXT NOT NULL,
                icao24 TEXT NOT NULL,
                callsign TEXT,
                latitude REAL,
                longitude REAL,
                altitude INTEGER,
                velocity INTEGER,
                heading REAL,
                vert_rate INTEGER,
                squawk TEXT,
                signal_level REAL
            );
            
            CREATE INDEX IF NOT EXISTS idx_recent_time ON adsb_recent(timestamp);
            CREATE INDEX IF NOT EXISTS idx_recent_icao ON adsb_recent(icao24);
            
            CREATE TABLE IF NOT EXISTS track_state (
                icao24 TEXT PRIMARY KEY,
                last_update TEXT,
                callsign TEXT,
                latitude REAL,
                longitude REAL,
                altitude INTEGER,
                velocity INTEGER,
                heading REAL,
                squawk TEXT,
                message_count INTEGER DEFAULT 0
            );
        """)
        self.conn.commit()
        
    def store_message(self, message: dict):
        """Store ADS-B message."""
        # Store in SQLite for recent queries
        self.conn.execute("""
            INSERT INTO adsb_recent 
            (timestamp, sensor_id, icao24, callsign, latitude, longitude,
             altitude, velocity, heading, vert_rate, squawk, signal_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message.get("timestamp", datetime.now(UTC).isoformat()),
            message.get("sensor_id", ""),
            message.get("icao24", ""),
            message.get("callsign"),
            message.get("latitude"),
            message.get("longitude"),
            message.get("altitude"),
            message.get("velocity"),
            message.get("heading"),
            message.get("vert_rate"),
            message.get("squawk"),
            message.get("signal_level")
        ))
        
        # Update track state
        self.conn.execute("""
            INSERT OR REPLACE INTO track_state
            (icao24, last_update, callsign, latitude, longitude, altitude,
             velocity, heading, squawk, message_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT message_count FROM track_state WHERE icao24 = ?), 0) + 1)
        """, (
            message.get("icao24"),
            message.get("timestamp", datetime.now(UTC).isoformat()),
            message.get("callsign"),
            message.get("latitude"),
            message.get("longitude"),
            message.get("altitude"),
            message.get("velocity"),
            message.get("heading"),
            message.get("squawk"),
            message.get("icao24")
        ))
        
        self.conn.commit()
        
        # Store in Parquet for long-term
        if self.parquet:
            self.parquet.add_message(message)
            
    def get_active_tracks(self, max_age_seconds: int = 60) -> list[dict]:
        """Get currently active tracks."""
        cutoff = (datetime.now(UTC) - timedelta(seconds=max_age_seconds)).isoformat()
        
        cursor = self.conn.execute("""
            SELECT * FROM track_state WHERE last_update > ?
        """, (cutoff,))
        
        return [dict(row) for row in cursor.fetchall()]
        
    def cleanup(self, max_age_hours: int = 24):
        """Remove old data from SQLite."""
        cutoff = (datetime.now(UTC) - timedelta(hours=max_age_hours)).isoformat()
        
        self.conn.execute("DELETE FROM adsb_recent WHERE timestamp < ?", (cutoff,))
        self.conn.commit()
        
        logger.info(f"Cleaned up data older than {max_age_hours} hours")
        
    def close(self):
        """Close connections and flush data."""
        if self.parquet:
            self.parquet.flush()
        self.conn.close()
