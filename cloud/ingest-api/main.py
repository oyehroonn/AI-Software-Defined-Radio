"""FastAPI ingest API for AeroSentry AI."""

import asyncio
import json
import logging
from datetime import datetime, UTC
from typing import Optional
from contextlib import asynccontextmanager
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connection managers
class ConnectionManager:
    """WebSocket connection manager for real-time alerts."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

# Pydantic models
class ADSBMessage(BaseModel):
    """ADS-B message from edge sensor."""
    sensor_id: str
    timestamp: datetime
    icao24: str
    df: Optional[int] = None
    raw: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[int] = None
    velocity: Optional[int] = None
    heading: Optional[float] = None
    vert_rate: Optional[int] = None
    callsign: Optional[str] = None
    squawk: Optional[str] = None
    signal_level: Optional[float] = None

class ADSBBatch(BaseModel):
    """Batch of ADS-B messages."""
    messages: list[ADSBMessage]

class AnomalyAlert(BaseModel):
    """Anomaly alert from detection system."""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    sensor_id: str
    icao24: str
    callsign: Optional[str] = None
    alert_type: str
    severity: str
    anomaly_score: float
    rule_triggers: list[dict] = []
    evidence: dict = {}
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class TrackFeatures(BaseModel):
    """Track features for analysis."""
    icao24: str
    callsign: Optional[str] = None
    window_start: Optional[float] = None
    window_end: Optional[float] = None
    velocity_mean: Optional[float] = None
    velocity_std: Optional[float] = None
    altitude_mean: Optional[float] = None
    altitude_std: Optional[float] = None
    max_turn_rate: Optional[float] = None
    max_implied_speed: Optional[float] = None
    msg_rate: Optional[float] = None
    msg_count: Optional[int] = None

class SensorRegistration(BaseModel):
    """Sensor registration request."""
    sensor_id: str
    name: Optional[str] = None
    latitude: float
    longitude: float
    altitude_m: Optional[float] = None
    config: Optional[dict] = None

class QueryRequest(BaseModel):
    """Natural language query request."""
    query: str

# In-memory storage (replace with actual DB in production)
messages_store: list[dict] = []
alerts_store: list[dict] = []
sensors_store: dict[str, dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("AeroSentry Ingest API starting up")
    yield
    logger.info("AeroSentry Ingest API shutting down")

app = FastAPI(
    title="AeroSentry AI Ingest API",
    description="Real-time ADS-B and anomaly data ingestion",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "active_connections": len(manager.active_connections)
    }

# Sensor registration
@app.post("/sensors/register")
async def register_sensor(sensor: SensorRegistration):
    """Register or update a sensor."""
    sensors_store[sensor.sensor_id] = {
        **sensor.model_dump(),
        "last_seen": datetime.now(UTC).isoformat(),
        "status": "active"
    }
    logger.info(f"Sensor registered: {sensor.sensor_id}")
    return {"status": "registered", "sensor_id": sensor.sensor_id}

@app.get("/sensors")
async def list_sensors():
    """List all registered sensors."""
    return {"sensors": list(sensors_store.values())}

# ADS-B message ingestion
@app.post("/ingest/adsb")
async def ingest_adsb(message: ADSBMessage):
    """Ingest a single ADS-B message."""
    msg_dict = message.model_dump()
    msg_dict["timestamp"] = msg_dict["timestamp"].isoformat()
    messages_store.append(msg_dict)
    
    # Update sensor last seen
    if message.sensor_id in sensors_store:
        sensors_store[message.sensor_id]["last_seen"] = datetime.now(UTC).isoformat()
    
    # Keep only last 10000 messages in memory
    if len(messages_store) > 10000:
        messages_store.pop(0)
        
    return {"status": "ingested"}

@app.post("/ingest/adsb/batch")
async def ingest_adsb_batch(batch: ADSBBatch):
    """Ingest a batch of ADS-B messages."""
    for message in batch.messages:
        msg_dict = message.model_dump()
        msg_dict["timestamp"] = msg_dict["timestamp"].isoformat()
        messages_store.append(msg_dict)
        
    # Update sensor last seen
    if batch.messages:
        sensor_id = batch.messages[0].sensor_id
        if sensor_id in sensors_store:
            sensors_store[sensor_id]["last_seen"] = datetime.now(UTC).isoformat()
    
    # Keep only last 10000 messages
    while len(messages_store) > 10000:
        messages_store.pop(0)
        
    return {"status": "ingested", "count": len(batch.messages)}

# Anomaly alerts
@app.post("/alerts")
async def create_alert(alert: AnomalyAlert):
    """Create a new anomaly alert."""
    alert_dict = alert.model_dump()
    alert_dict["timestamp"] = alert_dict["timestamp"].isoformat()
    alerts_store.append(alert_dict)
    
    # Broadcast to WebSocket clients
    await manager.broadcast({
        "type": "alert",
        "data": alert_dict
    })
    
    logger.info(f"Alert created: {alert.alert_id} - {alert.alert_type} ({alert.severity})")
    return {"status": "created", "alert_id": alert.alert_id}

@app.get("/alerts")
async def list_alerts(
    severity: Optional[str] = None,
    icao24: Optional[str] = None,
    limit: int = 100
):
    """List recent alerts with optional filters."""
    filtered = alerts_store.copy()
    
    if severity:
        filtered = [a for a in filtered if a.get("severity") == severity]
    if icao24:
        filtered = [a for a in filtered if a.get("icao24") == icao24]
        
    return {"alerts": filtered[-limit:]}

@app.get("/alerts/{alert_id}")
async def get_alert(alert_id: str):
    """Get a specific alert by ID."""
    for alert in alerts_store:
        if alert.get("alert_id") == alert_id:
            return alert
    raise HTTPException(status_code=404, detail="Alert not found")

# Track data
@app.get("/tracks")
async def list_tracks(limit: int = 100):
    """Get current active tracks."""
    # Aggregate recent messages by icao24
    tracks = {}
    for msg in reversed(messages_store[-1000:]):
        icao24 = msg.get("icao24")
        if icao24 and icao24 not in tracks:
            tracks[icao24] = {
                "icao24": icao24,
                "callsign": msg.get("callsign"),
                "latitude": msg.get("latitude"),
                "longitude": msg.get("longitude"),
                "altitude": msg.get("altitude"),
                "velocity": msg.get("velocity"),
                "heading": msg.get("heading"),
                "last_seen": msg.get("timestamp")
            }
            if len(tracks) >= limit:
                break
                
    return {"tracks": list(tracks.values())}

@app.get("/tracks/{icao24}")
async def get_track(icao24: str, limit: int = 100):
    """Get track history for a specific aircraft."""
    track_messages = [
        m for m in messages_store 
        if m.get("icao24") == icao24
    ][-limit:]
    
    return {
        "icao24": icao24,
        "message_count": len(track_messages),
        "history": track_messages
    }

# WebSocket for real-time updates
@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alert streaming."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await websocket.send_json({"type": "pong", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/tracks")
async def websocket_tracks(websocket: WebSocket):
    """WebSocket endpoint for real-time track updates."""
    await websocket.accept()
    try:
        last_count = 0
        while True:
            await asyncio.sleep(1)
            # Send track updates
            if len(messages_store) != last_count:
                tracks = {}
                for msg in reversed(messages_store[-100:]):
                    icao24 = msg.get("icao24")
                    if icao24 and icao24 not in tracks:
                        tracks[icao24] = msg
                        
                await websocket.send_json({
                    "type": "tracks",
                    "data": list(tracks.values())
                })
                last_count = len(messages_store)
    except WebSocketDisconnect:
        pass

# Statistics
@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    unique_aircraft = set(m.get("icao24") for m in messages_store)
    unique_sensors = set(m.get("sensor_id") for m in messages_store)
    
    severity_counts = {}
    for alert in alerts_store:
        sev = alert.get("severity", "unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
    return {
        "messages_in_memory": len(messages_store),
        "alerts_count": len(alerts_store),
        "unique_aircraft": len(unique_aircraft),
        "active_sensors": len(unique_sensors),
        "registered_sensors": len(sensors_store),
        "alert_severity_breakdown": severity_counts
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
