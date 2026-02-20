"""Web UI server for AeroSentry AI."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

app = FastAPI(title="AeroSentry Web UI")

# Paths
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ConnectionManager:
    """Manages WebSocket connections."""
    
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "web-ui"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main dashboard."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "AeroSentry AI",
            "api_url": "/api"
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                channel = message.get("channel", "all")
                logger.info(f"Client subscribed to {channel}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/stats")
async def get_stats():
    """Get current statistics."""
    return {
        "active_aircraft": 0,
        "alerts_24h": 0,
        "messages_rate": 0,
        "sensors_online": 0
    }


@app.get("/api/aircraft")
async def get_aircraft():
    """Get current aircraft list."""
    return {"aircraft": []}


@app.get("/api/alerts")
async def get_alerts(limit: int = 50):
    """Get recent alerts."""
    return {"alerts": []}


async def push_update(update_type: str, data: dict):
    """Push update to all connected clients."""
    await manager.broadcast({
        "type": update_type,
        "data": data
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
