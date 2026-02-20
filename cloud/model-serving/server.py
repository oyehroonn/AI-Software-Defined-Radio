"""Model serving server for AeroSentry AI."""

import asyncio
import json
import logging
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="AeroSentry Model Serving")

# Model path
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models"))


class TrackFeatures(BaseModel):
    """Input features for track anomaly detection."""
    icao24: str
    sensor_id: str
    avg_velocity: float = 0
    max_velocity: float = 0
    min_velocity: float = 0
    velocity_std: float = 0
    avg_altitude: float = 0
    max_altitude: float = 0
    min_altitude: float = 0
    altitude_std: float = 0
    avg_climb_rate: float = 0
    max_climb_rate: float = 0
    avg_turn_rate: float = 0
    max_turn_rate: float = 0
    total_distance: float = 0
    max_jump: float = 0
    position_variance: float = 0
    message_rate: float = 0
    rate_variance: float = 0
    track_duration: float = 0
    point_count: int = 0


class PHYFeatures(BaseModel):
    """Input features for PHY-layer detection."""
    icao24: str
    sensor_id: str
    cfo: float = 0
    amplitude_mean: float = 0
    amplitude_std: float = 0
    preamble_correlation: float = 0
    phase_mean: float = 0
    phase_std: float = 0
    rise_time: float = 0
    overshoot: float = 0
    spectral_entropy: float = 0
    bandwidth: float = 0


class PredictionResponse(BaseModel):
    """Model prediction response."""
    icao24: str
    anomaly_score: float
    is_anomaly: bool
    model_type: str
    confidence: float
    details: dict = {}


# Lazy load models
_track_model = None
_phy_model = None


def get_track_model():
    """Load track anomaly model."""
    global _track_model
    
    if _track_model is None:
        try:
            from edge.edge_inference.anomaly_model import TrackAnomalyDetector
            _track_model = TrackAnomalyDetector()
            
            # Load saved model if exists
            model_file = MODEL_PATH / "track_anomaly.pkl"
            if model_file.exists():
                _track_model.load(str(model_file))
                logger.info("Loaded track anomaly model")
            else:
                logger.warning("No saved track model found, using default")
                
        except Exception as e:
            logger.error(f"Failed to load track model: {e}")
            
    return _track_model


def get_phy_model():
    """Load PHY detector model."""
    global _phy_model
    
    if _phy_model is None:
        try:
            from cloud.model_serving.phy_detector import CalibratedPhyDetector
            _phy_model = CalibratedPhyDetector()
            
            # Load saved model if exists
            model_file = MODEL_PATH / "phy_detector.pt"
            if model_file.exists():
                _phy_model.load(str(model_file))
                logger.info("Loaded PHY detector model")
            else:
                logger.warning("No saved PHY model found, using default")
                
        except Exception as e:
            logger.error(f"Failed to load PHY model: {e}")
            
    return _phy_model


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "model-serving",
        "models": {
            "track": _track_model is not None,
            "phy": _phy_model is not None
        }
    }


@app.post("/predict/track", response_model=PredictionResponse)
async def predict_track(features: TrackFeatures):
    """Predict track anomaly."""
    model = get_track_model()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Track model not available")
        
    try:
        # Convert to feature dict
        feature_dict = {
            "avg_velocity": features.avg_velocity,
            "max_velocity": features.max_velocity,
            "min_velocity": features.min_velocity,
            "velocity_std": features.velocity_std,
            "avg_altitude": features.avg_altitude,
            "max_altitude": features.max_altitude,
            "min_altitude": features.min_altitude,
            "altitude_std": features.altitude_std,
            "avg_climb_rate": features.avg_climb_rate,
            "max_climb_rate": features.max_climb_rate,
            "avg_turn_rate": features.avg_turn_rate,
            "max_turn_rate": features.max_turn_rate,
            "total_distance": features.total_distance,
            "max_jump": features.max_jump,
            "position_variance": features.position_variance,
            "message_rate": features.message_rate,
            "rate_variance": features.rate_variance,
            "track_duration": features.track_duration,
            "point_count": features.point_count,
        }
        
        # Get prediction
        score = model.predict(feature_dict)
        is_anomaly = score > 0.5
        
        return PredictionResponse(
            icao24=features.icao24,
            anomaly_score=score,
            is_anomaly=is_anomaly,
            model_type="isolation_forest",
            confidence=abs(score - 0.5) * 2,
            details={"feature_count": len(feature_dict)}
        )
        
    except Exception as e:
        logger.error(f"Track prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/phy", response_model=PredictionResponse)
async def predict_phy(features: PHYFeatures):
    """Predict PHY-layer spoofing."""
    model = get_phy_model()
    
    if model is None:
        raise HTTPException(status_code=503, detail="PHY model not available")
        
    try:
        # Convert to feature array
        import numpy as np
        feature_array = np.array([[
            features.cfo,
            features.amplitude_mean,
            features.amplitude_std,
            features.preamble_correlation,
            features.phase_mean,
            features.phase_std,
            features.rise_time,
            features.overshoot,
            features.spectral_entropy,
            features.bandwidth
        ]])
        
        # Get prediction
        result = model.predict(feature_array, features.icao24)
        
        return PredictionResponse(
            icao24=features.icao24,
            anomaly_score=result.get("spoof_probability", 0),
            is_anomaly=result.get("is_spoofed", False),
            model_type="phy_dnn",
            confidence=result.get("confidence", 0),
            details={
                "uncertainty": result.get("uncertainty", 0),
                "embedding_distance": result.get("embedding_distance", 0)
            }
        )
        
    except Exception as e:
        logger.error(f"PHY prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models."""
    models = []
    
    for model_file in MODEL_PATH.glob("*"):
        models.append({
            "name": model_file.stem,
            "path": str(model_file),
            "size_bytes": model_file.stat().st_size,
            "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
        })
        
    return {"models": models}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
