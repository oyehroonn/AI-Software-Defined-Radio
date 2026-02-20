"""Pytest configuration and fixtures for AeroSentry AI tests."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from datetime import datetime, UTC


@pytest.fixture
def sample_adsb_message():
    """Create a sample ADS-B message."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "sensor_id": "test-sensor-001",
        "icao24": "abc123",
        "callsign": "TEST123",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "altitude": 35000,
        "velocity": 450,
        "heading": 270.5,
        "vert_rate": 0,
        "squawk": "1200",
        "signal_level": -85.0
    }


@pytest.fixture
def sample_track_features():
    """Create sample track features."""
    return {
        "avg_velocity": 400,
        "max_velocity": 450,
        "min_velocity": 350,
        "velocity_std": 30,
        "avg_altitude": 35000,
        "max_altitude": 37000,
        "min_altitude": 33000,
        "altitude_std": 1000,
        "avg_climb_rate": 0,
        "max_climb_rate": 2000,
        "avg_turn_rate": 0,
        "max_turn_rate": 2,
        "total_distance": 50,
        "max_jump": 1,
        "position_variance": 0.1,
        "message_rate": 1,
        "rate_variance": 0.1,
        "track_duration": 300,
        "point_count": 100,
    }


@pytest.fixture
def sample_phy_features():
    """Create sample PHY-layer features."""
    return {
        "cfo": 150.5,
        "amplitude_mean": 0.8,
        "amplitude_std": 0.05,
        "preamble_correlation": 0.95,
        "phase_mean": 0.1,
        "phase_std": 0.02,
        "rise_time": 0.5e-6,
        "overshoot": 0.02,
        "ringing": 0.01,
        "spectral_entropy": 0.7,
        "bandwidth": 2e6,
    }


@pytest.fixture
def sample_alert():
    """Create a sample anomaly alert."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "sensor_id": "test-sensor-001",
        "icao24": "abc123",
        "alert_type": "impossible_speed",
        "severity": "high",
        "confidence": 0.95,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "details": {
            "observed_velocity": 1500,
            "threshold": 900
        }
    }


@pytest.fixture
def sample_voice_transcript():
    """Create a sample voice transcript."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "sensor_id": "test-sensor-001",
        "frequency": 118.7,
        "text": "Delta 123 cleared for takeoff runway 28L",
        "confidence": 0.92,
        "duration_seconds": 3.5,
        "entities": {
            "callsigns": ["Delta 123"],
            "runways": ["28L"],
            "clearances": ["takeoff"]
        }
    }
