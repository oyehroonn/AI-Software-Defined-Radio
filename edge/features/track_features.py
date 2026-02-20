"""Track feature extraction for anomaly detection."""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import numpy as np


def haversine(pos1: tuple[float, float], pos2: tuple[float, float]) -> float:
    """Calculate great-circle distance between two points in nautical miles."""
    lat1, lon1 = math.radians(pos1[0]), math.radians(pos1[1])
    lat2, lon2 = math.radians(pos2[0]), math.radians(pos2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in nautical miles
    r = 3440.065
    return r * c


@dataclass
class TrackWindow:
    """Sliding window of track data for feature extraction."""
    icao24: str
    timestamps: list[float] = field(default_factory=list)
    positions: list[tuple[float, float]] = field(default_factory=list)
    altitudes: list[int] = field(default_factory=list)
    velocities: list[int] = field(default_factory=list)
    headings: list[float] = field(default_factory=list)
    vertical_rates: list[int] = field(default_factory=list)
    callsign: Optional[str] = None
    
    def add_point(
        self,
        timestamp: float,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        altitude: Optional[int] = None,
        velocity: Optional[int] = None,
        heading: Optional[float] = None,
        vertical_rate: Optional[int] = None,
        callsign: Optional[str] = None
    ):
        """Add a data point to the track window."""
        self.timestamps.append(timestamp)
        
        if latitude is not None and longitude is not None:
            self.positions.append((latitude, longitude))
        if altitude is not None:
            self.altitudes.append(altitude)
        if velocity is not None:
            self.velocities.append(velocity)
        if heading is not None:
            self.headings.append(heading)
        if vertical_rate is not None:
            self.vertical_rates.append(vertical_rate)
        if callsign:
            self.callsign = callsign
            
    def compute_features(self) -> dict:
        """Extract anomaly detection features from track window."""
        features = {
            "icao24": self.icao24,
            "callsign": self.callsign,
            "window_start": min(self.timestamps) if self.timestamps else None,
            "window_end": max(self.timestamps) if self.timestamps else None
        }
        
        if len(self.timestamps) < 2:
            return features
            
        dt = np.diff(self.timestamps)
        dt = np.where(dt > 0, dt, 0.1)  # Avoid division by zero
        
        # Velocity features
        if len(self.velocities) >= 2:
            v = np.array(self.velocities)
            features["velocity_mean"] = float(np.mean(v))
            features["velocity_std"] = float(np.std(v))
            features["velocity_min"] = float(np.min(v))
            features["velocity_max"] = float(np.max(v))
            
            v_delta = np.abs(np.diff(v))
            features["velocity_max_delta"] = float(np.max(v_delta))
            
            # Acceleration (knots per second)
            if len(dt) == len(v) - 1:
                acceleration = v_delta / dt[:len(v_delta)]
                features["max_acceleration"] = float(np.max(acceleration))
                
        # Altitude features
        if len(self.altitudes) >= 2:
            alt = np.array(self.altitudes)
            features["altitude_mean"] = float(np.mean(alt))
            features["altitude_std"] = float(np.std(alt))
            features["altitude_min"] = float(np.min(alt))
            features["altitude_max"] = float(np.max(alt))
            features["altitude_range"] = float(np.max(alt) - np.min(alt))
            
            # Climb/descent rate (ft/min)
            alt_delta = np.diff(alt)
            if len(dt) == len(alt_delta):
                alt_rate = alt_delta / (dt[:len(alt_delta)] / 60)
                features["max_climb_rate"] = float(np.max(alt_rate))
                features["max_descent_rate"] = float(np.min(alt_rate))
                features["climb_rate_std"] = float(np.std(alt_rate))
                
        # Heading features (detect impossible turn rates)
        if len(self.headings) >= 2:
            h = np.array(self.headings)
            
            # Handle heading wraparound
            heading_delta = np.abs(np.diff(h))
            heading_delta = np.minimum(heading_delta, 360 - heading_delta)
            
            features["heading_std"] = float(np.std(h))
            features["max_heading_delta"] = float(np.max(heading_delta))
            
            # Turn rate (degrees per second)
            if len(dt) == len(heading_delta):
                turn_rate = heading_delta / dt[:len(heading_delta)]
                features["max_turn_rate"] = float(np.max(turn_rate))
                features["mean_turn_rate"] = float(np.mean(turn_rate))
                
        # Vertical rate features
        if len(self.vertical_rates) >= 2:
            vr = np.array(self.vertical_rates)
            features["vertical_rate_mean"] = float(np.mean(vr))
            features["vertical_rate_std"] = float(np.std(vr))
            features["vertical_rate_max"] = float(np.max(np.abs(vr)))
            
        # Position continuity features
        if len(self.positions) >= 2:
            distances = []
            for i in range(len(self.positions) - 1):
                dist = haversine(self.positions[i], self.positions[i+1])
                distances.append(dist)
                
            distances = np.array(distances)
            
            # Implied speed from position deltas (nautical miles per second -> knots)
            if len(dt) >= len(distances):
                speeds_implied = distances / dt[:len(distances)] * 3600
                features["max_implied_speed"] = float(np.max(speeds_implied))
                features["mean_implied_speed"] = float(np.mean(speeds_implied))
                
                # Check for position jumps (gaps > 30 seconds)
                features["position_gap_count"] = int(np.sum(dt[:len(distances)] > 30))
                
                # Consistency between reported velocity and implied speed
                if self.velocities and len(self.velocities) > 1:
                    v_reported = np.array(self.velocities[:-1])
                    if len(v_reported) == len(speeds_implied):
                        speed_diff = np.abs(v_reported - speeds_implied)
                        features["speed_consistency_error"] = float(np.mean(speed_diff))
                        features["max_speed_discrepancy"] = float(np.max(speed_diff))
                        
        # Message rate features
        if len(self.timestamps) >= 2:
            total_time = self.timestamps[-1] - self.timestamps[0]
            if total_time > 0:
                features["msg_rate"] = len(self.timestamps) / total_time
            features["msg_count"] = len(self.timestamps)
            features["track_duration"] = total_time
            
            # Inter-message interval statistics
            features["msg_interval_mean"] = float(np.mean(dt))
            features["msg_interval_std"] = float(np.std(dt))
            features["msg_interval_max"] = float(np.max(dt))
            
        return features


class TrackManager:
    """Manages track windows for multiple aircraft."""
    
    def __init__(
        self,
        window_duration: float = 30.0,
        max_gap: float = 60.0
    ):
        self.window_duration = window_duration
        self.max_gap = max_gap
        self.tracks: dict[str, TrackWindow] = {}
        self.last_update: dict[str, float] = {}
        
    def update(
        self,
        icao24: str,
        timestamp: float,
        **kwargs
    ) -> Optional[dict]:
        """Update track and return features if window complete."""
        # Check if track is stale
        if icao24 in self.last_update:
            gap = timestamp - self.last_update[icao24]
            if gap > self.max_gap:
                # Start new track
                if icao24 in self.tracks:
                    del self.tracks[icao24]
                    
        self.last_update[icao24] = timestamp
        
        # Create or get track
        if icao24 not in self.tracks:
            self.tracks[icao24] = TrackWindow(icao24=icao24)
            
        track = self.tracks[icao24]
        track.add_point(timestamp, **kwargs)
        
        # Trim old points outside window
        cutoff = timestamp - self.window_duration
        while track.timestamps and track.timestamps[0] < cutoff:
            track.timestamps.pop(0)
            if track.positions:
                track.positions.pop(0)
            if track.altitudes:
                track.altitudes.pop(0)
            if track.velocities:
                track.velocities.pop(0)
            if track.headings:
                track.headings.pop(0)
            if track.vertical_rates:
                track.vertical_rates.pop(0)
                
        # Check if we have enough data for features
        if len(track.timestamps) >= 3:
            return track.compute_features()
            
        return None
        
    def get_all_features(self) -> list[dict]:
        """Get features for all active tracks."""
        features_list = []
        for icao24, track in self.tracks.items():
            if len(track.timestamps) >= 3:
                features_list.append(track.compute_features())
        return features_list
        
    def cleanup_stale(self, current_time: float):
        """Remove stale tracks."""
        stale = [
            icao24 for icao24, last in self.last_update.items()
            if current_time - last > self.max_gap
        ]
        for icao24 in stale:
            if icao24 in self.tracks:
                del self.tracks[icao24]
            del self.last_update[icao24]
