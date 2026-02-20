"""Synthetic attack scenario generation for testing and evaluation."""

import logging
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def inject_spoofed_track(
    base_track: pd.DataFrame,
    icao24_fake: str,
    offset_lat: float = 0.1,
    offset_lon: float = 0.1,
    callsign_fake: Optional[str] = None
) -> pd.DataFrame:
    """
    Create synthetic spoofed track by cloning and offsetting real track.
    
    Args:
        base_track: DataFrame with legitimate track data
        icao24_fake: Fake ICAO24 address to use
        offset_lat: Latitude offset in degrees
        offset_lon: Longitude offset in degrees
        callsign_fake: Optional fake callsign
        
    Returns:
        DataFrame with spoofed track labeled
    """
    spoofed = base_track.copy()
    spoofed["icao24"] = icao24_fake
    
    if "latitude" in spoofed.columns:
        spoofed["latitude"] = spoofed["latitude"] + offset_lat
    if "longitude" in spoofed.columns:
        spoofed["longitude"] = spoofed["longitude"] + offset_lon
        
    if callsign_fake:
        spoofed["callsign"] = callsign_fake
        
    spoofed["label"] = "spoofed"
    spoofed["attack_type"] = "spoof_clone"
    
    logger.info(f"Generated spoofed track with {len(spoofed)} points")
    return spoofed


def inject_replay_attack(
    track: pd.DataFrame,
    delay_seconds: int = 300,
    icao24_new: Optional[str] = None
) -> pd.DataFrame:
    """
    Simulate replay attack by time-shifting track.
    
    Args:
        track: Original track data
        delay_seconds: Time delay for replay
        icao24_new: Optional new ICAO24 (if attacker changes it)
        
    Returns:
        DataFrame with replayed track
    """
    replayed = track.copy()
    
    if "time" in replayed.columns:
        replayed["time"] = replayed["time"] + timedelta(seconds=delay_seconds)
    if "timestamp" in replayed.columns:
        replayed["timestamp"] = replayed["timestamp"] + timedelta(seconds=delay_seconds)
        
    if icao24_new:
        replayed["icao24"] = icao24_new
        
    replayed["label"] = "replay"
    replayed["attack_type"] = "replay"
    replayed["replay_delay_sec"] = delay_seconds
    
    logger.info(f"Generated replay attack with {delay_seconds}s delay")
    return replayed


def inject_ghost_aircraft(
    region_bounds: tuple[float, float, float, float],
    start_time: datetime,
    duration_seconds: int = 60,
    msg_count: int = 10,
    icao24: str = "GHOST1"
) -> pd.DataFrame:
    """
    Generate brief, inconsistent ghost track.
    
    Args:
        region_bounds: (min_lat, min_lon, max_lat, max_lon)
        start_time: Start time for ghost track
        duration_seconds: How long the ghost appears
        msg_count: Number of messages
        icao24: ICAO24 for ghost aircraft
        
    Returns:
        DataFrame with ghost aircraft track
    """
    min_lat, min_lon, max_lat, max_lon = region_bounds
    
    np.random.seed()
    
    records = []
    base_lat = np.random.uniform(min_lat, max_lat)
    base_lon = np.random.uniform(min_lon, max_lon)
    
    for i in range(msg_count):
        time_offset = np.random.uniform(0, duration_seconds)
        
        # Add kinematic inconsistencies (impossible jumps)
        lat = base_lat + np.random.normal(0, 0.5)  # Large position variance
        lon = base_lon + np.random.normal(0, 0.5)
        
        records.append({
            "time": start_time + timedelta(seconds=time_offset),
            "icao24": icao24,
            "callsign": f"GHOST{np.random.randint(100, 999)}",
            "latitude": lat,
            "longitude": lon,
            "altitude": np.random.randint(5000, 45000),
            "velocity": np.random.randint(100, 800),
            "heading": np.random.uniform(0, 360),
            "vert_rate": np.random.randint(-5000, 5000),
            "label": "ghost",
            "attack_type": "ghost_aircraft"
        })
        
    df = pd.DataFrame(records).sort_values("time")
    logger.info(f"Generated ghost aircraft with {len(df)} points")
    return df


def inject_saturation_attack(
    base_time: datetime,
    duration_seconds: int = 10,
    msg_rate: int = 1000,
    icao24: str = "FLOOD1",
    position: tuple[float, float] = (40.0, -74.0)
) -> pd.DataFrame:
    """
    Generate message flood scenario.
    
    Args:
        base_time: Start time for flood
        duration_seconds: Duration of flood
        msg_rate: Messages per second
        icao24: ICAO24 to flood with
        position: Base position for messages
        
    Returns:
        DataFrame with flood messages
    """
    total_msgs = msg_rate * duration_seconds
    
    records = []
    lat, lon = position
    
    for i in range(total_msgs):
        time_offset = i / msg_rate
        
        records.append({
            "time": base_time + timedelta(seconds=time_offset),
            "icao24": icao24,
            "callsign": "FLOOD",
            "latitude": lat + np.random.normal(0, 0.001),
            "longitude": lon + np.random.normal(0, 0.001),
            "altitude": 30000 + np.random.randint(-100, 100),
            "velocity": 450 + np.random.randint(-10, 10),
            "heading": 90,
            "vert_rate": 0,
            "label": "flood",
            "attack_type": "saturation"
        })
        
    df = pd.DataFrame(records)
    logger.info(f"Generated saturation attack with {len(df)} messages")
    return df


def inject_teleportation(
    track: pd.DataFrame,
    teleport_distance_nm: float = 100,
    teleport_index: Optional[int] = None
) -> pd.DataFrame:
    """
    Inject impossible position jump into track.
    
    Args:
        track: Original track
        teleport_distance_nm: Jump distance in nautical miles
        teleport_index: Where to inject (random if None)
        
    Returns:
        Track with teleportation anomaly
    """
    modified = track.copy()
    
    if teleport_index is None:
        teleport_index = len(modified) // 2
        
    if teleport_index >= len(modified):
        teleport_index = len(modified) - 1
        
    # Convert nm to degrees (approximate)
    degree_offset = teleport_distance_nm / 60
    
    if "latitude" in modified.columns:
        modified.loc[modified.index[teleport_index:], "latitude"] += degree_offset
        
    modified["label"] = "legitimate"
    modified.loc[modified.index[teleport_index], "label"] = "teleport"
    modified["attack_type"] = "teleportation"
    
    logger.info(f"Injected teleportation at index {teleport_index}")
    return modified


def inject_velocity_anomaly(
    track: pd.DataFrame,
    impossible_velocity: int = 1500,
    anomaly_duration: int = 5
) -> pd.DataFrame:
    """
    Inject impossible velocity into track.
    
    Args:
        track: Original track
        impossible_velocity: Velocity in knots (> Mach 1.5)
        anomaly_duration: Number of points with anomaly
        
    Returns:
        Track with velocity anomaly
    """
    modified = track.copy()
    
    start_idx = len(modified) // 2
    end_idx = min(start_idx + anomaly_duration, len(modified))
    
    if "velocity" in modified.columns:
        modified.loc[modified.index[start_idx:end_idx], "velocity"] = impossible_velocity
        
    modified["label"] = "legitimate"
    modified.loc[modified.index[start_idx:end_idx], "label"] = "impossible_velocity"
    modified["attack_type"] = "velocity_anomaly"
    
    logger.info(f"Injected impossible velocity ({impossible_velocity} kts)")
    return modified


def create_attack_dataset(
    legitimate_data: pd.DataFrame,
    attack_ratio: float = 0.1
) -> pd.DataFrame:
    """
    Create a mixed dataset with legitimate and attack data.
    
    Args:
        legitimate_data: DataFrame with legitimate tracks
        attack_ratio: Proportion of data to convert to attacks
        
    Returns:
        Combined DataFrame with labels
    """
    legitimate_data = legitimate_data.copy()
    legitimate_data["label"] = "legitimate"
    legitimate_data["attack_type"] = "none"
    
    # Group by icao24 to get individual tracks
    tracks = [group for _, group in legitimate_data.groupby("icao24")]
    
    num_attacks = int(len(tracks) * attack_ratio)
    attack_tracks = np.random.choice(tracks, size=min(num_attacks, len(tracks)), replace=False)
    
    attack_data = []
    
    for i, track in enumerate(attack_tracks):
        attack_type = np.random.choice([
            "spoof", "replay", "ghost", "teleport", "velocity"
        ])
        
        if attack_type == "spoof":
            attack_data.append(inject_spoofed_track(
                track, 
                f"SPOOF{i:04X}",
                offset_lat=np.random.uniform(0.05, 0.2),
                offset_lon=np.random.uniform(0.05, 0.2)
            ))
        elif attack_type == "replay":
            attack_data.append(inject_replay_attack(
                track,
                delay_seconds=np.random.randint(60, 600)
            ))
        elif attack_type == "teleport":
            attack_data.append(inject_teleportation(
                track,
                teleport_distance_nm=np.random.uniform(50, 200)
            ))
        elif attack_type == "velocity":
            attack_data.append(inject_velocity_anomaly(
                track,
                impossible_velocity=np.random.randint(1200, 2000)
            ))
            
    # Add some ghost aircraft
    if len(legitimate_data) > 0 and "latitude" in legitimate_data.columns:
        bounds = (
            legitimate_data["latitude"].min(),
            legitimate_data["longitude"].min(),
            legitimate_data["latitude"].max(),
            legitimate_data["longitude"].max()
        )
        
        for i in range(max(1, num_attacks // 5)):
            start_time = legitimate_data["time"].min() if "time" in legitimate_data.columns else datetime.now()
            attack_data.append(inject_ghost_aircraft(
                bounds,
                start_time + timedelta(seconds=np.random.randint(0, 3600)),
                icao24=f"GHOST{i:03X}"
            ))
            
    # Combine all data
    all_data = [legitimate_data] + attack_data
    combined = pd.concat(all_data, ignore_index=True)
    
    logger.info(f"Created dataset: {len(legitimate_data)} legitimate, "
               f"{len(combined) - len(legitimate_data)} attack samples")
               
    return combined
