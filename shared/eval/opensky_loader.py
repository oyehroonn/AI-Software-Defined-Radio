"""OpenSky Network data loader for training and evaluation."""

import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import pyopensky
try:
    from pyopensky.trino import Trino
    PYOPENSKY_AVAILABLE = True
except ImportError:
    PYOPENSKY_AVAILABLE = False
    logger.warning("pyopensky not available, using mock data")


class OpenSkyLoader:
    """Load historical data from OpenSky Network."""
    
    def __init__(self):
        if PYOPENSKY_AVAILABLE:
            self.trino = Trino()
        else:
            self.trino = None
            
    def fetch_state_vectors(
        self,
        start: datetime,
        end: datetime,
        bounds: Optional[tuple[float, float, float, float]] = None,
        icao24: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical state vectors from OpenSky.
        
        Args:
            start: Start time
            end: End time
            bounds: (west, south, east, north) bounding box
            icao24: Specific aircraft ICAO24 address
            
        Returns:
            DataFrame with state vectors
        """
        if not PYOPENSKY_AVAILABLE:
            return self._generate_mock_data(start, end)
            
        try:
            df = self.trino.history(
                start=start,
                stop=end,
                bounds=bounds,
                icao24=icao24,
                columns=[
                    "time", "icao24", "callsign", "lat", "lon",
                    "velocity", "heading", "vertrate", "baroaltitude",
                    "geoaltitude", "onground", "squawk"
                ]
            )
            
            # Rename columns to match our schema
            df = df.rename(columns={
                "lat": "latitude",
                "lon": "longitude",
                "baroaltitude": "altitude",
                "vertrate": "vert_rate"
            })
            
            logger.info(f"Fetched {len(df)} state vectors from OpenSky")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch from OpenSky: {e}")
            return pd.DataFrame()
            
    def fetch_raw_messages(
        self,
        start: datetime,
        end: datetime,
        icao24: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch raw ADS-B messages for detailed analysis."""
        if not PYOPENSKY_AVAILABLE:
            return pd.DataFrame()
            
        try:
            return self.trino.rawdata(
                start=start,
                stop=end,
                icao24=icao24
            )
        except Exception as e:
            logger.error(f"Failed to fetch raw data: {e}")
            return pd.DataFrame()
            
    def fetch_flights(
        self,
        start: datetime,
        end: datetime,
        airport: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch flight information."""
        if not PYOPENSKY_AVAILABLE:
            return pd.DataFrame()
            
        try:
            if airport:
                return self.trino.flights_from_airports(
                    start=start,
                    stop=end,
                    airport=airport
                )
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch flights: {e}")
            return pd.DataFrame()
            
    def _generate_mock_data(
        self,
        start: datetime,
        end: datetime,
        num_aircraft: int = 50
    ) -> pd.DataFrame:
        """Generate mock data for testing without OpenSky access."""
        import numpy as np
        
        np.random.seed(42)
        
        records = []
        duration = (end - start).total_seconds()
        
        for i in range(num_aircraft):
            icao24 = f"{i:06X}"
            callsign = f"TEST{i:04d}"
            
            # Random starting position
            lat = np.random.uniform(25, 50)
            lon = np.random.uniform(-125, -70)
            alt = np.random.uniform(10000, 40000)
            vel = np.random.uniform(200, 500)
            hdg = np.random.uniform(0, 360)
            
            # Generate track
            num_points = int(duration / 10)  # One point every 10 seconds
            
            for j in range(num_points):
                time = start + timedelta(seconds=j * 10)
                
                # Add some randomness
                lat += np.random.normal(0, 0.01) + 0.001 * np.sin(hdg * np.pi / 180)
                lon += np.random.normal(0, 0.01) + 0.001 * np.cos(hdg * np.pi / 180)
                alt += np.random.normal(0, 50)
                vel += np.random.normal(0, 5)
                hdg += np.random.normal(0, 1)
                hdg = hdg % 360
                
                records.append({
                    "time": time,
                    "icao24": icao24,
                    "callsign": callsign,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": int(alt),
                    "velocity": int(vel),
                    "heading": hdg,
                    "vert_rate": int(np.random.normal(0, 500)),
                    "onground": False,
                    "squawk": "1200"
                })
                
        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} mock state vectors")
        return df


def fetch_historical_flights(
    start: datetime,
    end: datetime,
    bounds: Optional[tuple[float, float, float, float]] = None
) -> pd.DataFrame:
    """Convenience function to fetch historical flight data."""
    loader = OpenSkyLoader()
    return loader.fetch_state_vectors(start, end, bounds)


def fetch_raw_messages(
    start: datetime,
    end: datetime,
    icao24: Optional[str] = None
) -> pd.DataFrame:
    """Convenience function to fetch raw ADS-B messages."""
    loader = OpenSkyLoader()
    return loader.fetch_raw_messages(start, end, icao24)
