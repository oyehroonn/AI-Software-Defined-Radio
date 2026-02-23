"""Abstract data source protocol for ADS-B data providers."""

from typing import Protocol, AsyncGenerator, runtime_checkable


@runtime_checkable
class DataSourceProtocol(Protocol):
    """Protocol defining the interface for ADS-B data sources.
    
    Any data source (Beast TCP, OpenSky API, file replay, etc.) should
    implement this interface to be usable with the EdgeNode.
    
    The stream() method should yield dictionaries with the following keys:
        - timestamp: ISO format timestamp string
        - icao24: ICAO24 hex address (lowercase)
        - callsign: Aircraft callsign (optional)
        - latitude: WGS-84 latitude (optional)
        - longitude: WGS-84 longitude (optional)
        - altitude: Altitude in feet (optional)
        - velocity: Ground speed in knots (optional)
        - heading: Track angle in degrees (optional)
        - vert_rate: Vertical rate in ft/min (optional)
        - squawk: Transponder code (optional)
        - signal_level: Signal strength (optional, source-specific)
    """
    
    async def connect(self) -> None:
        """Establish connection to the data source.
        
        Should initialize any necessary connections, sessions, or resources.
        May be called multiple times (should be idempotent).
        """
        ...
        
    async def disconnect(self) -> None:
        """Close connection and release resources.
        
        Should gracefully close any open connections or sessions.
        May be called multiple times (should be idempotent).
        """
        ...
        
    async def stream(self) -> AsyncGenerator[dict, None]:
        """Stream aircraft state messages from the data source.
        
        Yields:
            dict: Aircraft state message with standard fields
            
        This is the main data retrieval method. Implementations should:
        - Automatically call connect() if not already connected
        - Handle reconnection on transient errors
        - Yield messages as they become available
        - Respect any rate limits of the underlying source
        """
        ...


class DataSourceInfo:
    """Metadata about a data source."""
    
    def __init__(
        self,
        name: str,
        source_type: str,
        description: str = "",
        supports_signal_level: bool = False,
        supports_raw_messages: bool = False,
        is_real_time: bool = True
    ):
        self.name = name
        self.source_type = source_type
        self.description = description
        self.supports_signal_level = supports_signal_level
        self.supports_raw_messages = supports_raw_messages
        self.is_real_time = is_real_time


# Pre-defined source info for known sources
BEAST_SOURCE_INFO = DataSourceInfo(
    name="Beast TCP",
    source_type="beast",
    description="Local SDR via Beast binary protocol",
    supports_signal_level=True,
    supports_raw_messages=True,
    is_real_time=True
)

OPENSKY_SOURCE_INFO = DataSourceInfo(
    name="OpenSky Network",
    source_type="opensky",
    description="OpenSky Network REST API",
    supports_signal_level=False,
    supports_raw_messages=False,
    is_real_time=True
)


def validate_message(msg: dict) -> bool:
    """Validate that a message has required fields.
    
    Args:
        msg: Message dictionary to validate
        
    Returns:
        True if message has minimum required fields
    """
    required = ["icao24"]
    return all(key in msg and msg[key] is not None for key in required)


def normalize_message(msg: dict) -> dict:
    """Normalize message fields to standard format.
    
    Args:
        msg: Raw message dictionary
        
    Returns:
        Normalized message with consistent field types
    """
    normalized = {
        "timestamp": msg.get("timestamp"),
        "icao24": msg.get("icao24", "").lower() if msg.get("icao24") else None,
        "callsign": msg.get("callsign", "").strip() if msg.get("callsign") else None,
        "latitude": float(msg["latitude"]) if msg.get("latitude") is not None else None,
        "longitude": float(msg["longitude"]) if msg.get("longitude") is not None else None,
        "altitude": int(msg["altitude"]) if msg.get("altitude") is not None else None,
        "velocity": int(msg["velocity"]) if msg.get("velocity") is not None else None,
        "heading": float(msg["heading"]) if msg.get("heading") is not None else None,
        "vert_rate": int(msg["vert_rate"]) if msg.get("vert_rate") is not None else None,
        "squawk": msg.get("squawk"),
        "signal_level": float(msg["signal_level"]) if msg.get("signal_level") is not None else None,
    }
    
    # Remove None values for cleaner output
    return {k: v for k, v in normalized.items() if v is not None}
