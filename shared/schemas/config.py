"""Configuration schemas for AeroSentry AI components."""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SDRDeviceType(Enum):
    """Supported SDR device types."""
    RTLSDR = "rtlsdr"
    AIRSPY = "airspy"
    SDRPLAY = "sdrplay"
    HACKRF = "hackrf"


@dataclass
class SensorConfig:
    """Configuration for an edge sensor node."""
    sensor_id: str
    latitude: float
    longitude: float
    altitude_m: float
    
    # SDR configuration
    sdr_device_type: SDRDeviceType = SDRDeviceType.RTLSDR
    sdr_device_index: int = 0
    sdr_gain: float = 40.0
    
    # ADS-B decoder settings
    adsb_enabled: bool = True
    adsb_sample_rate: int = 2_000_000
    adsb_center_freq: int = 1_090_000_000
    
    # VHF airband settings
    vhf_enabled: bool = False
    vhf_frequencies: list[float] = field(default_factory=list)
    
    # IQ capture settings
    iq_capture_enabled: bool = False
    iq_sample_rate: int = 2_000_000
    iq_burst_window_us: int = 120
    
    # Cloud connectivity
    cloud_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    
    # Offline mode
    offline_mode: bool = False
    local_storage_path: str = "/var/lib/aerosentry"


@dataclass
class CloudConfig:
    """Configuration for cloud services."""
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    grpc_port: int = 50051
    
    # Database
    timescaledb_host: str = "localhost"
    timescaledb_port: int = 5432
    timescaledb_database: str = "aerosentry"
    timescaledb_user: str = "aerosentry"
    timescaledb_password: str = ""
    
    # Streaming
    nats_url: str = "nats://localhost:4222"
    nats_stream: str = "aerosentry"
    
    # Object storage
    s3_endpoint: str = "http://localhost:9000"
    s3_bucket: str = "aerosentry"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    
    # Model serving
    model_serving_url: str = "http://localhost:8001"
    
    # LLM configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_api_key: Optional[str] = None


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""
    # Rule-based detection
    rules_enabled: bool = True
    
    # Kinematic limits
    max_velocity_knots: float = 1200  # ~Mach 1.5
    max_implied_speed_knots: float = 2000
    max_climb_rate_fpm: float = 15000
    max_turn_rate_deg_sec: float = 10
    min_message_rate: float = 0.1
    
    # ML detection
    ml_enabled: bool = True
    isolation_forest_contamination: float = 0.01
    anomaly_score_threshold: float = 0.5
    
    # PHY detection
    phy_detection_enabled: bool = False
    phy_model_path: Optional[str] = None
    phy_consistency_threshold: float = 0.7
    
    # Alert settings
    alert_cooldown_seconds: int = 60
    multi_sensor_correlation: bool = True


@dataclass
class RetentionConfig:
    """Data retention configuration."""
    adsb_messages_days: int = 30
    voice_recordings_hours: int = 24
    voice_transcripts_days: int = 30
    anomaly_alerts_days: int = 365
    phy_features_days: int = 30


@dataclass 
class MeshtasticConfig:
    """Meshtastic mesh network configuration."""
    enabled: bool = False
    serial_port: str = "/dev/ttyUSB0"
    channel: int = 0
    alert_prefix: str = "!SKY:"


class DataSourceType(Enum):
    """Supported data source types."""
    BEAST = "beast"      # Local SDR via Beast TCP protocol
    OPENSKY = "opensky"  # OpenSky Network REST API


@dataclass
class OpenSkyConfig:
    """OpenSky Network API configuration.
    
    OpenSky provides free access to global ADS-B data via REST API.
    Registration at https://opensky-network.org/ increases rate limits.
    
    Rate Limits:
        - Anonymous: 10 requests per 10 seconds
        - Authenticated (free): 40 requests per 10 seconds
    """
    enabled: bool = False
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Polling configuration
    poll_interval_seconds: float = 10.0
    
    # Bounding box to filter aircraft (reduces data volume)
    bbox_lat_min: Optional[float] = None
    bbox_lat_max: Optional[float] = None
    bbox_lon_min: Optional[float] = None
    bbox_lon_max: Optional[float] = None
    
    def get_bbox(self) -> Optional[tuple[float, float, float, float]]:
        """Get bounding box as tuple if all values are set."""
        if all([
            self.bbox_lat_min is not None,
            self.bbox_lat_max is not None,
            self.bbox_lon_min is not None,
            self.bbox_lon_max is not None
        ]):
            return (
                self.bbox_lat_min,
                self.bbox_lat_max,
                self.bbox_lon_min,
                self.bbox_lon_max
            )
        return None


@dataclass
class BeastConfig:
    """Beast TCP data source configuration.
    
    For connecting to local SDR via readsb/dump1090 Beast protocol.
    """
    host: str = "localhost"
    port: int = 30005


@dataclass
class DataSourceConfig:
    """Combined data source configuration."""
    source_type: DataSourceType = DataSourceType.BEAST
    beast: BeastConfig = field(default_factory=BeastConfig)
    opensky: OpenSkyConfig = field(default_factory=OpenSkyConfig)
