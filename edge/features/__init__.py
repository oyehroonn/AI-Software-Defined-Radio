"""Feature extraction modules for AeroSentry AI."""

from .beast_ingest import BeastClient, decode_adsb_message, DecodedMessage
from .track_features import TrackWindow, TrackManager
from .phy_features import PHYFeatures, extract_phy_features, PHYFeatureExtractor
from .data_source import DataSourceProtocol, DataSourceInfo, validate_message, normalize_message
from .opensky_client import OpenSkyLiveClient, create_opensky_client, parse_bbox_string

__all__ = [
    # Beast/SDR
    "BeastClient",
    "decode_adsb_message",
    "DecodedMessage",
    # OpenSky
    "OpenSkyLiveClient",
    "create_opensky_client",
    "parse_bbox_string",
    # Data Source Protocol
    "DataSourceProtocol",
    "DataSourceInfo",
    "validate_message",
    "normalize_message",
    # Track Features
    "TrackWindow",
    "TrackManager",
    # PHY Features
    "PHYFeatures",
    "extract_phy_features",
    "PHYFeatureExtractor",
]
