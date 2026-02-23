"""Main entry point for AeroSentry edge node."""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, UTC
from typing import Optional

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("aerosentry.edge")

# Import components
from edge.features.beast_ingest import BeastClient
from edge.features.track_features import TrackManager
from edge.edge_inference.rules import RuleEngine
from edge.edge_inference.anomaly_model import EnsembleAnomalyDetector

# Optional: OpenSky client
try:
    from edge.features.opensky_client import OpenSkyLiveClient, parse_bbox_string
    OPENSKY_AVAILABLE = True
except ImportError:
    OPENSKY_AVAILABLE = False
    logger.info("OpenSky client not available (install aiohttp to enable)")

try:
    from edge.edge_store.parquet_manager import LocalDataStore
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logger.warning("Local storage not available")

try:
    from cloud.stream.nats_config import create_nats_client
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False
    logger.warning("NATS client not available")


class EdgeNode:
    """AeroSentry edge node main application.
    
    Supports multiple data sources:
        - beast: Local SDR via Beast TCP protocol (default)
        - opensky: OpenSky Network REST API (requires free registration)
    
    Set DATA_SOURCE environment variable to select the source.
    """
    
    def __init__(self):
        # Configuration
        self.sensor_id = os.getenv("SENSOR_ID", "edge-001")
        self.sensor_lat = float(os.getenv("SENSOR_LAT", "0"))
        self.sensor_lon = float(os.getenv("SENSOR_LON", "0"))
        
        # Data source configuration
        self.data_source_type = os.getenv("DATA_SOURCE", "beast").lower()
        
        # Beast configuration
        self.beast_host = os.getenv("BEAST_HOST", "localhost")
        self.beast_port = int(os.getenv("BEAST_PORT", "30005"))
        
        # OpenSky configuration (OAuth2)
        self.opensky_client_id = os.getenv("OPENSKY_CLIENT_ID")
        self.opensky_client_secret = os.getenv("OPENSKY_CLIENT_SECRET")
        self.opensky_poll_interval = float(os.getenv("OPENSKY_POLL_INTERVAL", "10"))
        self.opensky_bbox = parse_bbox_string(os.getenv("OPENSKY_BBOX", "")) if OPENSKY_AVAILABLE else None
        
        # Components
        self.data_source = None  # Will be BeastClient or OpenSkyLiveClient
        self.track_manager = TrackManager()
        self.rule_engine = RuleEngine()
        self.ensemble_detector = EnsembleAnomalyDetector()
        self.nats_client = None
        self.local_store = None
        
        # State
        self.running = False
        self.message_count = 0
        self.alert_count = 0
        
    def _create_data_source(self):
        """Create the appropriate data source based on configuration."""
        if self.data_source_type == "opensky":
            if not OPENSKY_AVAILABLE:
                raise RuntimeError(
                    "OpenSky data source requested but aiohttp is not installed. "
                    "Install with: pip install aiohttp"
                )
            
            logger.info("Using OpenSky Network as data source")
            return OpenSkyLiveClient(
                client_id=self.opensky_client_id,
                client_secret=self.opensky_client_secret,
                poll_interval=self.opensky_poll_interval,
                bbox=self.opensky_bbox
            )
            
        elif self.data_source_type == "beast":
            logger.info(f"Using Beast TCP as data source ({self.beast_host}:{self.beast_port})")
            return BeastClient(
                host=self.beast_host,
                port=self.beast_port,
                receiver_lat=self.sensor_lat,
                receiver_lon=self.sensor_lon
            )
            
        else:
            raise ValueError(f"Unknown data source type: {self.data_source_type}")
        
    async def start(self):
        """Start edge node."""
        logger.info(f"Starting AeroSentry edge node: {self.sensor_id}")
        logger.info(f"Data source: {self.data_source_type}")
        
        # Initialize local storage
        if STORAGE_AVAILABLE:
            try:
                self.local_store = LocalDataStore()
                logger.info("Local storage initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize local storage: {e}")
                
        # Connect to NATS
        if NATS_AVAILABLE:
            try:
                nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
                self.nats_client = await create_nats_client(nats_url)
                if self.nats_client:
                    logger.info("Connected to NATS")
            except Exception as e:
                logger.warning(f"Failed to connect to NATS: {e}")
                
        # Create and connect data source
        self.data_source = self._create_data_source()
        
        self.running = True
        
        # Start message processing
        await self._process_messages()
        
    async def stop(self):
        """Stop edge node."""
        logger.info("Stopping edge node...")
        self.running = False
        
        if self.data_source:
            await self.data_source.disconnect()
            
        if self.nats_client:
            await self.nats_client.disconnect()
            
        if self.local_store:
            self.local_store.close()
            
        logger.info(f"Edge node stopped. Processed {self.message_count} messages, {self.alert_count} alerts")
        
    async def _process_messages(self):
        """Main message processing loop."""
        try:
            async for message in self.data_source.stream():
                if not self.running:
                    break
                    
                await self._handle_message(message)
                
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            import traceback
            traceback.print_exc()
            
    async def _handle_message(self, message: dict):
        """Handle a single ADS-B message."""
        self.message_count += 1
        
        # Log progress periodically
        if self.message_count % 100 == 0:
            logger.info(f"Processed {self.message_count} messages, {self.alert_count} alerts")
        
        # Add sensor info
        message["sensor_id"] = self.sensor_id
        message["timestamp"] = message.get("timestamp", datetime.now(UTC).isoformat())
        
        # Store locally
        if self.local_store:
            self.local_store.store_message(message)
            
        # Update track
        icao24 = message.get("icao24")
        if icao24:
            track_window = self.track_manager.update(icao24, message)
            
            # Check for anomalies if we have enough data
            if track_window and track_window.can_compute():
                features = track_window.compute_features()
                
                # Run detection
                await self._detect_anomalies(icao24, features, message)
                
        # Publish to NATS
        if self.nats_client:
            await self.nats_client.publish_adsb(self.sensor_id, message)
            
    async def _detect_anomalies(
        self,
        icao24: str,
        features: dict,
        last_message: dict
    ):
        """Run anomaly detection on track features."""
        
        # Rule-based detection
        triggers = self.rule_engine.evaluate(features, icao24)
        
        # ML detection
        ml_score = self.ensemble_detector.ml_detector.predict(features)
        
        # Combined score
        rule_score = max((t.severity.value / 4.0 for t in triggers), default=0)
        combined_score = self.ensemble_detector.combine_scores(rule_score, ml_score)
        
        # Generate alerts for triggers
        for trigger in triggers:
            if trigger.severity.value >= 2:  # Medium or higher
                alert = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "sensor_id": self.sensor_id,
                    "icao24": icao24,
                    "alert_type": trigger.rule_id,
                    "severity": trigger.severity.name.lower(),
                    "confidence": combined_score,
                    "latitude": last_message.get("latitude"),
                    "longitude": last_message.get("longitude"),
                    "details": trigger.details
                }
                
                await self._handle_alert(alert)
                
    async def _handle_alert(self, alert: dict):
        """Handle generated alert."""
        self.alert_count += 1
        
        logger.warning(
            f"ALERT: {alert['alert_type']} for {alert['icao24']} "
            f"(severity: {alert['severity']}, confidence: {alert['confidence']:.2f})"
        )
        
        # Publish to NATS
        if self.nats_client:
            await self.nats_client.publish_alert(self.sensor_id, alert)


async def main():
    """Main entry point."""
    node = EdgeNode()
    
    # Handle signals
    loop = asyncio.get_event_loop()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(node.stop()))
        
    try:
        await node.start()
    except KeyboardInterrupt:
        await node.stop()


if __name__ == "__main__":
    asyncio.run(main())
