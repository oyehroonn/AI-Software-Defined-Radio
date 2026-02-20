"""NATS JetStream configuration and client for AeroSentry AI."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Any
from datetime import timedelta

logger = logging.getLogger(__name__)

try:
    import nats
    from nats.js.api import StreamConfig, ConsumerConfig, AckPolicy, DeliverPolicy
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False
    logger.warning("NATS client not available")


@dataclass
class NATSConfig:
    """NATS connection configuration."""
    url: str = "nats://localhost:4222"
    user: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    
    # Stream settings
    stream_name: str = "aerosentry"
    stream_subjects: list[str] = None
    
    # Retention
    max_age_hours: int = 24
    max_bytes: int = 1_000_000_000  # 1GB
    max_msgs: int = 10_000_000
    
    def __post_init__(self):
        if self.stream_subjects is None:
            self.stream_subjects = [
                "aerosentry.adsb.>",
                "aerosentry.alerts.>",
                "aerosentry.tracks.>",
                "aerosentry.phy.>",
                "aerosentry.voice.>"
            ]


class StreamManager:
    """Manages NATS JetStream streams."""
    
    # Stream definitions
    STREAMS = {
        "aerosentry-adsb": {
            "subjects": ["aerosentry.adsb.>"],
            "description": "ADS-B message stream",
            "max_age": timedelta(hours=24),
            "max_bytes": 500_000_000,
        },
        "aerosentry-alerts": {
            "subjects": ["aerosentry.alerts.>"],
            "description": "Anomaly alerts stream",
            "max_age": timedelta(days=7),
            "max_bytes": 100_000_000,
        },
        "aerosentry-tracks": {
            "subjects": ["aerosentry.tracks.>"],
            "description": "Track state updates",
            "max_age": timedelta(hours=1),
            "max_bytes": 200_000_000,
        },
        "aerosentry-phy": {
            "subjects": ["aerosentry.phy.>"],
            "description": "PHY-layer features",
            "max_age": timedelta(hours=24),
            "max_bytes": 500_000_000,
        },
        "aerosentry-voice": {
            "subjects": ["aerosentry.voice.>"],
            "description": "Voice transcripts",
            "max_age": timedelta(hours=24),
            "max_bytes": 100_000_000,
        }
    }
    
    def __init__(self, js):
        self.js = js
        
    async def setup_streams(self):
        """Create or update all streams."""
        if not NATS_AVAILABLE:
            logger.warning("NATS not available, skipping stream setup")
            return
            
        for stream_name, config in self.STREAMS.items():
            try:
                stream_config = StreamConfig(
                    name=stream_name,
                    subjects=config["subjects"],
                    description=config["description"],
                    max_age=config["max_age"].total_seconds(),
                    max_bytes=config["max_bytes"],
                    storage="file",
                    retention="limits",
                    discard="old"
                )
                
                try:
                    await self.js.add_stream(stream_config)
                    logger.info(f"Created stream: {stream_name}")
                except Exception:
                    await self.js.update_stream(stream_config)
                    logger.info(f"Updated stream: {stream_name}")
                    
            except Exception as e:
                logger.error(f"Failed to setup stream {stream_name}: {e}")
                
    async def get_stream_info(self, stream_name: str) -> Optional[dict]:
        """Get stream information."""
        try:
            info = await self.js.stream_info(stream_name)
            return {
                "name": info.config.name,
                "subjects": info.config.subjects,
                "messages": info.state.messages,
                "bytes": info.state.bytes,
                "first_seq": info.state.first_seq,
                "last_seq": info.state.last_seq
            }
        except Exception as e:
            logger.error(f"Failed to get stream info: {e}")
            return None


class NATSClient:
    """NATS JetStream client for AeroSentry."""
    
    def __init__(self, config: Optional[NATSConfig] = None):
        self.config = config or NATSConfig()
        self.nc = None
        self.js = None
        self.stream_manager = None
        self._subscriptions = []
        
    async def connect(self):
        """Connect to NATS server."""
        if not NATS_AVAILABLE:
            logger.warning("NATS not available")
            return False
            
        try:
            connect_opts = {"servers": [self.config.url]}
            
            if self.config.user and self.config.password:
                connect_opts["user"] = self.config.user
                connect_opts["password"] = self.config.password
            elif self.config.token:
                connect_opts["token"] = self.config.token
                
            self.nc = await nats.connect(**connect_opts)
            self.js = self.nc.jetstream()
            self.stream_manager = StreamManager(self.js)
            
            logger.info(f"Connected to NATS at {self.config.url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            return False
            
    async def disconnect(self):
        """Disconnect from NATS."""
        for sub in self._subscriptions:
            try:
                await sub.unsubscribe()
            except Exception:
                pass
                
        if self.nc:
            await self.nc.close()
            logger.info("Disconnected from NATS")
            
    async def setup(self):
        """Setup streams after connection."""
        if self.stream_manager:
            await self.stream_manager.setup_streams()
            
    async def publish_adsb(self, sensor_id: str, message: dict):
        """Publish ADS-B message."""
        subject = f"aerosentry.adsb.{sensor_id}"
        await self._publish(subject, message)
        
    async def publish_alert(self, sensor_id: str, alert: dict):
        """Publish anomaly alert."""
        severity = alert.get("severity", "unknown")
        subject = f"aerosentry.alerts.{severity}.{sensor_id}"
        await self._publish(subject, alert)
        
    async def publish_track(self, icao24: str, track: dict):
        """Publish track update."""
        subject = f"aerosentry.tracks.{icao24}"
        await self._publish(subject, track)
        
    async def publish_phy(self, sensor_id: str, features: dict):
        """Publish PHY features."""
        subject = f"aerosentry.phy.{sensor_id}"
        await self._publish(subject, features)
        
    async def publish_voice(self, sensor_id: str, transcript: dict):
        """Publish voice transcript."""
        subject = f"aerosentry.voice.{sensor_id}"
        await self._publish(subject, transcript)
        
    async def _publish(self, subject: str, data: dict):
        """Publish message to subject."""
        if not self.js:
            logger.warning("Not connected to NATS")
            return
            
        try:
            payload = json.dumps(data, default=str).encode()
            ack = await self.js.publish(subject, payload)
            logger.debug(f"Published to {subject}: seq={ack.seq}")
        except Exception as e:
            logger.error(f"Failed to publish to {subject}: {e}")
            
    async def subscribe(
        self,
        subject: str,
        handler: Callable[[Any], None],
        durable: Optional[str] = None,
        deliver_policy: str = "new"
    ):
        """Subscribe to subject with handler."""
        if not self.js:
            logger.warning("Not connected to NATS")
            return None
            
        try:
            async def message_handler(msg):
                try:
                    data = json.loads(msg.data.decode())
                    await handler(data)
                    await msg.ack()
                except Exception as e:
                    logger.error(f"Handler error: {e}")
                    
            policy = DeliverPolicy.NEW if deliver_policy == "new" else DeliverPolicy.ALL
            
            sub = await self.js.subscribe(
                subject,
                cb=message_handler,
                durable=durable,
                deliver_policy=policy
            )
            
            self._subscriptions.append(sub)
            logger.info(f"Subscribed to {subject}")
            return sub
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {subject}: {e}")
            return None


async def create_nats_client(url: str = "nats://localhost:4222") -> NATSClient:
    """Create and connect NATS client."""
    config = NATSConfig(url=url)
    client = NATSClient(config)
    
    if await client.connect():
        await client.setup()
        return client
        
    return None
