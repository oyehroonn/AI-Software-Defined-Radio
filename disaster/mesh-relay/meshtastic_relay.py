"""Meshtastic LoRa mesh network relay for emergency alerts."""

import logging
import threading
import time
import json
from datetime import datetime, UTC
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import meshtastic
try:
    import meshtastic
    import meshtastic.serial_interface
    MESHTASTIC_AVAILABLE = True
except ImportError:
    MESHTASTIC_AVAILABLE = False
    logger.warning("Meshtastic not available")


@dataclass
class MeshMessage:
    """Message received or sent via mesh network."""
    timestamp: datetime
    from_id: str
    to_id: str
    text: str
    snr: Optional[float] = None
    hop_limit: int = 3
    is_alert: bool = False


class AlertCompressor:
    """Compress alerts for LoRa transmission (max ~200 bytes)."""
    
    PREFIX = "!SKY:"
    
    SEVERITY_MAP = {
        "critical": "C",
        "high": "H",
        "medium": "M",
        "low": "L"
    }
    
    ALERT_TYPE_MAP = {
        "spoofing_suspected": "SPF",
        "replay_attack": "RPY",
        "kinematic_anomaly": "KIN",
        "ghost_aircraft": "GHO",
        "message_flood": "FLD",
        "track_inconsistency": "TRK",
        "phy_anomaly": "PHY"
    }
    
    def compress(self, alert: dict) -> str:
        """Compress alert to minimal text format."""
        severity = self.SEVERITY_MAP.get(
            alert.get("severity", "").lower(), "?"
        )
        
        alert_type = self.ALERT_TYPE_MAP.get(
            alert.get("alert_type", ""), "UNK"
        )
        
        icao24 = alert.get("icao24", "??????")[:6]
        
        lat = alert.get("latitude", 0) or 0
        lon = alert.get("longitude", 0) or 0
        
        # Format: !SKY:C|ABC123|SPF|40.71,-74.01
        compressed = (
            f"{self.PREFIX}{severity}"
            f"|{icao24}"
            f"|{alert_type}"
            f"|{lat:.2f},{lon:.2f}"
        )
        
        # Add score if space permits
        score = alert.get("anomaly_score")
        if score and len(compressed) < 180:
            compressed += f"|S{score:.2f}"
            
        return compressed
        
    def decompress(self, text: str) -> Optional[dict]:
        """Decompress alert from text format."""
        if not text.startswith(self.PREFIX):
            return None
            
        try:
            content = text[len(self.PREFIX):]
            parts = content.split("|")
            
            if len(parts) < 4:
                return None
                
            severity_char = parts[0]
            severity = next(
                (k for k, v in self.SEVERITY_MAP.items() if v == severity_char),
                "unknown"
            )
            
            icao24 = parts[1]
            
            alert_type_short = parts[2]
            alert_type = next(
                (k for k, v in self.ALERT_TYPE_MAP.items() if v == alert_type_short),
                "unknown"
            )
            
            lat, lon = parts[3].split(",")
            
            alert = {
                "severity": severity,
                "icao24": icao24,
                "alert_type": alert_type,
                "latitude": float(lat),
                "longitude": float(lon),
                "received_via_mesh": True
            }
            
            # Parse score if present
            for part in parts[4:]:
                if part.startswith("S"):
                    alert["anomaly_score"] = float(part[1:])
                    
            return alert
            
        except Exception as e:
            logger.error(f"Failed to decompress alert: {e}")
            return None


class MeshRelay:
    """Relay alerts over LoRa mesh network."""
    
    def __init__(
        self,
        serial_port: str = "/dev/ttyUSB0",
        channel: int = 0,
        use_mock: bool = False
    ):
        self.serial_port = serial_port
        self.channel = channel
        self.compressor = AlertCompressor()
        
        self.interface = None
        self.running = False
        self._receive_thread = None
        
        self.message_handlers: list[Callable[[MeshMessage], None]] = []
        self.alert_handlers: list[Callable[[dict], None]] = []
        
        if use_mock or not MESHTASTIC_AVAILABLE:
            logger.info("Using mock mesh relay")
            self._mock_mode = True
        else:
            self._mock_mode = False
            
    def connect(self):
        """Connect to Meshtastic device."""
        if self._mock_mode:
            logger.info("Mock mesh relay connected")
            return True
            
        try:
            self.interface = meshtastic.serial_interface.SerialInterface(
                self.serial_port
            )
            logger.info(f"Connected to Meshtastic on {self.serial_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Meshtastic: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from device."""
        if self.interface:
            self.interface.close()
            self.interface = None
            
    def start(self):
        """Start receiving messages."""
        self.running = True
        
        if not self._mock_mode and self.interface:
            # Register callback
            def on_receive(packet, interface):
                self._handle_packet(packet)
            
            self.interface.onReceive = on_receive
            
        logger.info("Mesh relay started")
        
    def stop(self):
        """Stop receiving messages."""
        self.running = False
        logger.info("Mesh relay stopped")
        
    def add_message_handler(self, handler: Callable[[MeshMessage], None]):
        """Add handler for incoming messages."""
        self.message_handlers.append(handler)
        
    def add_alert_handler(self, handler: Callable[[dict], None]):
        """Add handler for received alerts."""
        self.alert_handlers.append(handler)
        
    def send_alert(self, alert: dict, want_ack: bool = True) -> bool:
        """
        Send compressed alert over mesh.
        
        Args:
            alert: Alert dictionary
            want_ack: Request acknowledgment
            
        Returns:
            True if sent successfully
        """
        compressed = self.compressor.compress(alert)
        
        logger.info(f"Sending alert: {compressed}")
        
        if self._mock_mode:
            logger.info(f"[MOCK] Mesh TX: {compressed}")
            return True
            
        if not self.interface:
            logger.error("Mesh interface not connected")
            return False
            
        try:
            self.interface.sendText(
                compressed,
                wantAck=want_ack,
                channelIndex=self.channel
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send mesh message: {e}")
            return False
            
    def send_summary(self, summary_text: str) -> bool:
        """Send airspace summary over mesh."""
        # Prefix for summary messages
        message = f"!SKY:SUM|{summary_text}"
        
        if len(message) > 200:
            message = message[:200]
            
        if self._mock_mode:
            logger.info(f"[MOCK] Mesh TX: {message}")
            return True
            
        if not self.interface:
            return False
            
        try:
            self.interface.sendText(message, channelIndex=self.channel)
            return True
        except Exception as e:
            logger.error(f"Failed to send summary: {e}")
            return False
            
    def _handle_packet(self, packet: dict):
        """Handle incoming mesh packet."""
        decoded = packet.get("decoded", {})
        text = decoded.get("text", "")
        
        if not text:
            return
            
        # Create message object
        message = MeshMessage(
            timestamp=datetime.now(UTC),
            from_id=packet.get("fromId", "unknown"),
            to_id=packet.get("toId", "broadcast"),
            text=text,
            snr=packet.get("snr"),
            is_alert=text.startswith(AlertCompressor.PREFIX)
        )
        
        # Call message handlers
        for handler in self.message_handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
                
        # If it's an alert, decompress and call alert handlers
        if message.is_alert:
            alert = self.compressor.decompress(text)
            if alert:
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Alert handler error: {e}")


class MeshRelayService:
    """Service for managing mesh relay operations."""
    
    def __init__(
        self,
        relay: Optional[MeshRelay] = None,
        auto_relay_critical: bool = True,
        relay_interval_sec: int = 300
    ):
        self.relay = relay or MeshRelay(use_mock=True)
        self.auto_relay_critical = auto_relay_critical
        self.relay_interval = relay_interval_sec
        
        self.pending_alerts: list[dict] = []
        self.sent_alerts: set[str] = set()
        
        self._running = False
        self._relay_thread = None
        
    def start(self):
        """Start relay service."""
        self.relay.connect()
        self.relay.start()
        
        self._running = True
        self._relay_thread = threading.Thread(target=self._relay_loop)
        self._relay_thread.start()
        
        logger.info("Mesh relay service started")
        
    def stop(self):
        """Stop relay service."""
        self._running = False
        
        if self._relay_thread:
            self._relay_thread.join()
            
        self.relay.stop()
        self.relay.disconnect()
        
        logger.info("Mesh relay service stopped")
        
    def queue_alert(self, alert: dict):
        """Queue alert for relay."""
        alert_id = alert.get("alert_id", "")
        
        # Don't relay duplicates
        if alert_id in self.sent_alerts:
            return
            
        # Critical alerts get relayed immediately
        if self.auto_relay_critical and alert.get("severity") == "critical":
            self._send_alert(alert)
        else:
            self.pending_alerts.append(alert)
            
    def _send_alert(self, alert: dict):
        """Send alert via mesh."""
        alert_id = alert.get("alert_id", str(id(alert)))
        
        if self.relay.send_alert(alert):
            self.sent_alerts.add(alert_id)
            logger.info(f"Relayed alert {alert_id} via mesh")
            
    def _relay_loop(self):
        """Background loop for periodic relay."""
        last_relay = 0
        
        while self._running:
            time.sleep(1)
            
            now = time.time()
            
            # Periodic relay of pending alerts
            if now - last_relay >= self.relay_interval:
                last_relay = now
                
                # Send highest priority pending alert
                if self.pending_alerts:
                    # Sort by severity
                    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
                    self.pending_alerts.sort(
                        key=lambda a: severity_order.get(a.get("severity", "low"), 4)
                    )
                    
                    alert = self.pending_alerts.pop(0)
                    self._send_alert(alert)
