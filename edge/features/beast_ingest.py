"""Beast format ADS-B message ingestion service."""

import asyncio
import struct
import logging
from datetime import datetime, UTC
from typing import Optional, AsyncGenerator
from dataclasses import dataclass

import pyModeS as pms

logger = logging.getLogger(__name__)


@dataclass
class DecodedMessage:
    """Decoded ADS-B message structure."""
    timestamp: datetime
    icao24: str
    downlink_format: int
    raw_hex: str
    callsign: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[int] = None
    velocity: Optional[int] = None
    heading: Optional[float] = None
    vertical_rate: Optional[int] = None
    squawk: Optional[str] = None
    signal_level: Optional[float] = None


class CPRDecoder:
    """Compact Position Reporting decoder for ADS-B positions."""
    
    def __init__(self, receiver_lat: float, receiver_lon: float):
        self.receiver_lat = receiver_lat
        self.receiver_lon = receiver_lon
        self.odd_messages: dict[str, tuple[str, datetime]] = {}
        self.even_messages: dict[str, tuple[str, datetime]] = {}
        self.max_age_seconds = 10
        
    def add_position_message(
        self, 
        icao24: str, 
        hex_msg: str, 
        timestamp: datetime
    ) -> Optional[tuple[float, float]]:
        """Add a position message and attempt to decode position."""
        tc = pms.adsb.typecode(hex_msg)
        if not (9 <= tc <= 18 or 20 <= tc <= 22):
            return None
            
        cpr_flag = pms.adsb.oe_flag(hex_msg)
        
        if cpr_flag == 0:  # Even
            self.even_messages[icao24] = (hex_msg, timestamp)
        else:  # Odd
            self.odd_messages[icao24] = (hex_msg, timestamp)
            
        even = self.even_messages.get(icao24)
        odd = self.odd_messages.get(icao24)
        
        if even and odd:
            even_msg, even_time = even
            odd_msg, odd_time = odd
            
            time_diff = abs((even_time - odd_time).total_seconds())
            if time_diff > self.max_age_seconds:
                return None
                
            try:
                if even_time > odd_time:
                    pos = pms.adsb.position(
                        even_msg, odd_msg, even_time.timestamp(), odd_time.timestamp()
                    )
                else:
                    pos = pms.adsb.position(
                        even_msg, odd_msg, even_time.timestamp(), odd_time.timestamp()
                    )
                    
                if pos:
                    return pos
            except Exception as e:
                logger.debug(f"CPR decode failed: {e}")
                
        return None


class BeastDecoder:
    """Decoder for Beast binary format messages."""
    
    BEAST_ESCAPE = 0x1A
    BEAST_MSG_TYPE_MLAT = 0x32  # '2'
    BEAST_MSG_TYPE_SHORT = 0x31  # '1'
    BEAST_MSG_TYPE_LONG = 0x33  # '3'
    
    def __init__(self):
        self.buffer = bytearray()
        
    def feed(self, data: bytes) -> list[tuple[bytes, int]]:
        """Feed raw bytes and return decoded messages."""
        self.buffer.extend(data)
        messages = []
        
        while len(self.buffer) > 0:
            # Find message start
            if self.buffer[0] != self.BEAST_ESCAPE:
                try:
                    idx = self.buffer.index(self.BEAST_ESCAPE)
                    self.buffer = self.buffer[idx:]
                except ValueError:
                    self.buffer.clear()
                    break
                    
            if len(self.buffer) < 2:
                break
                
            msg_type = self.buffer[1]
            
            if msg_type == self.BEAST_MSG_TYPE_SHORT:
                msg_len = 2 + 6 + 1 + 7  # escape + type + timestamp + signal + message
            elif msg_type == self.BEAST_MSG_TYPE_LONG:
                msg_len = 2 + 6 + 1 + 14
            elif msg_type == self.BEAST_MSG_TYPE_MLAT:
                msg_len = 2 + 6 + 1 + 14
            else:
                self.buffer = self.buffer[1:]
                continue
                
            if len(self.buffer) < msg_len:
                break
                
            # Extract and unescape message
            raw_msg = self._unescape(self.buffer[:msg_len])
            self.buffer = self.buffer[msg_len:]
            
            if raw_msg:
                # Extract signal level (byte 8 after escape and type)
                signal = raw_msg[8] if len(raw_msg) > 8 else 0
                # Extract message bytes
                msg_bytes = raw_msg[9:]
                messages.append((msg_bytes, signal))
                
        return messages
        
    def _unescape(self, data: bytearray) -> Optional[bytearray]:
        """Remove Beast escape sequences."""
        result = bytearray()
        i = 0
        while i < len(data):
            if data[i] == self.BEAST_ESCAPE and i + 1 < len(data):
                if data[i + 1] == self.BEAST_ESCAPE:
                    result.append(self.BEAST_ESCAPE)
                    i += 2
                else:
                    result.append(data[i])
                    i += 1
            else:
                result.append(data[i])
                i += 1
        return result


def decode_adsb_message(
    hex_msg: str, 
    cpr_decoder: Optional[CPRDecoder] = None,
    timestamp: Optional[datetime] = None
) -> Optional[DecodedMessage]:
    """Decode a hex ADS-B message to structured data."""
    if timestamp is None:
        timestamp = datetime.now(UTC)
        
    # Validate CRC
    if pms.crc(hex_msg) != 0:
        return None
        
    icao24 = pms.adsb.icao(hex_msg)
    df = pms.df(hex_msg)
    
    record = DecodedMessage(
        timestamp=timestamp,
        icao24=icao24,
        downlink_format=df,
        raw_hex=hex_msg
    )
    
    if df == 17:  # ADS-B Extended Squitter
        tc = pms.adsb.typecode(hex_msg)
        
        # Aircraft identification (TC 1-4)
        if 1 <= tc <= 4:
            record.callsign = pms.adsb.callsign(hex_msg)
            
        # Airborne position (TC 9-18)
        elif 9 <= tc <= 18:
            record.altitude = pms.adsb.altitude(hex_msg)
            if cpr_decoder:
                pos = cpr_decoder.add_position_message(icao24, hex_msg, timestamp)
                if pos:
                    record.latitude, record.longitude = pos
                    
        # Airborne velocity (TC 19)
        elif tc == 19:
            velocity_data = pms.adsb.velocity(hex_msg)
            if velocity_data:
                record.velocity = int(velocity_data[0]) if velocity_data[0] else None
                record.heading = velocity_data[1]
                record.vertical_rate = velocity_data[2]
                
        # Surface position (TC 5-8)
        elif 5 <= tc <= 8:
            record.velocity = pms.adsb.surface_velocity(hex_msg)
            
    elif df == 4 or df == 20:  # Altitude reply
        record.altitude = pms.altcode(hex_msg)
        
    elif df == 5 or df == 21:  # Identity reply
        record.squawk = pms.idcode(hex_msg)
        
    return record


class BeastClient:
    """Async client for Beast format TCP connection."""
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 30005,
        receiver_lat: float = 0.0,
        receiver_lon: float = 0.0
    ):
        self.host = host
        self.port = port
        self.decoder = BeastDecoder()
        self.cpr_decoder = CPRDecoder(receiver_lat, receiver_lon)
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        
    async def connect(self):
        """Establish connection to Beast server."""
        self.reader, self.writer = await asyncio.open_connection(
            self.host, self.port
        )
        logger.info(f"Connected to Beast server at {self.host}:{self.port}")
        
    async def disconnect(self):
        """Close connection."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            
    async def read_messages(self) -> AsyncGenerator[DecodedMessage, None]:
        """Async generator yielding decoded messages."""
        if not self.reader:
            raise RuntimeError("Not connected")
            
        while True:
            try:
                data = await self.reader.read(4096)
                if not data:
                    break
                    
                for msg_bytes, signal in self.decoder.feed(data):
                    hex_msg = msg_bytes.hex().upper()
                    msg = decode_adsb_message(
                        hex_msg, 
                        self.cpr_decoder,
                        datetime.now(UTC)
                    )
                    if msg:
                        msg.signal_level = signal / 255.0 * 100
                        yield msg
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error reading Beast stream: {e}")
                await asyncio.sleep(1)
                
    async def stream(self) -> AsyncGenerator[dict, None]:
        """Stream messages as dicts (common DataSource interface).
        
        This method implements the DataSourceProtocol interface, allowing
        BeastClient to be used interchangeably with other data sources
        like OpenSkyLiveClient.
        
        Yields:
            dict: Aircraft state message with standard fields
        """
        if not self.reader:
            await self.connect()
            
        async for msg in self.read_messages():
            yield {
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else datetime.now(UTC).isoformat(),
                "icao24": msg.icao24,
                "callsign": msg.callsign,
                "latitude": msg.latitude,
                "longitude": msg.longitude,
                "altitude": msg.altitude,
                "velocity": msg.velocity,
                "heading": msg.heading,
                "vert_rate": msg.vertical_rate,
                "squawk": msg.squawk,
                "signal_level": msg.signal_level,
                "source": "beast"
            }


async def main():
    """Example usage."""
    client = BeastClient(
        host="localhost",
        port=30005,
        receiver_lat=40.7128,
        receiver_lon=-74.0060
    )
    
    await client.connect()
    
    try:
        async for msg in client.read_messages():
            print(f"[{msg.timestamp}] {msg.icao24}: "
                  f"alt={msg.altitude} vel={msg.velocity} "
                  f"lat={msg.latitude} lon={msg.longitude}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
