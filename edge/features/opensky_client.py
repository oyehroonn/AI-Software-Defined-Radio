"""OpenSky Network REST API client for live aircraft data.

Supports OAuth2 Client Credentials flow for accounts created since mid-March 2025.
Legacy accounts can still use basic auth (username/password) but this is deprecated.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, UTC
from typing import Optional, AsyncGenerator

logger = logging.getLogger(__name__)

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not available, OpenSky client will not work")


# Unit conversion constants
METERS_TO_FEET = 3.28084
MPS_TO_KNOTS = 1.94384

# OAuth2 endpoints
AUTH_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
TOKEN_EXPIRY_BUFFER = 60  # Refresh token 60 seconds before expiry


class OpenSkyLiveClient:
    """Real-time data client for OpenSky Network REST API.
    
    OpenSky API Documentation: https://openskynetwork.github.io/opensky-api/
    
    Authentication:
        - OAuth2 Client Credentials (required for accounts created since March 2025)
        - Basic Auth (deprecated, for legacy accounts only)
        - Anonymous (limited to 400 credits/day)
    
    Rate Limits (authenticated):
        - 4000 API credits per day (default)
        - 8000 API credits per day (active contributors)
    
    Credit Usage:
        - < 25 sq deg area: 1 credit
        - 25-100 sq deg: 2 credits
        - 100-400 sq deg: 3 credits
        - > 400 sq deg or global: 4 credits
    """
    
    API_URL = "https://opensky-network.org/api/states/all"
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        poll_interval: float = 10.0,
        bbox: Optional[tuple[float, float, float, float]] = None
    ):
        """Initialize OpenSky client.
        
        Args:
            client_id: OAuth2 client ID (from OpenSky account page)
            client_secret: OAuth2 client secret
            poll_interval: Seconds between API polls (default 10)
            bbox: Bounding box (lat_min, lat_max, lon_min, lon_max) to filter aircraft
        """
        self.client_id = client_id or os.getenv("OPENSKY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("OPENSKY_CLIENT_SECRET")
        self.poll_interval = poll_interval
        self.bbox = bbox
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._last_request_time = 0
        self._consecutive_errors = 0
        self._max_backoff = 60.0
        
        # OAuth2 token management
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0
        
        # Track seen aircraft to detect new messages
        self._last_seen: dict[str, float] = {}
    
    async def _fetch_access_token(self) -> bool:
        """Fetch new OAuth2 access token using client credentials.
        
        Returns:
            True if token was successfully obtained, False otherwise
        """
        if not self.client_id or not self.client_secret:
            return False
            
        if not self.session:
            return False
            
        try:
            async with self.session.post(
                AUTH_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self._access_token = data.get("access_token")
                    expires_in = data.get("expires_in", 1800)  # Default 30 min
                    self._token_expires_at = time.time() + expires_in
                    logger.info(f"OAuth2 token obtained, expires in {expires_in}s")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get OAuth2 token: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"OAuth2 token request failed: {e}")
            return False
    
    async def _ensure_valid_token(self) -> bool:
        """Ensure we have a valid access token, refreshing if needed.
        
        Returns:
            True if we have a valid token (or don't need one), False on error
        """
        if not self.client_id or not self.client_secret:
            return True  # Anonymous access
            
        # Check if token needs refresh
        if self._access_token and time.time() < (self._token_expires_at - TOKEN_EXPIRY_BUFFER):
            return True  # Token still valid
            
        logger.info("Refreshing OAuth2 access token...")
        return await self._fetch_access_token()
    
    def _get_auth_headers(self) -> dict:
        """Get authentication headers for API requests.
        
        Returns:
            Dict with Authorization header if authenticated, empty dict otherwise
        """
        if self._access_token:
            return {"Authorization": f"Bearer {self._access_token}"}
        return {}
        
    async def connect(self):
        """Initialize HTTP session and authenticate."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required for OpenSky client. Install with: pip install aiohttp")
        
        # Create session without auth (we'll use Bearer token in headers)
        self.session = aiohttp.ClientSession()
        self._running = True
        
        if self.client_id and self.client_secret:
            # Attempt OAuth2 authentication
            if await self._fetch_access_token():
                logger.info(f"OpenSky client authenticated via OAuth2 (client: {self.client_id})")
            else:
                logger.warning("OAuth2 authentication failed, falling back to anonymous access")
        else:
            logger.warning("OpenSky client running without authentication (limited to 400 credits/day)")
            
        logger.info("OpenSky client connected")
        
    async def disconnect(self):
        """Close HTTP session."""
        self._running = False
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("OpenSky client disconnected")
        
    def _build_url(self) -> str:
        """Build API URL with bounding box if specified."""
        if self.bbox:
            lat_min, lat_max, lon_min, lon_max = self.bbox
            return (
                f"{self.API_URL}"
                f"?lamin={lat_min}&lamax={lat_max}"
                f"&lomin={lon_min}&lomax={lon_max}"
            )
        return self.API_URL
        
    def _parse_state_vector(self, state: list, request_time: int) -> Optional[dict]:
        """Parse OpenSky state vector array into message dict.
        
        OpenSky state vector format (index):
            0: icao24 - ICAO24 address
            1: callsign - Callsign (can be None)
            2: origin_country - Country of origin
            3: time_position - Unix timestamp of last position update
            4: last_contact - Unix timestamp of last contact
            5: longitude - WGS-84 longitude
            6: latitude - WGS-84 latitude
            7: baro_altitude - Barometric altitude in meters
            8: on_ground - Boolean
            9: velocity - Ground speed in m/s
            10: true_track - Track angle in degrees (clockwise from north)
            11: vertical_rate - Vertical rate in m/s
            12: sensors - IDs of sensors that received messages
            13: geo_altitude - Geometric altitude in meters
            14: squawk - Transponder code
            15: spi - Special purpose indicator
            16: position_source - Origin of position (0=ADS-B, 1=ASTERIX, 2=MLAT)
        """
        if len(state) < 17:
            return None
            
        icao24 = state[0]
        if not icao24:
            return None
            
        # Get position timestamp or use request time
        time_position = state[3] or state[4] or request_time
        
        # Convert altitude from meters to feet
        altitude = None
        if state[7] is not None:
            altitude = int(state[7] * METERS_TO_FEET)
        elif state[13] is not None:
            altitude = int(state[13] * METERS_TO_FEET)
            
        # Convert velocity from m/s to knots
        velocity = None
        if state[9] is not None:
            velocity = int(state[9] * MPS_TO_KNOTS)
            
        # Convert vertical rate from m/s to ft/min
        vert_rate = None
        if state[11] is not None:
            vert_rate = int(state[11] * METERS_TO_FEET * 60)
            
        # Clean up callsign
        callsign = state[1].strip() if state[1] else None
        
        return {
            "timestamp": datetime.fromtimestamp(time_position, UTC).isoformat(),
            "icao24": icao24.lower(),
            "callsign": callsign,
            "latitude": state[6],
            "longitude": state[5],
            "altitude": altitude,
            "velocity": velocity,
            "heading": state[10],
            "vert_rate": vert_rate,
            "squawk": state[14],
            "on_ground": state[8],
            "signal_level": None,  # OpenSky doesn't provide signal level
            "source": "opensky"
        }
        
    async def _fetch_states(self) -> list[dict]:
        """Fetch current state vectors from OpenSky API."""
        if not self.session:
            return []
        
        # Ensure we have a valid token before making the request
        if not await self._ensure_valid_token():
            logger.warning("Could not obtain valid token, attempting anonymous access")
            
        url = self._build_url()
        headers = self._get_auth_headers()
        
        try:
            async with self.session.get(url, headers=headers, timeout=30) as response:
                # Log rate limit info if available
                remaining = response.headers.get("X-Rate-Limit-Remaining")
                if remaining:
                    logger.debug(f"API credits remaining: {remaining}")
                
                if response.status == 200:
                    data = await response.json()
                    self._consecutive_errors = 0
                    
                    if not data or "states" not in data or not data["states"]:
                        logger.debug("No aircraft in response")
                        return []
                        
                    request_time = data.get("time", int(datetime.now(UTC).timestamp()))
                    
                    messages = []
                    for state in data["states"]:
                        msg = self._parse_state_vector(state, request_time)
                        if msg:
                            messages.append(msg)
                            
                    logger.debug(f"Fetched {len(messages)} aircraft from OpenSky")
                    return messages
                    
                elif response.status == 429:
                    retry_after = response.headers.get("X-Rate-Limit-Retry-After-Seconds", "unknown")
                    logger.warning(f"OpenSky rate limit exceeded, retry after {retry_after}s")
                    self._consecutive_errors += 1
                    return []
                    
                elif response.status == 401:
                    # Token may have expired, try to refresh
                    logger.warning("OpenSky authentication failed (401), attempting token refresh...")
                    self._access_token = None  # Force refresh
                    if await self._fetch_access_token():
                        logger.info("Token refreshed, retry on next poll")
                    else:
                        logger.error("Token refresh failed - check credentials")
                    self._consecutive_errors += 1
                    return []
                    
                else:
                    logger.error(f"OpenSky API error: {response.status}")
                    self._consecutive_errors += 1
                    return []
                    
        except asyncio.TimeoutError:
            logger.warning("OpenSky API request timed out")
            self._consecutive_errors += 1
            return []
            
        except Exception as e:
            logger.error(f"OpenSky API request failed: {e}")
            self._consecutive_errors += 1
            return []
            
    def _calculate_backoff(self) -> float:
        """Calculate backoff time based on consecutive errors."""
        if self._consecutive_errors == 0:
            return self.poll_interval
            
        # Exponential backoff: poll_interval * 2^errors, capped at max_backoff
        backoff = self.poll_interval * (2 ** self._consecutive_errors)
        return min(backoff, self._max_backoff)
        
    async def stream(self) -> AsyncGenerator[dict, None]:
        """Stream aircraft state vectors by polling the API.
        
        Yields:
            dict: Aircraft state message in standard format
        """
        if not self.session:
            await self.connect()
            
        logger.info(f"Starting OpenSky stream (poll interval: {self.poll_interval}s)")
        
        if self.bbox:
            logger.info(f"Filtering to bounding box: {self.bbox}")
            
        while self._running:
            try:
                messages = await self._fetch_states()
                
                for msg in messages:
                    yield msg
                    
                # Calculate next poll delay with backoff
                delay = self._calculate_backoff()
                await asyncio.sleep(delay)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(self._calculate_backoff())
                
    async def fetch_once(self) -> list[dict]:
        """Fetch state vectors once (non-streaming).
        
        Returns:
            list[dict]: List of aircraft state messages
        """
        if not self.session:
            await self.connect()
            
        return await self._fetch_states()


def parse_bbox_string(bbox_str: str) -> Optional[tuple[float, float, float, float]]:
    """Parse bounding box from string format.
    
    Args:
        bbox_str: Format "lat_min,lat_max,lon_min,lon_max"
        
    Returns:
        Tuple of (lat_min, lat_max, lon_min, lon_max) or None
    """
    if not bbox_str:
        return None
        
    try:
        parts = [float(x.strip()) for x in bbox_str.split(",")]
        if len(parts) != 4:
            return None
        return tuple(parts)
    except (ValueError, TypeError):
        return None


async def create_opensky_client(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    poll_interval: Optional[float] = None,
    bbox: Optional[tuple[float, float, float, float]] = None
) -> OpenSkyLiveClient:
    """Factory function to create and connect an OpenSky client.
    
    Reads configuration from environment variables if not provided:
        - OPENSKY_CLIENT_ID (OAuth2 client ID)
        - OPENSKY_CLIENT_SECRET (OAuth2 client secret)
        - OPENSKY_POLL_INTERVAL
        - OPENSKY_BBOX
    """
    # Get config from environment if not provided
    client_id = client_id or os.getenv("OPENSKY_CLIENT_ID")
    client_secret = client_secret or os.getenv("OPENSKY_CLIENT_SECRET")
    
    if poll_interval is None:
        poll_interval = float(os.getenv("OPENSKY_POLL_INTERVAL", "10"))
        
    if bbox is None:
        bbox_str = os.getenv("OPENSKY_BBOX")
        if bbox_str:
            bbox = parse_bbox_string(bbox_str)
            
    client = OpenSkyLiveClient(
        client_id=client_id,
        client_secret=client_secret,
        poll_interval=poll_interval,
        bbox=bbox
    )
    
    await client.connect()
    return client
