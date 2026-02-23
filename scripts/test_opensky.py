#!/usr/bin/env python3
"""Test script for OpenSky Network API integration.

This script verifies that the OpenSky API connection works correctly
and displays live aircraft data from the configured region.

Authentication:
    Uses OAuth2 Client Credentials flow (required for accounts created since March 2025).
    Create API credentials at https://opensky-network.org/account

Usage:
    # Test with default settings (anonymous, global data - limited to 400 credits/day)
    python scripts/test_opensky.py
    
    # Test with OAuth2 credentials
    OPENSKY_CLIENT_ID=your_client_id OPENSKY_CLIENT_SECRET=your_secret python scripts/test_opensky.py
    
    # Test with bounding box (San Francisco Bay Area) - uses less API credits
    OPENSKY_BBOX=37.0,38.5,-123.0,-121.0 python scripts/test_opensky.py
    
    # Full example with credentials and bounding box
    OPENSKY_CLIENT_ID=myclient OPENSKY_CLIENT_SECRET=mysecret OPENSKY_BBOX=40.4,41.2,-74.5,-73.5 python scripts/test_opensky.py
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_opensky_connection():
    """Test basic OpenSky API connection."""
    try:
        from edge.features.opensky_client import OpenSkyLiveClient, parse_bbox_string
    except ImportError as e:
        logger.error(f"Failed to import OpenSky client: {e}")
        logger.error("Make sure aiohttp is installed: pip install aiohttp")
        return False
        
    # Get configuration from environment (OAuth2)
    client_id = os.getenv("OPENSKY_CLIENT_ID")
    client_secret = os.getenv("OPENSKY_CLIENT_SECRET")
    bbox_str = os.getenv("OPENSKY_BBOX")
    poll_interval = float(os.getenv("OPENSKY_POLL_INTERVAL", "10"))
    
    bbox = parse_bbox_string(bbox_str) if bbox_str else None
    
    logger.info("=" * 60)
    logger.info("OpenSky Network API Test (OAuth2)")
    logger.info("=" * 60)
    logger.info(f"Client ID: {'(set)' if client_id else '(anonymous)'}")
    logger.info(f"Bounding box: {bbox if bbox else '(global)'}")
    logger.info(f"Poll interval: {poll_interval}s")
    logger.info("=" * 60)
    
    client = OpenSkyLiveClient(
        client_id=client_id,
        client_secret=client_secret,
        poll_interval=poll_interval,
        bbox=bbox
    )
    
    try:
        await client.connect()
        logger.info("Connected to OpenSky API successfully!")
        
        # Fetch a single batch of states
        logger.info("\nFetching aircraft states...")
        messages = await client.fetch_once()
        
        if not messages:
            logger.warning("No aircraft found in the specified area.")
            logger.info("Try adjusting your bounding box or check if you're rate limited.")
            return True  # Connection worked, just no data
            
        logger.info(f"\nFound {len(messages)} aircraft:\n")
        
        # Display header
        print(f"{'ICAO24':<10} {'Callsign':<10} {'Altitude':>10} {'Speed':>8} {'Heading':>8} {'Position':<25}")
        print("-" * 80)
        
        # Display aircraft (limit to 20 for readability)
        for i, msg in enumerate(messages[:20]):
            icao24 = msg.get('icao24', 'N/A')
            callsign = msg.get('callsign', 'N/A') or 'N/A'
            altitude = msg.get('altitude')
            velocity = msg.get('velocity')
            heading = msg.get('heading')
            lat = msg.get('latitude')
            lon = msg.get('longitude')
            
            alt_str = f"{altitude:>8} ft" if altitude else "N/A"
            vel_str = f"{velocity:>6} kt" if velocity else "N/A"
            hdg_str = f"{heading:>6.1f}°" if heading else "N/A"
            pos_str = f"({lat:.4f}, {lon:.4f})" if lat and lon else "N/A"
            
            print(f"{icao24:<10} {callsign:<10} {alt_str:>10} {vel_str:>8} {hdg_str:>8} {pos_str:<25}")
            
        if len(messages) > 20:
            print(f"\n... and {len(messages) - 20} more aircraft")
            
        logger.info("\nOpenSky API test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await client.disconnect()


async def test_opensky_streaming(duration_seconds: int = 30):
    """Test OpenSky streaming for a specified duration."""
    try:
        from edge.features.opensky_client import OpenSkyLiveClient, parse_bbox_string
    except ImportError as e:
        logger.error(f"Failed to import OpenSky client: {e}")
        return False
        
    client_id = os.getenv("OPENSKY_CLIENT_ID")
    client_secret = os.getenv("OPENSKY_CLIENT_SECRET")
    bbox_str = os.getenv("OPENSKY_BBOX")
    poll_interval = float(os.getenv("OPENSKY_POLL_INTERVAL", "10"))
    
    bbox = parse_bbox_string(bbox_str) if bbox_str else None
    
    client = OpenSkyLiveClient(
        client_id=client_id,
        client_secret=client_secret,
        poll_interval=poll_interval,
        bbox=bbox
    )
    
    logger.info(f"\nStreaming aircraft data for {duration_seconds} seconds...")
    logger.info("Press Ctrl+C to stop early.\n")
    
    message_count = 0
    unique_aircraft = set()
    start_time = datetime.now()
    
    try:
        await client.connect()
        
        async for msg in client.stream():
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= duration_seconds:
                break
                
            message_count += 1
            icao24 = msg.get('icao24')
            unique_aircraft.add(icao24)
            
            callsign = msg.get('callsign', 'N/A') or 'N/A'
            altitude = msg.get('altitude')
            
            alt_str = f"{altitude} ft" if altitude else "N/A"
            
            # Print progress every 50 messages
            if message_count <= 10 or message_count % 50 == 0:
                logger.info(f"[{message_count}] {icao24} ({callsign}) @ {alt_str}")
                
        logger.info(f"\nStreaming completed!")
        logger.info(f"Total messages: {message_count}")
        logger.info(f"Unique aircraft: {len(unique_aircraft)}")
        logger.info(f"Duration: {elapsed:.1f} seconds")
        return True
        
    except asyncio.CancelledError:
        logger.info("\nStreaming cancelled by user.")
        return True
        
    except Exception as e:
        logger.error(f"Streaming test failed: {e}")
        return False
        
    finally:
        await client.disconnect()


async def test_with_anomaly_detection():
    """Test OpenSky data with the anomaly detection pipeline."""
    try:
        from edge.features.opensky_client import OpenSkyLiveClient, parse_bbox_string
        from edge.features.track_features import TrackManager
        from edge.edge_inference.rules import RuleEngine
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return False
        
    client_id = os.getenv("OPENSKY_CLIENT_ID")
    client_secret = os.getenv("OPENSKY_CLIENT_SECRET")
    bbox_str = os.getenv("OPENSKY_BBOX")
    
    bbox = parse_bbox_string(bbox_str) if bbox_str else None
    
    client = OpenSkyLiveClient(
        client_id=client_id,
        client_secret=client_secret,
        poll_interval=10.0,
        bbox=bbox
    )
    
    track_manager = TrackManager()
    rule_engine = RuleEngine()
    
    logger.info("\nTesting with anomaly detection pipeline...")
    logger.info("Collecting data for 60 seconds to build track history...\n")
    
    message_count = 0
    alert_count = 0
    start_time = datetime.now()
    
    try:
        await client.connect()
        
        async for msg in client.stream():
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= 60:
                break
                
            message_count += 1
            icao24 = msg.get('icao24')
            
            if icao24:
                # Update track
                track_window = track_manager.update(icao24, msg)
                
                # Check for anomalies
                if track_window and track_window.can_compute():
                    features = track_window.compute_features()
                    triggers = rule_engine.evaluate(features, icao24)
                    
                    for trigger in triggers:
                        if trigger.severity.value >= 2:  # Medium or higher
                            alert_count += 1
                            logger.warning(
                                f"ALERT: {trigger.rule_id} for {icao24} "
                                f"(severity: {trigger.severity.name})"
                            )
                            
            # Progress update
            if message_count % 100 == 0:
                logger.info(f"Processed {message_count} messages, {alert_count} alerts, "
                          f"{len(track_manager.tracks)} active tracks")
                
        logger.info(f"\nTest completed!")
        logger.info(f"Total messages: {message_count}")
        logger.info(f"Total alerts: {alert_count}")
        logger.info(f"Active tracks: {len(track_manager.tracks)}")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await client.disconnect()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test OpenSky Network API integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--mode",
        choices=["connection", "stream", "detection"],
        default="connection",
        help="Test mode: connection (single fetch), stream (continuous), detection (with anomaly detection)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration in seconds for streaming mode (default: 30)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "connection":
        success = asyncio.run(test_opensky_connection())
    elif args.mode == "stream":
        success = asyncio.run(test_opensky_streaming(args.duration))
    elif args.mode == "detection":
        success = asyncio.run(test_with_anomaly_detection())
    else:
        success = False
        
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
