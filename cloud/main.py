"""Main entry point for AeroSentry cloud services."""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, UTC

import uvicorn

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("aerosentry.cloud")


async def start_services():
    """Start all cloud services."""
    logger.info("Starting AeroSentry cloud services...")
    
    # Import services
    from cloud.ingest_api.main import app as ingest_app
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # Start API server
    config = uvicorn.Config(
        ingest_app,
        host=host,
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    logger.info(f"Ingest API starting on {host}:{port}")
    
    await server.serve()


def main():
    """Main entry point."""
    try:
        asyncio.run(start_services())
    except KeyboardInterrupt:
        logger.info("Cloud services stopped")


if __name__ == "__main__":
    main()
