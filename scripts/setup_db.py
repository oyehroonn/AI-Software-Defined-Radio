#!/usr/bin/env python3
"""Database setup script for AeroSentry AI."""

import asyncio
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_database():
    """Setup TimescaleDB schema."""
    try:
        import asyncpg
    except ImportError:
        logger.error("asyncpg not installed. Run: pip install asyncpg")
        sys.exit(1)
        
    # Connection parameters
    host = os.getenv("TIMESCALE_HOST", "localhost")
    port = int(os.getenv("TIMESCALE_PORT", "5432"))
    database = os.getenv("TIMESCALE_DB", "aerosentry")
    user = os.getenv("TIMESCALE_USER", "aerosentry")
    password = os.getenv("TIMESCALE_PASSWORD", "aerosentry_secure")
    
    logger.info(f"Connecting to {host}:{port}/{database}")
    
    try:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        sys.exit(1)
        
    logger.info("Connected to database")
    
    # Read schema file
    schema_path = Path(__file__).parent.parent / "cloud" / "feature-store" / "schema.sql"
    
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        sys.exit(1)
        
    schema_sql = schema_path.read_text()
    
    # Execute schema
    logger.info("Applying schema...")
    
    try:
        await conn.execute(schema_sql)
        logger.info("Schema applied successfully")
    except Exception as e:
        logger.error(f"Failed to apply schema: {e}")
        sys.exit(1)
    finally:
        await conn.close()
        
    logger.info("Database setup complete")


def main():
    asyncio.run(setup_database())


if __name__ == "__main__":
    main()
