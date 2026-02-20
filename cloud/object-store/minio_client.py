"""MinIO/S3 client for IQ sample and recording storage."""

import io
import logging
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional, BinaryIO
import json

logger = logging.getLogger(__name__)

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    logger.warning("MinIO client not available")


class MinIOClient:
    """MinIO/S3 client wrapper."""
    
    def __init__(
        self,
        endpoint: str = None,
        access_key: str = None,
        secret_key: str = None,
        secure: bool = False
    ):
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.secure = secure
        self.client = None
        
        self._connect()
        
    def _connect(self):
        """Connect to MinIO server."""
        if not MINIO_AVAILABLE:
            logger.warning("MinIO not available")
            return
            
        try:
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )
            logger.info(f"Connected to MinIO at {self.endpoint}")
        except Exception as e:
            logger.error(f"Failed to connect to MinIO: {e}")
            
    def ensure_bucket(self, bucket_name: str) -> bool:
        """Ensure bucket exists."""
        if not self.client:
            return False
            
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Created bucket: {bucket_name}")
            return True
        except S3Error as e:
            logger.error(f"Failed to ensure bucket: {e}")
            return False
            
    def put_object(
        self,
        bucket: str,
        object_name: str,
        data: BinaryIO,
        length: int,
        content_type: str = "application/octet-stream",
        metadata: dict = None
    ) -> bool:
        """Upload object to bucket."""
        if not self.client:
            return False
            
        try:
            self.client.put_object(
                bucket,
                object_name,
                data,
                length,
                content_type=content_type,
                metadata=metadata
            )
            logger.debug(f"Uploaded {object_name} to {bucket}")
            return True
        except S3Error as e:
            logger.error(f"Failed to upload object: {e}")
            return False
            
    def get_object(self, bucket: str, object_name: str) -> Optional[bytes]:
        """Download object from bucket."""
        if not self.client:
            return None
            
        try:
            response = self.client.get_object(bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            logger.error(f"Failed to get object: {e}")
            return None
            
    def list_objects(self, bucket: str, prefix: str = None) -> list[dict]:
        """List objects in bucket."""
        if not self.client:
            return []
            
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            return [
                {
                    "name": obj.object_name,
                    "size": obj.size,
                    "modified": obj.last_modified.isoformat() if obj.last_modified else None
                }
                for obj in objects
            ]
        except S3Error as e:
            logger.error(f"Failed to list objects: {e}")
            return []
            
    def delete_object(self, bucket: str, object_name: str) -> bool:
        """Delete object from bucket."""
        if not self.client:
            return False
            
        try:
            self.client.remove_object(bucket, object_name)
            logger.debug(f"Deleted {object_name} from {bucket}")
            return True
        except S3Error as e:
            logger.error(f"Failed to delete object: {e}")
            return False


class IQSampleStore:
    """Storage for IQ samples with metadata."""
    
    BUCKET_IQ = "aerosentry-iq"
    BUCKET_VOICE = "aerosentry-voice"
    
    def __init__(self, client: MinIOClient = None):
        self.client = client or MinIOClient()
        
        # Ensure buckets exist
        self.client.ensure_bucket(self.BUCKET_IQ)
        self.client.ensure_bucket(self.BUCKET_VOICE)
        
    def _generate_iq_path(
        self,
        sensor_id: str,
        icao24: str,
        timestamp: datetime
    ) -> str:
        """Generate object path for IQ sample."""
        date_prefix = timestamp.strftime("%Y/%m/%d")
        time_str = timestamp.strftime("%H%M%S_%f")
        return f"{sensor_id}/{date_prefix}/{icao24}_{time_str}.iq"
        
    def store_iq_sample(
        self,
        sensor_id: str,
        icao24: str,
        timestamp: datetime,
        iq_data: bytes,
        sample_rate: float,
        center_freq: float,
        metadata: dict = None
    ) -> Optional[str]:
        """Store IQ sample with metadata."""
        
        object_name = self._generate_iq_path(sensor_id, icao24, timestamp)
        
        # Combine metadata
        full_metadata = {
            "sensor_id": sensor_id,
            "icao24": icao24,
            "timestamp": timestamp.isoformat(),
            "sample_rate": str(sample_rate),
            "center_freq": str(center_freq),
        }
        if metadata:
            full_metadata.update({k: str(v) for k, v in metadata.items()})
            
        # Upload
        data_stream = io.BytesIO(iq_data)
        
        if self.client.put_object(
            self.BUCKET_IQ,
            object_name,
            data_stream,
            len(iq_data),
            content_type="application/octet-stream",
            metadata=full_metadata
        ):
            logger.info(f"Stored IQ sample: {object_name}")
            return object_name
            
        return None
        
    def retrieve_iq_sample(self, object_name: str) -> Optional[bytes]:
        """Retrieve IQ sample."""
        return self.client.get_object(self.BUCKET_IQ, object_name)
        
    def list_iq_samples(
        self,
        sensor_id: str = None,
        date: datetime = None
    ) -> list[dict]:
        """List IQ samples."""
        prefix = None
        if sensor_id:
            prefix = f"{sensor_id}/"
            if date:
                prefix += date.strftime("%Y/%m/%d/")
                
        return self.client.list_objects(self.BUCKET_IQ, prefix)
        
    def store_voice_recording(
        self,
        sensor_id: str,
        frequency: float,
        timestamp: datetime,
        audio_data: bytes,
        transcript: str = None,
        metadata: dict = None
    ) -> Optional[str]:
        """Store voice recording with optional transcript."""
        
        date_prefix = timestamp.strftime("%Y/%m/%d")
        time_str = timestamp.strftime("%H%M%S_%f")
        freq_str = f"{frequency:.1f}".replace(".", "_")
        object_name = f"{sensor_id}/{date_prefix}/{freq_str}_{time_str}.wav"
        
        # Metadata
        full_metadata = {
            "sensor_id": sensor_id,
            "frequency": str(frequency),
            "timestamp": timestamp.isoformat(),
        }
        if transcript:
            full_metadata["transcript"] = transcript[:500]  # Limit size
        if metadata:
            full_metadata.update({k: str(v) for k, v in metadata.items()})
            
        # Upload audio
        data_stream = io.BytesIO(audio_data)
        
        if self.client.put_object(
            self.BUCKET_VOICE,
            object_name,
            data_stream,
            len(audio_data),
            content_type="audio/wav",
            metadata=full_metadata
        ):
            logger.info(f"Stored voice recording: {object_name}")
            return object_name
            
        return None
        
    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        iq_objects = self.client.list_objects(self.BUCKET_IQ)
        voice_objects = self.client.list_objects(self.BUCKET_VOICE)
        
        return {
            "iq_samples": {
                "count": len(iq_objects),
                "total_bytes": sum(o["size"] for o in iq_objects)
            },
            "voice_recordings": {
                "count": len(voice_objects),
                "total_bytes": sum(o["size"] for o in voice_objects)
            }
        }
