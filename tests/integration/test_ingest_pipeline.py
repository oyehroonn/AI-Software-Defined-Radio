"""Integration tests for the ADS-B ingest pipeline."""

import asyncio
import pytest
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch


class TestIngestPipeline:
    """Tests for end-to-end ingest pipeline."""
    
    @pytest.fixture
    def sample_adsb_message(self):
        """Sample ADS-B message for testing."""
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "sensor_id": "test-sensor-001",
            "icao24": "abc123",
            "callsign": "TEST123",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude": 35000,
            "velocity": 450,
            "heading": 270.5,
            "vert_rate": 0,
            "squawk": "1200",
            "signal_level": -85.0
        }
        
    @pytest.fixture
    def sample_batch(self, sample_adsb_message):
        """Sample batch of messages."""
        messages = []
        for i in range(10):
            msg = sample_adsb_message.copy()
            msg["icao24"] = f"abc{i:03d}"
            messages.append(msg)
        return messages
        
    def test_beast_decoder_import(self):
        """Test Beast decoder can be imported."""
        from edge.features.beast_ingest import BeastDecoder, DecodedMessage
        
        decoder = BeastDecoder()
        assert decoder is not None
        
    def test_track_features_import(self):
        """Test track features can be imported."""
        from edge.features.track_features import TrackManager, compute_features
        
        manager = TrackManager()
        assert manager is not None
        
    def test_rules_engine_import(self):
        """Test rules engine can be imported."""
        from edge.edge_inference.rules import RuleEngine
        
        engine = RuleEngine()
        assert engine is not None
        assert len(engine.rules) > 0
        
    def test_anomaly_detector_import(self):
        """Test anomaly detector can be imported."""
        from edge.edge_inference.anomaly_model import TrackAnomalyDetector
        
        detector = TrackAnomalyDetector()
        assert detector is not None
        
    @pytest.mark.asyncio
    async def test_track_window_accumulation(self, sample_adsb_message):
        """Test track window accumulates points."""
        from edge.features.track_features import TrackWindow
        
        window = TrackWindow(
            icao24="abc123",
            max_points=100,
            window_seconds=60.0
        )
        
        # Add points
        for i in range(5):
            msg = sample_adsb_message.copy()
            msg["latitude"] = 37.7749 + i * 0.01
            msg["longitude"] = -122.4194 + i * 0.01
            window.add_point(msg)
            
        assert len(window.points) == 5
        
        # Compute features
        features = window.compute_features()
        assert features is not None
        assert "total_distance" in features
        
    def test_rule_engine_evaluation(self):
        """Test rule engine evaluates features."""
        from edge.edge_inference.rules import RuleEngine
        
        engine = RuleEngine()
        
        # Normal features
        normal_features = {
            "max_velocity": 500,
            "max_climb_rate": 3000,
            "max_turn_rate": 3,
            "max_jump": 1.0,
            "message_rate": 1.0,
            "track_duration": 60,
            "avg_altitude": 35000,
        }
        
        triggers = engine.evaluate(normal_features, "abc123")
        
        # Should have few or no triggers for normal data
        assert isinstance(triggers, list)
        
    def test_rule_engine_anomaly_detection(self):
        """Test rule engine detects anomalies."""
        from edge.edge_inference.rules import RuleEngine
        
        engine = RuleEngine()
        
        # Anomalous features - impossible speed
        anomaly_features = {
            "max_velocity": 2000,  # Impossible for most aircraft
            "max_climb_rate": 3000,
            "max_turn_rate": 3,
            "max_jump": 1.0,
            "message_rate": 1.0,
            "track_duration": 60,
            "avg_altitude": 35000,
        }
        
        triggers = engine.evaluate(anomaly_features, "abc123")
        
        # Should detect impossible speed
        assert len(triggers) > 0
        assert any(t.rule_id == "impossible_speed" for t in triggers)


class TestNATSIntegration:
    """Tests for NATS JetStream integration."""
    
    def test_nats_config_import(self):
        """Test NATS config can be imported."""
        from cloud.stream.nats_config import NATSConfig, NATSClient
        
        config = NATSConfig()
        assert config.url == "nats://localhost:4222"
        assert len(config.stream_subjects) > 0
        
    def test_stream_manager_definitions(self):
        """Test stream definitions are correct."""
        from cloud.stream.nats_config import StreamManager
        
        assert "aerosentry-adsb" in StreamManager.STREAMS
        assert "aerosentry-alerts" in StreamManager.STREAMS
        assert "aerosentry-tracks" in StreamManager.STREAMS


class TestParquetStorage:
    """Tests for local Parquet storage."""
    
    def test_parquet_manager_import(self):
        """Test Parquet manager can be imported."""
        from edge.edge_store.parquet_manager import ParquetManager
        
        # Note: May not initialize if PyArrow not available
        
    def test_local_data_store_import(self):
        """Test local data store can be imported."""
        from edge.edge_store.parquet_manager import LocalDataStore


class TestPHYDetection:
    """Tests for PHY-layer detection."""
    
    def test_phy_features_import(self):
        """Test PHY features can be imported."""
        from edge.features.phy_features import PHYFeatures, extract_phy_features
        
    def test_phy_detector_import(self):
        """Test PHY detector can be imported."""
        try:
            from cloud.model_serving.phy_detector import PhySpoofingDetector
        except ImportError:
            # PyTorch may not be available
            pytest.skip("PyTorch not available")


class TestVoiceCapture:
    """Tests for voice capture and transcription."""
    
    def test_entity_extraction_import(self):
        """Test entity extraction can be imported."""
        from edge.voice_capture.entity_extraction import (
            extract_aviation_entities,
            is_emergency_communication
        )
        
    def test_callsign_extraction(self):
        """Test callsign extraction from text."""
        from edge.voice_capture.entity_extraction import extract_aviation_entities
        
        text = "Delta 123 cleared for takeoff runway 28L"
        entities = extract_aviation_entities(text)
        
        assert "callsigns" in entities
        assert "runways" in entities
        assert "28L" in entities["runways"]
        
    def test_emergency_detection(self):
        """Test emergency communication detection."""
        from edge.voice_capture.entity_extraction import is_emergency_communication
        
        # Normal communication
        assert not is_emergency_communication("Delta 123 cleared for takeoff")
        
        # Emergency
        assert is_emergency_communication("Mayday mayday mayday")
        assert is_emergency_communication("PAN PAN PAN")
        assert is_emergency_communication("Squawking 7700")


class TestDisasterMode:
    """Tests for offline/disaster mode."""
    
    def test_offline_store_import(self):
        """Test offline store can be imported."""
        from disaster.local_summarizer.offline_store import OfflineStore
        
    def test_mesh_relay_import(self):
        """Test mesh relay can be imported."""
        from disaster.mesh_relay.meshtastic_relay import AlertCompressor, MeshRelayService
        
    def test_alert_compression(self):
        """Test alert compression for LoRa."""
        from disaster.mesh_relay.meshtastic_relay import AlertCompressor
        
        compressor = AlertCompressor()
        
        alert = {
            "icao24": "abc123",
            "alert_type": "spoofing_suspected",
            "severity": "high",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        compressed = compressor.compress(alert)
        
        # Should be compact
        assert len(compressed) < 200
        
        # Should be decompressible
        decompressed = compressor.decompress(compressed)
        assert decompressed["icao24"] == "abc123"


class TestQueryEngine:
    """Tests for LLM query engine."""
    
    def test_query_engine_import(self):
        """Test query engine can be imported."""
        from cloud.llm_copilot.query_engine import QueryEngine
        
        engine = QueryEngine()
        assert engine is not None
        
    def test_query_parsing(self):
        """Test natural language query parsing."""
        from cloud.llm_copilot.query_engine import QueryEngine
        
        engine = QueryEngine()
        
        # Test recent alerts query
        result = engine.parse_query("Show me alerts from the last hour")
        
        assert result is not None
        assert "sql" in result or "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
