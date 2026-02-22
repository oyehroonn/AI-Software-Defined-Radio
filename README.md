# AeroSentry AI - Aviation Security Monitoring System

AeroSentry AI is a comprehensive Software-Defined Radio (SDR) based aviation surveillance system that monitors ADS-B signals for anomalies, detects potential spoofing attacks, captures ATC voice communications, and provides AI-powered analysis of airspace security.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Data Flow](#data-flow)
4. [Project Structure](#project-structure)
5. [Core Components](#core-components)
6. [Detection Methods](#detection-methods)
7. [Installation](#installation)
8. [Configuration](#configuration)
9. [Running the System](#running-the-system)
10. [API Reference](#api-reference)
11. [Database Schema](#database-schema)
12. [Deployment Options](#deployment-options)
13. [Testing](#testing)

---

## System Overview

AeroSentry AI operates as a distributed system with two main deployment tiers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EDGE NODES                                      │
│  (Raspberry Pi / Edge Compute Devices at Each Monitoring Location)          │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ SDR Receiver │  │ ADS-B Decode │  │   Feature    │  │   Anomaly    │    │
│  │  (RTL-SDR)   │──│  (readsb)    │──│  Extraction  │──│  Detection   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│         │                                                      │            │
│         ▼                                                      ▼            │
│  ┌──────────────┐                                    ┌──────────────┐      │
│  │  IQ Capture  │                                    │ Local Store  │      │
│  │ (PHY Layer)  │                                    │  (Parquet)   │      │
│  └──────────────┘                                    └──────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ NATS JetStream
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLOUD SERVICES                                  │
│  (Central Server for Multi-Sensor Aggregation)                              │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Ingest API  │  │ TimescaleDB  │  │Model Serving │  │ LLM Copilot  │    │
│  │  (FastAPI)   │──│ (Time-Series)│──│  (PyTorch)   │──│  (GPT-4)     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│         │                                                      │            │
│         ▼                                                      ▼            │
│  ┌──────────────┐                                    ┌──────────────┐      │
│  │    MinIO     │                                    │    Web UI    │      │
│  │ (IQ Storage) │                                    │  (Leaflet)   │      │
│  └──────────────┘                                    └──────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Capabilities

- **Real-time ADS-B monitoring** - Decode and track all aircraft in range
- **Multi-layer anomaly detection** - Rule-based + ML + PHY-layer analysis
- **Spoofing detection** - Identify fake aircraft using RF fingerprinting
- **ATC voice capture** - Record and transcribe air traffic communications
- **Offline/disaster mode** - Continue operation when cloud connectivity is lost
- **AI-powered analysis** - Natural language queries and incident reports

---

## Architecture

### Edge Node Architecture

Each edge node runs on low-cost hardware (Raspberry Pi 4/5 or similar) with an RTL-SDR dongle:

```
                    RTL-SDR (1090 MHz)
                           │
                           ▼
                 ┌─────────────────┐
                 │    ultrafeeder   │  Docker container running readsb
                 │   (readsb/tar)   │  Decodes raw SDR signals to Beast format
                 └────────┬────────┘
                          │ Beast TCP (port 30005)
                          ▼
                 ┌─────────────────┐
                 │  BeastClient    │  Async TCP client for Beast protocol
                 │  (beast_ingest) │  Handles message framing and escape sequences
                 └────────┬────────┘
                          │ DecodedMessage objects
                          ▼
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐
    │  TrackManager   │     │   IQ Capture    │
    │  (track_features)│     │  (iq_capture)   │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐
    │  TrackWindow    │     │  PHY Features   │
    │  - Kinematic    │     │  - CFO (freq)   │
    │  - Position     │     │  - Amplitude    │
    │  - Message rate │     │  - Phase        │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             └───────────┬───────────┘
                         ▼
              ┌─────────────────┐
              │  EnsembleDetector│
              │  - Rule Engine   │  11 physics-based rules
              │  - ML Model      │  Isolation Forest
              │  - PHY Detector  │  Deep Neural Network
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Alert Pipeline │  Cooldown, deduplication, routing
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  NATS    │ │  Local   │ │  Mesh    │
    │  Stream  │ │  Store   │ │  Relay   │
    └──────────┘ └──────────┘ └──────────┘
```

### Cloud Services Architecture

```
                   NATS JetStream (from edge nodes)
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐      ┌──────────────┐
│  aerosentry  │     │  aerosentry  │      │  aerosentry  │
│  .adsb.*     │     │  .alerts.*   │      │  .phy.*      │
│  (messages)  │     │  (anomalies) │      │  (features)  │
└──────┬───────┘     └──────┬───────┘      └──────┬───────┘
       │                    │                     │
       └────────────────────┼─────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Ingest API    │
                   │   (FastAPI)     │
                   │                 │
                   │  POST /ingest/* │
                   │  GET /tracks    │
                   │  GET /alerts    │
                   │  WS /ws/*       │
                   └────────┬────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
           ▼                ▼                ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │ TimescaleDB  │ │    MinIO     │ │   Grafana    │
   │              │ │              │ │              │
   │ - adsb_msgs  │ │ - IQ samples │ │ - Dashboards │
   │ - alerts     │ │ - Voice WAV  │ │ - Metrics    │
   │ - phy_feats  │ │              │ │              │
   │ - voice_txt  │ │              │ │              │
   └──────────────┘ └──────────────┘ └──────────────┘
           │
           ▼
   ┌──────────────┐     ┌──────────────┐
   │Model Serving │     │ LLM Copilot  │
   │              │     │              │
   │ - Track ML   │────▶│ - Query NL   │
   │ - PHY DNN    │     │ - Reports    │
   └──────────────┘     └──────────────┘
           │                    │
           └──────────┬─────────┘
                      ▼
              ┌──────────────┐
              │   Web UI     │
              │  (Leaflet)   │
              │              │
              │ - Live map   │
              │ - Alerts     │
              │ - Copilot    │
              └──────────────┘
```

---

## Data Flow

### Message Processing Flow

```
1. RF Signal Capture
   └─► RTL-SDR receives 1090 MHz signal
       └─► readsb demodulates and decodes Mode S
           └─► Outputs Beast binary format on TCP port 30005

2. Beast Protocol Parsing (beast_ingest.py)
   └─► BeastDecoder.feed(raw_bytes)
       ├─► Parse escape sequences (0x1A)
       ├─► Extract timestamp, signal level, message bytes
       └─► Return list of (message_bytes, signal_level)

3. ADS-B Decoding (beast_ingest.py)
   └─► decode_adsb_message(hex_msg)
       ├─► Validate CRC
       ├─► Extract ICAO24 address
       ├─► Decode by Downlink Format (DF):
       │   ├─► DF17: Extended Squitter
       │   │   ├─► TC 1-4: Aircraft identification
       │   │   ├─► TC 5-8: Surface position
       │   │   ├─► TC 9-18: Airborne position (CPR decode)
       │   │   └─► TC 19: Velocity
       │   ├─► DF4/20: Altitude
       │   └─► DF5/21: Squawk
       └─► Return DecodedMessage dataclass

4. Track Management (track_features.py)
   └─► TrackManager.update(icao24, message)
       ├─► Get or create TrackWindow for aircraft
       ├─► Add point to sliding window (60 second default)
       └─► If enough points, compute_features():
           ├─► Kinematic: velocity stats, climb rate, turn rate
           ├─► Position: distance traveled, max jump
           └─► Message: rate, variance, duration

5. Anomaly Detection (rules.py, anomaly_model.py)
   └─► EnsembleAnomalyDetector.detect(features)
       ├─► RuleEngine.evaluate(features)
       │   └─► Check 11 rules: impossible_speed, teleport, etc.
       ├─► TrackAnomalyDetector.predict(features)
       │   └─► Isolation Forest anomaly score
       └─► combine_scores(rule_score, ml_score, phy_score)

6. Alert Generation (alerts.py)
   └─► AlertPipeline.process(anomaly)
       ├─► Apply cooldown (prevent duplicate alerts)
       ├─► Assign severity based on scores
       └─► Route to: NATS, WebSocket, local store
```

### PHY-Layer Analysis Flow

```
1. IQ Capture Trigger
   └─► On ADS-B message decode, capture ±500µs of IQ samples
       └─► RingBuffer stores continuous IQ stream
           └─► Extract burst around message timing

2. PHY Feature Extraction (phy_features.py)
   └─► extract_phy_features(iq_burst, sample_rate)
       ├─► CFO: Carrier frequency offset from 1090 MHz
       │   └─► Each transmitter has unique oscillator drift
       ├─► Amplitude: Mean, std, rise time, overshoot
       │   └─► Transmitter power characteristics
       ├─► Phase: Mean, std, jitter
       │   └─► Phase noise fingerprint
       └─► Spectral: Entropy, bandwidth
           └─► Modulation quality

3. PHY Spoofing Detection (phy_detector.py)
   └─► CalibratedPhyDetector.predict(features, icao24)
       ├─► Stage 1: Message classifier (legitimate vs spoofed)
       ├─► Stage 2: Emitter encoder (RF fingerprint embedding)
       ├─► Compare embedding to historical for this ICAO24
       └─► Return: spoof_probability, confidence, uncertainty
```

---

## Project Structure

```
AI-Software-Defined-Radio/
│
├── edge/                           # Edge node components
│   ├── main.py                     # Edge node entry point
│   ├── adsb-decode/                # ADS-B decoder wrapper
│   │   └── readsb_wrapper.py       # readsb configuration
│   ├── features/                   # Feature extraction
│   │   ├── beast_ingest.py         # Beast protocol decoder
│   │   ├── track_features.py       # Track kinematic features
│   │   └── phy_features.py         # PHY-layer features
│   ├── edge-inference/             # On-device detection
│   │   ├── rules.py                # Rule-based detection
│   │   └── anomaly_model.py        # ML anomaly detection
│   ├── edge-store/                 # Local storage
│   │   └── parquet_manager.py      # Parquet file management
│   ├── iq-capture/                 # IQ sample capture
│   │   └── iq_capture.py           # GNU Radio flowgraph
│   └── voice-capture/              # ATC voice capture
│       ├── vhf_capture.py          # VHF AM demodulation
│       ├── transcription.py        # Whisper ASR
│       └── entity_extraction.py    # Aviation entity NER
│
├── cloud/                          # Cloud services
│   ├── main.py                     # Cloud entry point
│   ├── ingest-api/                 # Data ingestion
│   │   ├── main.py                 # FastAPI application
│   │   └── alerts.py               # Alert pipeline
│   ├── feature-store/              # Database
│   │   └── schema.sql              # TimescaleDB schema
│   ├── stream/                     # Message streaming
│   │   └── nats_config.py          # NATS JetStream setup
│   ├── object-store/               # Object storage
│   │   └── minio_client.py         # MinIO client
│   ├── model-serving/              # ML inference
│   │   ├── server.py               # Model serving API
│   │   └── phy_detector.py         # PHY DNN detector
│   ├── llm-copilot/                # AI assistant
│   │   ├── server.py               # Copilot API
│   │   ├── query_engine.py         # NL-to-SQL
│   │   └── response_generator.py   # Evidence-gated responses
│   └── web-ui/                     # Frontend
│       ├── server.py               # Web server
│       ├── incident_export.py      # Export functionality
│       ├── templates/index.html    # HTML template
│       └── static/app.js           # Frontend JavaScript
│
├── disaster/                       # Offline mode
│   ├── local-summarizer/           # Local analysis
│   │   ├── offline_store.py        # SQLite storage
│   │   └── summarizer.py           # Airspace summaries
│   └── mesh-relay/                 # LoRa mesh
│       └── meshtastic_relay.py     # Meshtastic integration
│
├── shared/                         # Shared utilities
│   ├── schemas/                    # Data schemas
│   │   ├── adsb_message.proto      # Protobuf definitions
│   │   └── config.py               # Configuration classes
│   ├── eval/                       # Evaluation tools
│   │   ├── opensky_loader.py       # OpenSky Network data
│   │   ├── synthetic_attacks.py    # Attack simulation
│   │   └── metrics.py              # Evaluation metrics
│   └── docs/                       # Documentation
│       ├── COMPLIANCE.md           # Legal compliance
│       └── THREAT_MODEL.md         # Security analysis
│
├── docker/                         # Docker configuration
│   ├── docker-compose.edge.yml     # Edge deployment
│   ├── docker-compose.cloud.yml    # Cloud deployment
│   ├── Dockerfile.edge             # Edge container
│   ├── Dockerfile.api              # API container
│   ├── Dockerfile.voice            # Voice capture container
│   ├── Dockerfile.mesh             # Mesh relay container
│   ├── Dockerfile.models           # Model serving container
│   ├── Dockerfile.copilot          # LLM copilot container
│   └── Dockerfile.webui            # Web UI container
│
├── deploy/                         # Deployment configs
│   ├── kubernetes/                 # K8s manifests
│   │   ├── namespace.yaml
│   │   ├── configmap.yaml
│   │   ├── timescaledb.yaml
│   │   ├── nats.yaml
│   │   ├── ingest-api.yaml
│   │   ├── model-serving.yaml
│   │   ├── web-ui.yaml
│   │   └── kustomization.yaml
│   └── grafana/                    # Grafana dashboards
│       └── provisioning/
│
├── scripts/                        # Utility scripts
│   ├── setup_db.py                 # Database initialization
│   └── run_evaluation.py           # Run detector evaluation
│
├── tests/                          # Test suite
│   ├── conftest.py                 # Pytest fixtures
│   ├── unit/                       # Unit tests
│   └── integration/                # Integration tests
│
├── models/                         # Trained ML models
├── requirements.txt                # Python dependencies
├── requirements-edge.txt           # Edge dependencies
├── requirements-cloud.txt          # Cloud dependencies
├── .env.example                    # Environment template
└── README.md                       # This file
```

---

## Core Components

### 1. Beast Protocol Decoder (`edge/features/beast_ingest.py`)

The Beast format is a binary protocol for ADS-B messages used by readsb/dump1090:

```python
class BeastDecoder:
    """Decodes Beast binary format from TCP stream."""
    
    # Message types
    BEAST_ESCAPE = 0x1A        # Escape character
    BEAST_MSG_TYPE_SHORT = 0x31  # 7-byte Mode S short
    BEAST_MSG_TYPE_LONG = 0x33   # 14-byte Mode S long
    
    def feed(self, data: bytes) -> list[tuple[bytes, int]]:
        """Feed raw bytes, return decoded messages with signal levels."""
```

**Message structure:**
```
[0x1A][type][6-byte timestamp][1-byte signal][7 or 14 byte message]
```

### 2. Track Feature Extraction (`edge/features/track_features.py`)

Extracts kinematic and behavioral features from aircraft tracks:

```python
class TrackWindow:
    """Sliding window of track points for feature computation."""
    
    def compute_features(self) -> dict:
        """Compute features from track window."""
        return {
            # Velocity features
            "avg_velocity": ...,      # Mean ground speed (knots)
            "max_velocity": ...,      # Maximum speed
            "velocity_std": ...,      # Speed variation
            
            # Altitude features  
            "avg_altitude": ...,      # Mean altitude (feet)
            "max_climb_rate": ...,    # Max vertical rate (ft/min)
            
            # Position features
            "total_distance": ...,    # Distance traveled (nm)
            "max_jump": ...,          # Largest position gap (nm)
            
            # Turn features
            "avg_turn_rate": ...,     # Mean turn rate (deg/sec)
            "max_turn_rate": ...,     # Maximum turn rate
            
            # Message features
            "message_rate": ...,      # Messages per second
            "track_duration": ...,    # Track length (seconds)
        }
```

### 3. Rule-Based Detection (`edge/edge-inference/rules.py`)

Physics-based rules that detect physically impossible behavior:

| Rule ID | Description | Threshold | Severity |
|---------|-------------|-----------|----------|
| `impossible_speed` | Velocity > Mach 1.5 | > 900 kts | HIGH |
| `teleport_detected` | Position jump implies > 2000 kts | > 10 nm | CRITICAL |
| `extreme_climb` | Climb rate exceeds limits | > 10,000 ft/min | MEDIUM |
| `impossible_turn` | Turn rate exceeds structural limits | > 10 deg/sec | HIGH |
| `sparse_track` | Unusually low message rate | < 0.1 msg/sec | LOW |
| `message_burst` | Abnormally high message rate | > 5 msg/sec | MEDIUM |
| `impossible_altitude` | Altitude exceeds ceiling | > 60,000 ft | MEDIUM |
| `speed_inconsistency` | Reported vs implied speed mismatch | > 200 kts | MEDIUM |
| `extreme_acceleration` | Acceleration exceeds limits | > 50 kts/sec | HIGH |
| `erratic_altitude` | High altitude variance | > 5000 ft std | MEDIUM |
| `vertical_rate_mismatch` | Reported vs computed climb mismatch | - | MEDIUM |

### 4. ML Anomaly Detection (`edge/edge-inference/anomaly_model.py`)

Unsupervised anomaly detection using Isolation Forest:

```python
class TrackAnomalyDetector:
    """Isolation Forest based track anomaly detector."""
    
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            max_samples='auto'
        )
        self.scaler = StandardScaler()
    
    def fit(self, training_features: list[dict]):
        """Train on normal traffic data."""
        
    def predict(self, features: dict) -> float:
        """Return anomaly score [0-1], higher = more anomalous."""
```

### 5. PHY-Layer Detection (`cloud/model-serving/phy_detector.py`)

Deep Neural Network for RF fingerprinting-based spoofing detection:

```python
class PhySpoofingDetector(nn.Module):
    """Two-stage DNN for PHY-layer spoofing detection."""
    
    def __init__(self, input_dim=10, embedding_dim=64):
        # Stage 1: Message classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [legitimate, spoofed]
        )
        
        # Stage 2: Emitter encoder (RF fingerprint)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
```

**PHY Features used:**
- Carrier Frequency Offset (CFO) - transmitter oscillator drift
- Amplitude statistics - power amplifier characteristics
- Preamble rise time and overshoot - modulator behavior
- Phase noise - oscillator quality
- Spectral features - modulation quality

### 6. LLM Copilot (`cloud/llm-copilot/`)

Evidence-gated AI assistant for airspace queries:

```python
class QueryEngine:
    """Natural language to SQL query engine."""
    
    QUERY_PATTERNS = {
        r"alerts?.*(last|past)\s+(\d+)\s*(hour|minute|day)": "recent_alerts",
        r"aircraft.*(near|around|at)": "aircraft_by_location",
        r"track.*icao24?.*(\w{6})": "track_by_icao",
    }
    
    SQL_TEMPLATES = {
        "recent_alerts": """
            SELECT * FROM anomaly_alerts 
            WHERE time > NOW() - INTERVAL '{duration}'
            ORDER BY time DESC
        """,
    }

class EvidenceGatedResponder:
    """Generate responses only from provided evidence."""
    
    SYSTEM_PROMPT = """
    You are an aviation security analyst. CRITICAL RULES:
    1. ONLY make claims supported by the provided data
    2. NEVER invent aircraft, alerts, or events
    3. ALWAYS cite the data source for claims
    4. If uncertain, say so explicitly
    """
```

---

## Detection Methods

### Multi-Layer Detection Architecture

```
                 ┌─────────────────────────────────────────┐
                 │           ENSEMBLE DETECTOR              │
                 │                                          │
                 │   Score = w1*Rule + w2*ML + w3*PHY      │
                 └────────────────┬────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
   ┌─────────────┐         ┌─────────────┐        ┌─────────────┐
   │ RULE ENGINE │         │  ML MODEL   │        │ PHY DETECTOR│
   │             │         │             │        │             │
   │ 11 physics  │         │ Isolation   │        │  DNN on RF  │
   │ based rules │         │ Forest      │        │ fingerprint │
   │             │         │             │        │             │
   │ Returns:    │         │ Returns:    │        │ Returns:    │
   │ - Rule IDs  │         │ - Anomaly   │        │ - Spoof     │
   │ - Severity  │         │   score     │        │   probability│
   │ - Evidence  │         │   [0-1]     │        │ - Embedding │
   └─────────────┘         └─────────────┘        └─────────────┘
```

### Attack Types Detected

| Attack Type | Detection Method | Indicators |
|-------------|------------------|------------|
| **Spoofed Track** | Rules + ML | Impossible kinematics, teleportation |
| **Ghost Aircraft** | PHY + ML | No consistent RF fingerprint |
| **Replay Attack** | PHY + Time | Duplicate embeddings, old timestamps |
| **Saturation** | Rules | Message burst rate |
| **GPS Spoofing** | Rules + ML | Position/velocity inconsistency |

---

## Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- RTL-SDR dongle (for live operation)

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/aerosentry-ai.git
cd aerosentry-ai/AI-Software-Defined-Radio

# Install dependencies
pip install -r requirements.txt

# For edge node only
pip install -r requirements-edge.txt

# For cloud services only
pip install -r requirements-cloud.txt
```

### Docker Installation

```bash
# Edge node
cd docker
docker compose -f docker-compose.edge.yml up -d

# Cloud services
docker compose -f docker-compose.cloud.yml up -d
```

---

## Configuration

### Environment Variables

Create `.env` file from template:

```bash
cp .env.example .env
```

**Edge Node Configuration:**

```env
# Sensor identification
SENSOR_ID=edge-001
LATITUDE=37.7749
LONGITUDE=-122.4194
ALTITUDE=100

# SDR settings
SDR_DEVICE=0
SDR_GAIN=autogain

# Cloud connection
CLOUD_ENDPOINT=https://your-cloud-server:8000
API_KEY=your-api-key

# NATS streaming
NATS_URL=nats://cloud-server:4222
```

**Cloud Configuration:**

```env
# Database
TIMESCALE_HOST=localhost
TIMESCALE_PORT=5432
TIMESCALE_DB=aerosentry
TIMESCALE_USER=aerosentry
TIMESCALE_PASSWORD=secure_password

# Object storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=aerosentry
MINIO_SECRET_KEY=secure_password

# LLM
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...
```

---

## Running the System

### Option 1: Docker Compose (Recommended)

**Start Edge Node:**
```bash
cd docker
docker compose -f docker-compose.edge.yml up -d
```

**Start Cloud Services:**
```bash
cd docker
docker compose -f docker-compose.cloud.yml up -d
```

**Start with Voice Capture:**
```bash
docker compose -f docker-compose.edge.yml --profile voice up -d
```

**Start with Mesh Relay:**
```bash
docker compose -f docker-compose.edge.yml --profile mesh up -d
```

### Option 2: Manual Execution

**Edge Node:**
```bash
# Start Beast source (readsb)
docker run -d --name readsb \
    -p 30005:30005 \
    -e READSB_DEVICE_TYPE=rtlsdr \
    ghcr.io/sdr-enthusiasts/docker-adsb-ultrafeeder

# Start edge processor
python -m edge.main
```

**Cloud Services:**
```bash
# Start TimescaleDB
docker run -d --name timescaledb \
    -p 5432:5432 \
    -e POSTGRES_DB=aerosentry \
    timescale/timescaledb:latest-pg15

# Initialize database
python scripts/setup_db.py

# Start API
python -m cloud.main
```

### Option 3: Kubernetes

```bash
# Apply all manifests
kubectl apply -k deploy/kubernetes/

# Or individual components
kubectl apply -f deploy/kubernetes/namespace.yaml
kubectl apply -f deploy/kubernetes/configmap.yaml
kubectl apply -f deploy/kubernetes/timescaledb.yaml
kubectl apply -f deploy/kubernetes/nats.yaml
kubectl apply -f deploy/kubernetes/ingest-api.yaml
```

---

## API Reference

### Ingest API (Port 8000)

**Health Check:**
```
GET /health
```

**Register Sensor:**
```
POST /sensors/register
{
    "sensor_id": "edge-001",
    "latitude": 37.7749,
    "longitude": -122.4194
}
```

**Ingest ADS-B Message:**
```
POST /ingest/adsb
{
    "timestamp": "2024-01-15T10:30:00Z",
    "sensor_id": "edge-001",
    "icao24": "abc123",
    "latitude": 37.75,
    "longitude": -122.42,
    "altitude": 35000,
    "velocity": 450
}
```

**Batch Ingest:**
```
POST /ingest/adsb/batch
{
    "messages": [...]
}
```

**Get Alerts:**
```
GET /alerts?limit=50&severity=high
```

**Get Tracks:**
```
GET /tracks?icao24=abc123
```

**WebSocket - Real-time Alerts:**
```
WS /ws/alerts
```

**WebSocket - Real-time Tracks:**
```
WS /ws/tracks
```

### LLM Copilot API (Port 8002)

**Natural Language Query:**
```
POST /query
{
    "query": "Show me all critical alerts in the last hour",
    "context": {}
}
```

**Generate Incident Report:**
```
POST /report
{
    "incident_id": "INC-2024-001",
    "start_time": "2024-01-15T10:00:00Z",
    "end_time": "2024-01-15T11:00:00Z",
    "icao24_list": ["abc123", "def456"]
}
```

### Model Serving API (Port 8001)

**Track Anomaly Prediction:**
```
POST /predict/track
{
    "icao24": "abc123",
    "avg_velocity": 450,
    "max_velocity": 480,
    ...
}
```

**PHY Spoofing Prediction:**
```
POST /predict/phy
{
    "icao24": "abc123",
    "cfo": 150.5,
    "amplitude_mean": 0.8,
    ...
}
```

---

## Database Schema

### Core Tables

**adsb_messages** - Raw ADS-B message storage (TimescaleDB hypertable)
```sql
CREATE TABLE adsb_messages (
    time            TIMESTAMPTZ NOT NULL,
    sensor_id       TEXT NOT NULL,
    icao24          TEXT NOT NULL,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    altitude        INTEGER,
    velocity        INTEGER,
    heading         REAL,
    callsign        TEXT,
    squawk          TEXT
);
```

**anomaly_alerts** - Detected anomalies
```sql
CREATE TABLE anomaly_alerts (
    time            TIMESTAMPTZ NOT NULL,
    alert_id        TEXT PRIMARY KEY,
    icao24          TEXT NOT NULL,
    alert_type      TEXT NOT NULL,
    severity        TEXT NOT NULL,
    anomaly_score   REAL,
    rule_triggers   JSONB,
    evidence        JSONB
);
```

**phy_features** - PHY-layer characteristics
```sql
CREATE TABLE phy_features (
    time            TIMESTAMPTZ NOT NULL,
    icao24          TEXT NOT NULL,
    cfo_mean        REAL,
    amp_mean        REAL,
    phase_std       REAL,
    ...
);
```

### Continuous Aggregates

**track_windows** - 30-second track statistics
```sql
SELECT
    time_bucket('30 seconds', time) AS bucket,
    icao24,
    AVG(velocity) AS avg_velocity,
    MAX(ABS(vert_rate)) AS max_vert_rate,
    COUNT(*) AS msg_count
FROM adsb_messages
GROUP BY bucket, icao24;
```

---

## Deployment Options

### Single Edge Node
Basic deployment with one SDR monitoring a single location.

### Multi-Sensor Network
Multiple edge nodes reporting to central cloud for wide-area coverage.

### Offline/Disaster Mode
When cloud connectivity is lost:
1. Data stored locally in SQLite
2. Summaries broadcast via Meshtastic LoRa mesh
3. Automatic sync when connectivity restored

### Kubernetes Cluster
Production deployment with:
- Auto-scaling ingest API
- High-availability TimescaleDB
- GPU-enabled model serving

---

## Testing

### Run Unit Tests
```bash
pytest tests/unit/ -v
```

### Run Integration Tests
```bash
pytest tests/integration/ -v
```

### Run Evaluation
```bash
python scripts/run_evaluation.py \
    --n-normal 100 \
    --n-attacks 40 \
    --output evaluation_report.md
```

### Test with Synthetic Data
```python
from shared.eval.synthetic_attacks import create_attack_dataset

dataset = create_attack_dataset(
    n_normal=100,
    n_spoofed=25,
    n_replay=25,
    n_ghost=25
)
```

---

## Legal Compliance

AeroSentry operates as a **receive-only** system:

- **FCC (US)**: No license required for passive reception
- **ETSI (EU)**: Compliant with EN 303 213
- **Privacy**: Voice transcripts follow GDPR data minimization
- **Data Retention**: Configurable, default 30 days

See `shared/docs/COMPLIANCE.md` for detailed compliance information.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest`
4. Submit a pull request

---

## License

[License information here]

---

## Support

For questions or issues, please open a GitHub issue or contact [support email].
