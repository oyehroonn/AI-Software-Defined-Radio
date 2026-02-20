# AeroSentry AI

**Community-deployable SDR sensor + AI layer for aviation RF intelligence**

Transform raw RF signals (ADS-B, ATC voice, spectrum) into real-time situational awareness, anomaly/attack detection, and offline disaster communications intelligence with evidence-linked alerts.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active_Development-yellow)

## Features

- **ADS-B Monitoring**: Real-time aircraft tracking with decoded position, velocity, and identification
- **Anomaly Detection**: Rule-based and ML-powered detection of spoofing, replay attacks, and kinematic anomalies
- **PHY-Layer Analysis**: RF fingerprinting for advanced spoofing detection
- **Voice Intelligence**: ATC communication capture with Whisper ASR transcription
- **RF Copilot**: LLM-powered query interface with evidence-gated responses
- **Offline Mode**: Local-first architecture with LoRa mesh relay for disaster scenarios

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Edge Node                                   │
├──────────────────┬──────────────────┬──────────────────────────────┤
│   SDR Hardware   │   IQ Capture     │   Voice Capture (VHF)        │
│   (RTL-SDR)      │   Service        │   Service                    │
├──────────────────┼──────────────────┼──────────────────────────────┤
│   readsb         │   Feature        │   Whisper ASR                │
│   Decoder        │   Extraction     │   Transcription              │
├──────────────────┴──────────────────┴──────────────────────────────┤
│                    Edge Anomaly Inference                           │
│              (Rules + IsolationForest + PHY DNN)                    │
├────────────────────────────────────────────────────────────────────┤
│                    Local Storage (SQLite/Parquet)                   │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼ gRPC/MQTT
┌─────────────────────────────────────────────────────────────────────┐
│                        Cloud Services                                │
├──────────────────┬──────────────────┬──────────────────────────────┤
│   Ingest API     │   NATS           │   TimescaleDB                │
│   (FastAPI)      │   JetStream      │   (Time-Series)              │
├──────────────────┼──────────────────┼──────────────────────────────┤
│   Model          │   LLM Copilot    │   Web UI                     │
│   Serving        │   (GPT-4/Claude) │   (React + Leaflet)          │
└──────────────────┴──────────────────┴──────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- RTL-SDR or compatible SDR hardware
- Python 3.11+

### Edge Node Deployment

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/aerosentry-ai.git
   cd aerosentry-ai
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your location and settings
   ```

3. **Start edge services**
   ```bash
   cd docker
   docker compose -f docker-compose.edge.yml up -d
   ```

4. **Access the UI**
   - tar1090 map: http://localhost:8080
   - AeroSentry API: http://localhost:8000

### Cloud Deployment

1. **Configure cloud environment**
   ```bash
   cp .env.cloud.example .env
   # Set database passwords, API keys, etc.
   ```

2. **Start cloud services**
   ```bash
   cd docker
   docker compose -f docker-compose.cloud.yml up -d
   ```

3. **Initialize database**
   ```bash
   docker exec -i aerosentry-timescaledb psql -U aerosentry < cloud/feature-store/schema.sql
   ```

## Configuration

### Edge Node (.env)

```bash
# Sensor identification
SENSOR_ID=edge-001
HOSTNAME=aerosentry-edge

# Location (required for position decoding)
LATITUDE=40.7128
LONGITUDE=-74.0060
ALTITUDE=10

# SDR settings
SDR_DEVICE=0
SDR_GAIN=autogain

# Cloud connection (optional)
CLOUD_ENDPOINT=https://your-cloud-server:8000
API_KEY=your-api-key
```

### Cloud Services (.env)

```bash
# Database
DB_USER=aerosentry
DB_PASSWORD=secure-password
DB_NAME=aerosentry

# Object storage
MINIO_USER=aerosentry
MINIO_PASSWORD=secure-password

# LLM (optional)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

## Project Structure

```
aerosentry-ai/
├── edge/
│   ├── iq-capture/          # GNU Radio IQ handling
│   ├── adsb-decode/         # readsb wrapper
│   ├── features/            # Feature extraction
│   ├── edge-inference/      # ML models
│   ├── edge-store/          # Local storage
│   └── voice-capture/       # VHF airband
├── cloud/
│   ├── ingest-api/          # FastAPI ingestion
│   ├── stream/              # NATS config
│   ├── feature-store/       # TimescaleDB schemas
│   ├── model-serving/       # ML deployment
│   ├── llm-copilot/         # Evidence-gated LLM
│   └── web-ui/              # Map interface
├── shared/
│   ├── schemas/             # Protobuf + configs
│   ├── eval/                # Benchmarks
│   └── docs/                # Documentation
├── disaster/
│   ├── mesh-relay/          # Meshtastic
│   └── local-summarizer/    # Offline AI
├── docker/                  # Docker Compose
├── deploy/                  # Kubernetes
└── tests/                   # Test suites
```

## Anomaly Detection

### Rule-Based Detection

| Rule | Description | Severity |
|------|-------------|----------|
| `impossible_speed` | Velocity > Mach 1.5 | HIGH |
| `teleport_detected` | Position implies > 2000 kts | CRITICAL |
| `extreme_climb` | Climb rate > 15000 ft/min | MEDIUM |
| `impossible_turn` | Turn rate > 10°/sec | HIGH |
| `message_burst` | Message rate > 5/sec | MEDIUM |

### ML Detection

- **IsolationForest**: Unsupervised anomaly detection on kinematic features
- **PHY DNN**: SODA-style deep network for RF fingerprint analysis
- **Ensemble**: Combined rule + ML scoring with calibrated uncertainty

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/sensors/register` | POST | Register sensor |
| `/ingest/adsb` | POST | Ingest ADS-B message |
| `/ingest/adsb/batch` | POST | Batch ingest |
| `/alerts` | GET/POST | List/create alerts |
| `/tracks` | GET | Get active tracks |
| `/tracks/{icao24}` | GET | Track history |

### WebSocket

- `/ws/alerts` - Real-time alert stream
- `/ws/tracks` - Real-time track updates

## Hardware Requirements

### Budget Edge Node (~$100)

| Component | Model | Cost |
|-----------|-------|------|
| SDR | RTL-SDR Blog V3 | $30 |
| Compute | Raspberry Pi 4 4GB | $55 |
| Antenna | 1090 MHz + filter | $15 |

### Performance Edge Node (~$350)

| Component | Model | Cost |
|-----------|-------|------|
| SDR | Airspy Mini | $99 |
| Compute | Intel N100 Mini PC | $150 |
| Antenna | Cavity filter + LNA | $50 |

## Legal Compliance

AeroSentry AI is designed as a **receive-only** system. See [COMPLIANCE.md](shared/docs/COMPLIANCE.md) for:

- FCC Part 15/97 compliance (USA)
- ETSI EN 302 208 (EU)
- Privacy considerations for voice capture
- Data retention policies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [SDR-Enthusiasts](https://github.com/sdr-enthusiasts) for ultrafeeder container
- [pyModeS](https://github.com/junzis/pyModeS) for ADS-B decoding
- [OpenSky Network](https://opensky-network.org/) for research data
- SODA paper authors for PHY detection inspiration
