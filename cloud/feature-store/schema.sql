-- AeroSentry AI - TimescaleDB Schema
-- Run this script to initialize the database

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ADS-B Messages table (hypertable for time-series data)
CREATE TABLE IF NOT EXISTS adsb_messages (
    time            TIMESTAMPTZ NOT NULL,
    sensor_id       TEXT NOT NULL,
    icao24          TEXT NOT NULL,
    df              SMALLINT,
    raw             TEXT,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    altitude        INTEGER,
    velocity        INTEGER,
    heading         REAL,
    vert_rate       INTEGER,
    callsign        TEXT,
    squawk          TEXT,
    signal_level    REAL
);

-- Convert to hypertable
SELECT create_hypertable('adsb_messages', 'time', if_not_exists => TRUE);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_adsb_icao ON adsb_messages (icao24, time DESC);
CREATE INDEX IF NOT EXISTS idx_adsb_sensor ON adsb_messages (sensor_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_adsb_callsign ON adsb_messages (callsign, time DESC) WHERE callsign IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_adsb_squawk ON adsb_messages (squawk, time DESC) WHERE squawk IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_adsb_position ON adsb_messages (latitude, longitude, time DESC) WHERE latitude IS NOT NULL;

-- Track windows continuous aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS track_windows
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('30 seconds', time) AS bucket,
    icao24,
    sensor_id,
    AVG(velocity) AS avg_velocity,
    STDDEV(velocity) AS std_velocity,
    MIN(velocity) AS min_velocity,
    MAX(velocity) AS max_velocity,
    AVG(altitude) AS avg_altitude,
    STDDEV(altitude) AS std_altitude,
    MIN(altitude) AS min_altitude,
    MAX(altitude) AS max_altitude,
    AVG(heading) AS avg_heading,
    AVG(vert_rate) AS avg_vert_rate,
    MAX(ABS(vert_rate)) AS max_vert_rate,
    COUNT(*) AS msg_count,
    MAX(callsign) AS callsign,
    MAX(squawk) AS squawk
FROM adsb_messages
WHERE latitude IS NOT NULL
GROUP BY bucket, icao24, sensor_id
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('track_windows',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '30 seconds',
    schedule_interval => INTERVAL '30 seconds',
    if_not_exists => TRUE
);

-- Anomaly alerts table
CREATE TABLE IF NOT EXISTS anomaly_alerts (
    time            TIMESTAMPTZ NOT NULL,
    alert_id        TEXT PRIMARY KEY,
    sensor_id       TEXT NOT NULL,
    icao24          TEXT NOT NULL,
    callsign        TEXT,
    alert_type      TEXT NOT NULL,
    severity        TEXT NOT NULL,
    anomaly_score   REAL,
    rule_triggers   JSONB,
    evidence        JSONB,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    acknowledged    BOOLEAN DEFAULT FALSE,
    resolved        BOOLEAN DEFAULT FALSE
);

SELECT create_hypertable('anomaly_alerts', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_alerts_icao ON anomaly_alerts (icao24, time DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON anomaly_alerts (severity, time DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_unresolved ON anomaly_alerts (time DESC) WHERE resolved = FALSE;

-- PHY features table
CREATE TABLE IF NOT EXISTS phy_features (
    time            TIMESTAMPTZ NOT NULL,
    sensor_id       TEXT NOT NULL,
    icao24          TEXT NOT NULL,
    cfo_mean        REAL,
    cfo_std         REAL,
    cfo_drift       REAL,
    amp_mean        REAL,
    amp_std         REAL,
    amp_skew        REAL,
    preamble_rise_time REAL,
    preamble_overshoot REAL,
    phase_std       REAL,
    phase_jitter    REAL,
    spectral_centroid REAL,
    spectral_spread REAL
);

SELECT create_hypertable('phy_features', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_phy_icao ON phy_features (icao24, time DESC);

-- Voice transcripts table
CREATE TABLE IF NOT EXISTS voice_transcripts (
    time            TIMESTAMPTZ NOT NULL,
    transcript_id   TEXT PRIMARY KEY,
    sensor_id       TEXT NOT NULL,
    frequency_mhz   REAL,
    duration_sec    REAL,
    text            TEXT,
    confidence      REAL,
    callsigns       TEXT[],
    runways         TEXT[],
    altitudes       INTEGER[],
    audio_path      TEXT
);

SELECT create_hypertable('voice_transcripts', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_voice_callsigns ON voice_transcripts USING GIN (callsigns);

-- Sensors registry
CREATE TABLE IF NOT EXISTS sensors (
    sensor_id       TEXT PRIMARY KEY,
    name            TEXT,
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    altitude_m      REAL,
    last_seen       TIMESTAMPTZ,
    status          TEXT DEFAULT 'unknown',
    config          JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Retention policies
SELECT add_retention_policy('adsb_messages', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('phy_features', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('voice_transcripts', INTERVAL '30 days', if_not_exists => TRUE);

-- Hourly message statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    sensor_id,
    COUNT(*) AS message_count,
    COUNT(DISTINCT icao24) AS unique_aircraft,
    AVG(signal_level) AS avg_signal_level
FROM adsb_messages
GROUP BY bucket, sensor_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('hourly_stats',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);
