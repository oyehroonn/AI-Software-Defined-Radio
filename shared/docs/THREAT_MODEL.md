# AeroSentry AI - Threat Model

## Overview

This document defines the threat landscape for ADS-B and aviation RF systems that AeroSentry AI is designed to detect and mitigate.

## Threat Categories

### 1. Spoofing Attacks

**Description**: Injection of fake aircraft signals via RF transmission.

**Attack Variants**:
- **Ghost Aircraft**: Create phantom aircraft that don't exist
- **Position Manipulation**: Broadcast false positions for real aircraft
- **Identity Cloning**: Use legitimate ICAO24 addresses with false data

**Detection Approaches**:
- Kinematic feasibility checks (impossible maneuvers)
- Multi-sensor triangulation inconsistencies
- PHY-layer RF fingerprinting
- Message rate anomalies

**Severity**: CRITICAL

### 2. Replay Attacks

**Description**: Re-broadcast of captured legitimate ADS-B messages at a later time.

**Characteristics**:
- Valid CRC checksums
- Plausible kinematic data
- Stale temporal context

**Detection Approaches**:
- Timestamp correlation with expected flight schedules
- Position prediction vs actual trajectory
- Flight database cross-reference (FlightAware, FR24)

**Severity**: HIGH

### 3. Saturation/Denial of Service

**Description**: Flooding receivers with excessive messages to overwhelm processing.

**Attack Variants**:
- **Message Flood**: High-rate transmission of valid messages
- **Noise Injection**: Broadband interference on 1090 MHz
- **Selective Jamming**: Targeted interference during critical phases

**Detection Approaches**:
- Message rate anomaly detection
- AGC/noise floor monitoring
- Multi-receiver correlation

**Severity**: HIGH

### 4. Ghost Aircraft

**Description**: Brief, inconsistent tracks designed to create confusion.

**Characteristics**:
- Short track duration (<60 seconds)
- Kinematic inconsistencies
- No correlation with flight databases
- Random or nonsensical callsigns

**Detection Approaches**:
- Track continuity analysis
- Callsign validation
- Message rate profiling

**Severity**: MEDIUM

### 5. RF Interference

**Description**: Unintentional or intentional degradation of signal quality.

**Sources**:
- Adjacent band interference
- Intermodulation products
- Equipment malfunction
- Deliberate jamming

**Detection Approaches**:
- SNR monitoring
- Decode success rate tracking
- Spectrum analysis

**Severity**: MEDIUM

### 6. Edge Node Compromise

**Description**: Malicious actor gains control of AeroSentry edge sensor.

**Attack Goals**:
- Inject false alerts
- Suppress legitimate alerts
- Exfiltrate data
- Pivot to cloud infrastructure

**Mitigation**:
- Signed edge software
- Cryptographic attestation
- Multi-sensor consensus requirements
- Network segmentation

**Severity**: HIGH

## Attacker Profiles

### Script Kiddie
- **Capability**: Off-the-shelf SDR, public tools
- **Motivation**: Curiosity, notoriety
- **Likely Attacks**: Basic spoofing, ghost aircraft

### Sophisticated Hobbyist
- **Capability**: Custom SDR setups, programming skills
- **Motivation**: Research, proof of concept
- **Likely Attacks**: Replay attacks, advanced spoofing

### State Actor
- **Capability**: Military-grade equipment, inside knowledge
- **Motivation**: Espionage, strategic disruption
- **Likely Attacks**: Coordinated multi-vector, infrastructure compromise

### Terrorist/Criminal
- **Capability**: Moderate to high
- **Motivation**: Create aviation incidents, extortion
- **Likely Attacks**: Targeted spoofing during critical flight phases

## Risk Matrix

| Threat | Likelihood | Impact | Risk Score |
|--------|------------|--------|------------|
| Spoofing (basic) | Medium | Critical | HIGH |
| Spoofing (advanced) | Low | Critical | MEDIUM |
| Replay Attack | Medium | High | MEDIUM |
| Saturation DoS | Low | High | LOW |
| Ghost Aircraft | High | Medium | MEDIUM |
| RF Interference | Medium | Medium | LOW |
| Edge Node Compromise | Low | High | LOW |

## Detection Confidence Levels

| Level | Description | Action |
|-------|-------------|--------|
| **Confirmed** | Multiple independent indicators | Immediate alert, notify authorities |
| **High Confidence** | Strong ML score + rule triggers | Alert operator, log evidence |
| **Suspicious** | Single indicator or weak signal | Log for analysis, passive monitoring |
| **Uncertain** | Inconclusive evidence | Continue monitoring, no alert |

## Response Procedures

### Confirmed Spoofing
1. Generate high-priority alert
2. Preserve all evidence (IQ samples, track data, PHY features)
3. Cross-reference with other sensors if available
4. Notify appropriate authorities if safety-critical
5. Export incident report

### Suspected Replay
1. Validate against flight databases
2. Check for temporal anomalies
3. Escalate if confirmed
4. Update detection models

### DoS/Interference
1. Switch to backup receiver if available
2. Log interference characteristics
3. Notify operators
4. Continue degraded operation
