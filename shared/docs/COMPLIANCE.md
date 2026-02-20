# AeroSentry AI - Legal and Compliance Framework

## Receive-Only Architecture Guarantee

AeroSentry AI is designed as a **receive-only** system. The architecture explicitly prohibits any RF transmission capability:

1. **No TX Code Paths**: The codebase contains no functions or methods that enable RF transmission
2. **Hardware Configuration**: SDR devices are configured in receive-only mode
3. **Code Review Policy**: All PRs must be reviewed for potential TX capability before merge

## Regional Compliance

### United States (FCC)

- **Part 15**: Receive-only devices are exempt from licensing requirements
- **Part 97**: Amateur radio operators may receive on any frequency
- **1090 MHz Reception**: Legal for passive monitoring without license
- **VHF Airband (118-137 MHz)**: Legal to monitor; transmission prohibited without license

### European Union (ETSI)

- **EN 302 208**: Compliance for RF receive equipment
- **RED Directive**: Receive-only equipment exempt from conformity assessment
- **Member State Variations**: Check local regulations for specific restrictions

### Privacy Considerations

#### ADS-B Data
- Aircraft positions are publicly transmitted and not considered PII
- ICAO24 addresses may be correlated to registration databases
- Do not store flight data beyond operational necessity

#### Voice Recordings
- **GDPR (EU)**: Voice recordings may contain PII
  - Implement automatic PII redaction in transcripts
  - Data retention limited to operational necessity (default: 24 hours)
  - Provide deletion mechanisms on request
- **USA**: Check state-level wiretapping laws
  - Federal law permits one-party consent for recording
  - ATC transmissions are generally not protected

## Data Retention Policies

| Data Type | Default Retention | Notes |
|-----------|------------------|-------|
| ADS-B Messages | 30 days | Aggregated statistics kept indefinitely |
| Voice Recordings | 24 hours | Extended to 7 days if anomaly detected |
| Voice Transcripts | 30 days | PII redacted |
| Anomaly Alerts | 1 year | Full evidence preserved |
| PHY Features | 30 days | Anonymized after 7 days |

## Export Control

This software processes publicly available RF signals and does not fall under:
- ITAR (International Traffic in Arms Regulations)
- EAR (Export Administration Regulations)

However, users integrating with military systems should consult legal counsel.

## Responsible Disclosure

If security vulnerabilities are discovered that could enable:
- Spoofing of aviation signals
- Interference with ATC communications
- Compromise of aviation safety systems

Please report responsibly to: security@aerosentry.example.com

Do NOT publish exploit code publicly.
