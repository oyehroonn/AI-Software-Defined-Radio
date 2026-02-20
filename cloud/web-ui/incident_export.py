"""Incident report export functionality."""

import io
import json
import zipfile
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class IncidentExport:
    """Incident export package."""
    incident_id: str
    timestamp: datetime
    summary: str
    alerts: list[dict]
    tracks: list[dict]
    phy_features: list[dict]
    voice_transcripts: list[dict]
    metadata: dict


class IncidentExporter:
    """Export incidents to various formats."""
    
    def __init__(self, db_connection=None):
        self.db = db_connection
        
    async def gather_incident_data(
        self,
        incident_id: str,
        start_time: datetime,
        end_time: datetime,
        icao24_list: Optional[list[str]] = None
    ) -> IncidentExport:
        """Gather all data related to an incident."""
        
        # Fetch alerts
        alerts = await self._fetch_alerts(start_time, end_time, icao24_list)
        
        # Fetch track data
        tracks = await self._fetch_tracks(start_time, end_time, icao24_list)
        
        # Fetch PHY features
        phy_features = await self._fetch_phy_features(start_time, end_time, icao24_list)
        
        # Fetch voice transcripts
        voice_transcripts = await self._fetch_voice(start_time, end_time)
        
        return IncidentExport(
            incident_id=incident_id,
            timestamp=datetime.now(UTC),
            summary=f"Incident {incident_id} from {start_time} to {end_time}",
            alerts=alerts,
            tracks=tracks,
            phy_features=phy_features,
            voice_transcripts=voice_transcripts,
            metadata={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "aircraft": icao24_list or [],
                "alert_count": len(alerts),
                "track_points": len(tracks)
            }
        )
        
    async def _fetch_alerts(
        self,
        start_time: datetime,
        end_time: datetime,
        icao24_list: Optional[list[str]]
    ) -> list[dict]:
        """Fetch alerts from database."""
        if not self.db:
            return []
            
        query = """
            SELECT * FROM anomaly_alerts 
            WHERE timestamp BETWEEN $1 AND $2
        """
        params = [start_time, end_time]
        
        if icao24_list:
            query += " AND icao24 = ANY($3)"
            params.append(icao24_list)
            
        query += " ORDER BY timestamp"
        
        try:
            rows = await self.db.fetch(query, *params)
            return [dict(r) for r in rows]
        except Exception:
            return []
            
    async def _fetch_tracks(
        self,
        start_time: datetime,
        end_time: datetime,
        icao24_list: Optional[list[str]]
    ) -> list[dict]:
        """Fetch track data from database."""
        if not self.db:
            return []
            
        query = """
            SELECT * FROM adsb_messages 
            WHERE timestamp BETWEEN $1 AND $2
        """
        params = [start_time, end_time]
        
        if icao24_list:
            query += " AND icao24 = ANY($3)"
            params.append(icao24_list)
            
        query += " ORDER BY timestamp"
        
        try:
            rows = await self.db.fetch(query, *params)
            return [dict(r) for r in rows]
        except Exception:
            return []
            
    async def _fetch_phy_features(
        self,
        start_time: datetime,
        end_time: datetime,
        icao24_list: Optional[list[str]]
    ) -> list[dict]:
        """Fetch PHY features from database."""
        if not self.db:
            return []
            
        query = """
            SELECT * FROM phy_features 
            WHERE timestamp BETWEEN $1 AND $2
        """
        params = [start_time, end_time]
        
        if icao24_list:
            query += " AND icao24 = ANY($3)"
            params.append(icao24_list)
            
        query += " ORDER BY timestamp"
        
        try:
            rows = await self.db.fetch(query, *params)
            return [dict(r) for r in rows]
        except Exception:
            return []
            
    async def _fetch_voice(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> list[dict]:
        """Fetch voice transcripts from database."""
        if not self.db:
            return []
            
        query = """
            SELECT * FROM voice_transcripts 
            WHERE timestamp BETWEEN $1 AND $2
            ORDER BY timestamp
        """
        
        try:
            rows = await self.db.fetch(query, start_time, end_time)
            return [dict(r) for r in rows]
        except Exception:
            return []
            
    def export_to_json(self, incident: IncidentExport) -> str:
        """Export incident to JSON string."""
        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
        return json.dumps(asdict(incident), default=serialize, indent=2)
        
    def export_to_csv_zip(self, incident: IncidentExport) -> bytes:
        """Export incident to ZIP file with CSVs."""
        buffer = io.BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Metadata
            zf.writestr(
                "metadata.json",
                json.dumps(incident.metadata, indent=2, default=str)
            )
            
            # Summary
            zf.writestr(
                "summary.txt",
                f"""Incident Report: {incident.incident_id}
Generated: {incident.timestamp.isoformat()}

{incident.summary}

Statistics:
- Alerts: {len(incident.alerts)}
- Track Points: {len(incident.tracks)}
- PHY Features: {len(incident.phy_features)}
- Voice Transcripts: {len(incident.voice_transcripts)}
"""
            )
            
            # Export as CSV if pandas available
            if PANDAS_AVAILABLE:
                if incident.alerts:
                    df = pd.DataFrame(incident.alerts)
                    zf.writestr("alerts.csv", df.to_csv(index=False))
                    
                if incident.tracks:
                    df = pd.DataFrame(incident.tracks)
                    zf.writestr("tracks.csv", df.to_csv(index=False))
                    
                if incident.phy_features:
                    df = pd.DataFrame(incident.phy_features)
                    zf.writestr("phy_features.csv", df.to_csv(index=False))
                    
                if incident.voice_transcripts:
                    df = pd.DataFrame(incident.voice_transcripts)
                    zf.writestr("voice_transcripts.csv", df.to_csv(index=False))
            else:
                # Fallback to JSON
                zf.writestr("alerts.json", json.dumps(incident.alerts, default=str))
                zf.writestr("tracks.json", json.dumps(incident.tracks, default=str))
                zf.writestr("phy_features.json", json.dumps(incident.phy_features, default=str))
                zf.writestr("voice_transcripts.json", json.dumps(incident.voice_transcripts, default=str))
                
        buffer.seek(0)
        return buffer.read()
        
    def export_to_kml(self, incident: IncidentExport) -> str:
        """Export track data to KML format for Google Earth."""
        
        kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>AeroSentry Incident: {}</name>
    <description>{}</description>
""".format(incident.incident_id, incident.summary)

        kml_footer = """</Document>
</kml>"""

        # Style definitions
        styles = """
    <Style id="normalTrack">
        <LineStyle>
            <color>ff00ff00</color>
            <width>2</width>
        </LineStyle>
    </Style>
    <Style id="alertTrack">
        <LineStyle>
            <color>ff0000ff</color>
            <width>3</width>
        </LineStyle>
    </Style>
    <Style id="alertIcon">
        <IconStyle>
            <color>ff0000ff</color>
            <scale>1.2</scale>
            <Icon>
                <href>http://maps.google.com/mapfiles/kml/shapes/caution.png</href>
            </Icon>
        </IconStyle>
    </Style>
"""

        # Group tracks by ICAO24
        tracks_by_icao: dict[str, list[dict]] = {}
        for point in incident.tracks:
            icao = point.get("icao24", "unknown")
            if icao not in tracks_by_icao:
                tracks_by_icao[icao] = []
            tracks_by_icao[icao].append(point)
            
        # Generate placemarks
        placemarks = []
        
        for icao, points in tracks_by_icao.items():
            # Sort by timestamp
            points.sort(key=lambda x: x.get("timestamp", ""))
            
            # Create track line
            coords = []
            for p in points:
                lat = p.get("latitude")
                lon = p.get("longitude")
                alt = p.get("altitude", 0)
                if lat and lon:
                    coords.append(f"{lon},{lat},{alt}")
                    
            if coords:
                placemark = f"""
    <Placemark>
        <name>Track: {icao}</name>
        <description>Callsign: {points[0].get('callsign', 'Unknown')}</description>
        <styleUrl>#normalTrack</styleUrl>
        <LineString>
            <altitudeMode>absolute</altitudeMode>
            <coordinates>{' '.join(coords)}</coordinates>
        </LineString>
    </Placemark>"""
                placemarks.append(placemark)
                
        # Add alert markers
        for alert in incident.alerts:
            lat = alert.get("latitude")
            lon = alert.get("longitude")
            if lat and lon:
                placemark = f"""
    <Placemark>
        <name>Alert: {alert.get('alert_type', 'Unknown')}</name>
        <description>
            ICAO24: {alert.get('icao24', 'Unknown')}
            Severity: {alert.get('severity', 'Unknown')}
            Time: {alert.get('timestamp', 'Unknown')}
        </description>
        <styleUrl>#alertIcon</styleUrl>
        <Point>
            <coordinates>{lon},{lat},0</coordinates>
        </Point>
    </Placemark>"""
                placemarks.append(placemark)
                
        return kml_header + styles + ''.join(placemarks) + kml_footer
