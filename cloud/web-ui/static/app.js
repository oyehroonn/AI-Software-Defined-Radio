/**
 * AeroSentry AI - Web UI Application
 * Real-time aircraft tracking and anomaly visualization
 */

class AeroSentryApp {
    constructor() {
        this.map = null;
        this.aircraftMarkers = new Map();
        this.alertMarkers = new Map();
        this.tracks = new Map();
        this.wsAlerts = null;
        this.wsTracks = null;
        this.selectedAircraft = null;
        
        this.init();
    }
    
    init() {
        this.initMap();
        this.initWebSockets();
        this.initUI();
        this.loadInitialData();
    }
    
    initMap() {
        // Initialize Leaflet map
        this.map = L.map('map').setView([40.7128, -74.0060], 8);
        
        // Add dark tile layer
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '© OpenStreetMap © CARTO',
            maxZoom: 19
        }).addTo(this.map);
        
        // Aircraft icon
        this.aircraftIcon = L.divIcon({
            className: 'aircraft-icon',
            html: '<div class="aircraft-marker"></div>',
            iconSize: [20, 20]
        });
        
        // Alert icon
        this.alertIcon = L.divIcon({
            className: 'alert-icon',
            html: '<div class="alert-marker"></div>',
            iconSize: [24, 24]
        });
    }
    
    initWebSockets() {
        // Connect to alerts WebSocket
        const wsUrl = `ws://${window.location.host}/ws/alerts`;
        this.wsAlerts = new WebSocket(wsUrl);
        
        this.wsAlerts.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'alert') {
                this.handleAlert(data.data);
            }
        };
        
        this.wsAlerts.onclose = () => {
            console.log('Alerts WebSocket closed, reconnecting...');
            setTimeout(() => this.initWebSockets(), 5000);
        };
        
        // Connect to tracks WebSocket
        const tracksUrl = `ws://${window.location.host}/ws/tracks`;
        this.wsTracks = new WebSocket(tracksUrl);
        
        this.wsTracks.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'tracks') {
                this.updateTracks(data.data);
            }
        };
    }
    
    initUI() {
        // Search functionality
        document.getElementById('search-input').addEventListener('input', (e) => {
            this.filterAircraft(e.target.value);
        });
        
        // Severity filters
        document.querySelectorAll('.severity-filter').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.target.classList.toggle('active');
                this.applyFilters();
            });
        });
        
        // Query input
        document.getElementById('query-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.executeQuery(e.target.value);
            }
        });
    }
    
    async loadInitialData() {
        try {
            // Load current tracks
            const tracksRes = await fetch('/tracks?limit=200');
            const tracksData = await tracksRes.json();
            this.updateTracks(tracksData.tracks);
            
            // Load recent alerts
            const alertsRes = await fetch('/alerts?limit=50');
            const alertsData = await alertsRes.json();
            this.updateAlertList(alertsData.alerts);
            
            // Load stats
            const statsRes = await fetch('/stats');
            const statsData = await statsRes.json();
            this.updateStats(statsData);
        } catch (error) {
            console.error('Failed to load initial data:', error);
        }
    }
    
    updateTracks(tracks) {
        const now = Date.now();
        
        tracks.forEach(track => {
            if (!track.latitude || !track.longitude) return;
            
            const icao24 = track.icao24;
            const position = [track.latitude, track.longitude];
            
            if (this.aircraftMarkers.has(icao24)) {
                // Update existing marker
                const marker = this.aircraftMarkers.get(icao24);
                marker.setLatLng(position);
                marker.setRotationAngle(track.heading || 0);
            } else {
                // Create new marker
                const marker = L.marker(position, {
                    icon: this.aircraftIcon,
                    rotationAngle: track.heading || 0
                }).addTo(this.map);
                
                marker.bindPopup(this.createPopupContent(track));
                marker.on('click', () => this.selectAircraft(icao24));
                
                this.aircraftMarkers.set(icao24, marker);
            }
            
            // Update track data
            this.tracks.set(icao24, { ...track, lastUpdate: now });
        });
        
        // Update aircraft list
        this.updateAircraftList();
        
        // Remove stale tracks (>60 seconds old)
        this.aircraftMarkers.forEach((marker, icao24) => {
            const track = this.tracks.get(icao24);
            if (!track || now - track.lastUpdate > 60000) {
                marker.remove();
                this.aircraftMarkers.delete(icao24);
                this.tracks.delete(icao24);
            }
        });
    }
    
    createPopupContent(track) {
        return `
            <div class="aircraft-popup">
                <h4>${track.callsign || track.icao24}</h4>
                <table>
                    <tr><td>ICAO24:</td><td>${track.icao24}</td></tr>
                    <tr><td>Altitude:</td><td>${track.altitude || 'N/A'} ft</td></tr>
                    <tr><td>Speed:</td><td>${track.velocity || 'N/A'} kts</td></tr>
                    <tr><td>Heading:</td><td>${track.heading?.toFixed(0) || 'N/A'}°</td></tr>
                </table>
            </div>
        `;
    }
    
    updateAircraftList() {
        const listEl = document.getElementById('aircraft-list');
        const tracks = Array.from(this.tracks.values())
            .sort((a, b) => (b.altitude || 0) - (a.altitude || 0));
        
        listEl.innerHTML = tracks.map(track => `
            <div class="aircraft-item ${this.selectedAircraft === track.icao24 ? 'selected' : ''}"
                 onclick="app.selectAircraft('${track.icao24}')">
                <div class="aircraft-callsign">${track.callsign || track.icao24}</div>
                <div class="aircraft-details">
                    ${track.altitude || '?'} ft • ${track.velocity || '?'} kts
                </div>
            </div>
        `).join('');
    }
    
    handleAlert(alert) {
        // Add to alerts list
        this.addAlertToList(alert);
        
        // Add marker on map if position available
        if (alert.latitude && alert.longitude) {
            const marker = L.marker([alert.latitude, alert.longitude], {
                icon: this.getAlertIcon(alert.severity)
            }).addTo(this.map);
            
            marker.bindPopup(this.createAlertPopup(alert));
            this.alertMarkers.set(alert.alert_id, marker);
            
            // Remove marker after 5 minutes
            setTimeout(() => {
                marker.remove();
                this.alertMarkers.delete(alert.alert_id);
            }, 300000);
        }
        
        // Show notification for high/critical alerts
        if (['high', 'critical'].includes(alert.severity)) {
            this.showNotification(alert);
        }
    }
    
    getAlertIcon(severity) {
        const colors = {
            critical: '#ff0000',
            high: '#ff6600',
            medium: '#ffcc00',
            low: '#00cc00'
        };
        
        return L.divIcon({
            className: 'alert-icon',
            html: `<div class="alert-marker" style="background-color: ${colors[severity] || '#ffffff'}"></div>`,
            iconSize: [24, 24]
        });
    }
    
    createAlertPopup(alert) {
        return `
            <div class="alert-popup severity-${alert.severity}">
                <h4>⚠️ ${alert.alert_type}</h4>
                <p><strong>Severity:</strong> ${alert.severity.toUpperCase()}</p>
                <p><strong>Aircraft:</strong> ${alert.callsign || alert.icao24}</p>
                <p><strong>Score:</strong> ${alert.anomaly_score?.toFixed(3) || 'N/A'}</p>
                <p><strong>Time:</strong> ${new Date(alert.timestamp).toLocaleTimeString()}</p>
            </div>
        `;
    }
    
    addAlertToList(alert) {
        const listEl = document.getElementById('alerts-list');
        const alertHtml = `
            <div class="alert-item severity-${alert.severity}" data-alert-id="${alert.alert_id}">
                <div class="alert-header">
                    <span class="alert-severity">${alert.severity.toUpperCase()}</span>
                    <span class="alert-time">${new Date(alert.timestamp).toLocaleTimeString()}</span>
                </div>
                <div class="alert-type">${alert.alert_type}</div>
                <div class="alert-aircraft">${alert.callsign || alert.icao24}</div>
            </div>
        `;
        
        listEl.insertAdjacentHTML('afterbegin', alertHtml);
        
        // Limit list to 50 items
        while (listEl.children.length > 50) {
            listEl.lastChild.remove();
        }
    }
    
    updateAlertList(alerts) {
        const listEl = document.getElementById('alerts-list');
        listEl.innerHTML = '';
        
        alerts.forEach(alert => this.addAlertToList(alert));
    }
    
    updateStats(stats) {
        document.getElementById('stat-aircraft').textContent = stats.unique_aircraft || 0;
        document.getElementById('stat-messages').textContent = stats.messages_in_memory || 0;
        document.getElementById('stat-alerts').textContent = stats.alerts_count || 0;
        document.getElementById('stat-sensors').textContent = stats.active_sensors || 0;
    }
    
    selectAircraft(icao24) {
        this.selectedAircraft = icao24;
        
        // Update UI
        this.updateAircraftList();
        
        // Center map on aircraft
        const track = this.tracks.get(icao24);
        if (track && track.latitude && track.longitude) {
            this.map.setView([track.latitude, track.longitude], 10);
            
            // Open popup
            const marker = this.aircraftMarkers.get(icao24);
            if (marker) {
                marker.openPopup();
            }
        }
        
        // Load aircraft details
        this.loadAircraftDetails(icao24);
    }
    
    async loadAircraftDetails(icao24) {
        try {
            const [trackRes, alertsRes] = await Promise.all([
                fetch(`/tracks/${icao24}?limit=100`),
                fetch(`/alerts?icao24=${icao24}&limit=10`)
            ]);
            
            const trackData = await trackRes.json();
            const alertsData = await alertsRes.json();
            
            this.showAircraftPanel(trackData, alertsData.alerts);
        } catch (error) {
            console.error('Failed to load aircraft details:', error);
        }
    }
    
    showAircraftPanel(trackData, alerts) {
        const panelEl = document.getElementById('aircraft-panel');
        panelEl.innerHTML = `
            <div class="panel-header">
                <h3>${trackData.icao24}</h3>
                <button onclick="app.closeAircraftPanel()">×</button>
            </div>
            <div class="panel-body">
                <p><strong>Messages:</strong> ${trackData.message_count}</p>
                ${alerts.length > 0 ? `
                    <h4>Recent Alerts</h4>
                    <ul>
                        ${alerts.map(a => `<li class="severity-${a.severity}">${a.alert_type}</li>`).join('')}
                    </ul>
                ` : '<p>No alerts for this aircraft</p>'}
            </div>
        `;
        panelEl.classList.add('visible');
    }
    
    closeAircraftPanel() {
        document.getElementById('aircraft-panel').classList.remove('visible');
        this.selectedAircraft = null;
        this.updateAircraftList();
    }
    
    filterAircraft(query) {
        query = query.toLowerCase();
        
        document.querySelectorAll('.aircraft-item').forEach(item => {
            const text = item.textContent.toLowerCase();
            item.style.display = text.includes(query) ? '' : 'none';
        });
    }
    
    applyFilters() {
        // Implementation for severity filters
    }
    
    async executeQuery(query) {
        try {
            const res = await fetch('/api/copilot/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            
            const data = await res.json();
            this.showQueryResult(data);
        } catch (error) {
            console.error('Query failed:', error);
        }
    }
    
    showQueryResult(result) {
        const resultEl = document.getElementById('query-result');
        resultEl.innerHTML = `
            <div class="query-response">
                <p>${result.response || result.text || 'No response'}</p>
                ${result.sql ? `<code>${result.sql}</code>` : ''}
            </div>
        `;
    }
    
    showNotification(alert) {
        if (Notification.permission === 'granted') {
            new Notification(`AeroSentry Alert: ${alert.alert_type}`, {
                body: `${alert.severity.toUpperCase()} - ${alert.callsign || alert.icao24}`,
                icon: '/static/icon.png'
            });
        }
    }
}

// Initialize app
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new AeroSentryApp();
    
    // Request notification permission
    if ('Notification' in window) {
        Notification.requestPermission();
    }
});
