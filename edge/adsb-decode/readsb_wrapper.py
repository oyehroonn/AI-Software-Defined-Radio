"""Wrapper for readsb ADS-B decoder configuration."""

import logging
import subprocess
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ReadsbConfig:
    """Configuration for readsb decoder."""
    
    # Device settings
    device_type: str = "rtlsdr"
    device_index: int = 0
    gain: str = "autogain"
    ppm: int = 0
    
    # Location
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: int = 0
    
    # Network ports
    beast_port: int = 30005
    sbs_port: int = 30003
    json_port: int = 30152
    
    # Output paths
    json_dir: str = "/run/readsb"
    globe_history_dir: str = "/var/globe_history"
    
    # Features
    enable_biastee: bool = False
    fix_df: bool = True
    aggressive: bool = False
    
    # Advanced
    max_range: int = 300  # nautical miles
    
    def to_args(self) -> list[str]:
        """Convert config to readsb command line arguments."""
        args = [
            f"--device-type={self.device_type}",
            f"--{self.device_type}-device={self.device_index}",
            f"--lat={self.latitude}",
            f"--lon={self.longitude}",
            f"--max-range={self.max_range}",
            f"--net-beast-port={self.beast_port}",
            f"--net-sbs-port={self.sbs_port}",
            f"--json-location-accuracy=2",
            "--net",
            "--net-connector=localhost,30104,beast_in",
        ]
        
        # Gain setting
        if self.gain == "autogain":
            args.append("--gain=-10")  # -10 = autogain in readsb
        else:
            args.append(f"--gain={self.gain}")
            
        # PPM correction
        if self.ppm != 0:
            args.append(f"--ppm={self.ppm}")
            
        # Altitude
        if self.altitude > 0:
            args.append(f"--altitude={self.altitude}")
            
        # Features
        if self.enable_biastee:
            args.append("--enable-biastee")
        if self.fix_df:
            args.append("--fix-df")
        if self.aggressive:
            args.append("--aggressive")
            
        # Output directories
        if self.json_dir:
            args.append(f"--write-json={self.json_dir}")
            args.append("--write-json-every=1")
            
        if self.globe_history_dir:
            args.append(f"--write-globe-history={self.globe_history_dir}")
            
        return args
        
    def to_env(self) -> dict[str, str]:
        """Convert config to environment variables for Docker."""
        env = {
            "READSB_DEVICE_TYPE": self.device_type,
            f"READSB_{self.device_type.upper()}_DEVICE": str(self.device_index),
            "READSB_LAT": str(self.latitude),
            "READSB_LON": str(self.longitude),
            "READSB_GAIN": self.gain,
            "READSB_MAX_RANGE": str(self.max_range),
        }
        
        if self.altitude > 0:
            env["READSB_ALT"] = str(self.altitude)
        if self.ppm != 0:
            env["READSB_PPM"] = str(self.ppm)
        if self.enable_biastee:
            env["READSB_ENABLE_BIASTEE"] = "true"
            
        return env


class ReadsbWrapper:
    """Wrapper for running and managing readsb decoder."""
    
    def __init__(self, config: Optional[ReadsbConfig] = None):
        self.config = config or ReadsbConfig()
        self.process: Optional[subprocess.Popen] = None
        
    def find_readsb(self) -> Optional[str]:
        """Find readsb binary."""
        paths = [
            "/usr/bin/readsb",
            "/usr/local/bin/readsb",
            "/opt/readsb/readsb"
        ]
        
        for path in paths:
            if Path(path).exists():
                return path
                
        # Try which
        try:
            result = subprocess.run(
                ["which", "readsb"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
            
        return None
        
    def start(self) -> bool:
        """Start readsb process."""
        binary = self.find_readsb()
        
        if not binary:
            logger.error("readsb binary not found")
            return False
            
        # Ensure output directories exist
        if self.config.json_dir:
            Path(self.config.json_dir).mkdir(parents=True, exist_ok=True)
        if self.config.globe_history_dir:
            Path(self.config.globe_history_dir).mkdir(parents=True, exist_ok=True)
            
        cmd = [binary] + self.config.to_args()
        
        logger.info(f"Starting readsb: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, **self.config.to_env()}
            )
            
            # Check if process started successfully
            try:
                self.process.wait(timeout=2)
                # Process exited quickly, likely an error
                stderr = self.process.stderr.read().decode()
                logger.error(f"readsb failed to start: {stderr}")
                return False
            except subprocess.TimeoutExpired:
                # Process still running, good
                logger.info(f"readsb started with PID {self.process.pid}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start readsb: {e}")
            return False
            
    def stop(self):
        """Stop readsb process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                
            logger.info("readsb stopped")
            self.process = None
            
    def is_running(self) -> bool:
        """Check if readsb is running."""
        if self.process is None:
            return False
        return self.process.poll() is None
        
    def get_status(self) -> dict:
        """Get readsb status."""
        status = {
            "running": self.is_running(),
            "pid": self.process.pid if self.process else None,
            "config": {
                "device_type": self.config.device_type,
                "device_index": self.config.device_index,
                "latitude": self.config.latitude,
                "longitude": self.config.longitude,
                "beast_port": self.config.beast_port
            }
        }
        
        # Check if JSON output exists
        if self.config.json_dir:
            aircraft_json = Path(self.config.json_dir) / "aircraft.json"
            if aircraft_json.exists():
                status["json_output"] = True
                status["json_mtime"] = aircraft_json.stat().st_mtime
            else:
                status["json_output"] = False
                
        return status


def generate_docker_compose_config(config: ReadsbConfig) -> dict:
    """Generate Docker Compose service configuration for ultrafeeder."""
    return {
        "image": "ghcr.io/sdr-enthusiasts/docker-adsb-ultrafeeder:latest",
        "restart": "unless-stopped",
        "device_cgroup_rules": ["c 189:* rwm"],
        "ports": [
            f"{config.beast_port}:{config.beast_port}",
            f"{config.sbs_port}:{config.sbs_port}",
            "8080:80"
        ],
        "environment": config.to_env(),
        "volumes": [
            f"{config.globe_history_dir}:/var/globe_history",
            "/dev:/dev:ro"
        ],
        "tmpfs": [
            "/run:exec,size=256M",
            "/tmp:size=128M"
        ]
    }
