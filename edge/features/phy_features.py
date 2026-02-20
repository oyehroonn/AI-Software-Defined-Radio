"""PHY-layer feature extraction for RF fingerprinting."""

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy import signal as scipy_signal
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, some PHY features disabled")


@dataclass
class PHYFeatures:
    """PHY-layer features extracted from IQ burst."""
    icao24: str
    timestamp: float
    
    # Carrier Frequency Offset features
    cfo_mean: float
    cfo_std: float
    cfo_drift: float
    
    # Amplitude envelope features
    amp_mean: float
    amp_std: float
    amp_skew: float
    amp_kurtosis: float
    
    # Preamble transient features
    preamble_rise_time: float
    preamble_overshoot: float
    preamble_ringing: float
    
    # Phase features
    phase_std: float
    phase_jitter: float
    phase_linearity: float
    
    # Spectral features
    spectral_centroid: float
    spectral_spread: float
    spectral_flatness: float
    
    # Signal quality
    snr_estimate: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "icao24": self.icao24,
            "timestamp": self.timestamp,
            "cfo_mean": self.cfo_mean,
            "cfo_std": self.cfo_std,
            "cfo_drift": self.cfo_drift,
            "amp_mean": self.amp_mean,
            "amp_std": self.amp_std,
            "amp_skew": self.amp_skew,
            "amp_kurtosis": self.amp_kurtosis,
            "preamble_rise_time": self.preamble_rise_time,
            "preamble_overshoot": self.preamble_overshoot,
            "preamble_ringing": self.preamble_ringing,
            "phase_std": self.phase_std,
            "phase_jitter": self.phase_jitter,
            "phase_linearity": self.phase_linearity,
            "spectral_centroid": self.spectral_centroid,
            "spectral_spread": self.spectral_spread,
            "spectral_flatness": self.spectral_flatness,
            "snr_estimate": self.snr_estimate
        }
        
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML model."""
        return np.array([
            self.cfo_mean,
            self.cfo_std,
            self.cfo_drift,
            self.amp_mean,
            self.amp_std,
            self.amp_skew,
            self.amp_kurtosis,
            self.preamble_rise_time,
            self.preamble_overshoot,
            self.preamble_ringing,
            self.phase_std,
            self.phase_jitter,
            self.phase_linearity,
            self.spectral_centroid,
            self.spectral_spread,
            self.spectral_flatness,
            self.snr_estimate
        ])


def compute_rise_time(
    amplitude: np.ndarray,
    sample_rate: float,
    low_pct: float = 0.1,
    high_pct: float = 0.9
) -> float:
    """Compute rise time of amplitude envelope."""
    if len(amplitude) < 2:
        return 0.0
        
    max_amp = np.max(amplitude)
    if max_amp == 0:
        return 0.0
        
    norm_amp = amplitude / max_amp
    
    # Find low and high threshold crossings
    low_threshold = low_pct
    high_threshold = high_pct
    
    low_idx = np.where(norm_amp >= low_threshold)[0]
    high_idx = np.where(norm_amp >= high_threshold)[0]
    
    if len(low_idx) == 0 or len(high_idx) == 0:
        return 0.0
        
    rise_samples = high_idx[0] - low_idx[0]
    rise_time_us = rise_samples / sample_rate * 1e6
    
    return float(rise_time_us)


def compute_overshoot(amplitude: np.ndarray) -> float:
    """Compute overshoot percentage in transient."""
    if len(amplitude) < 10:
        return 0.0
        
    # Assume steady state is average of last 20%
    steady_start = int(len(amplitude) * 0.8)
    steady_state = np.mean(amplitude[steady_start:])
    
    if steady_state == 0:
        return 0.0
        
    max_amp = np.max(amplitude[:steady_start])
    overshoot = (max_amp - steady_state) / steady_state * 100
    
    return float(max(0, overshoot))


def compute_ringing(amplitude: np.ndarray) -> float:
    """Compute ringing metric from amplitude envelope."""
    if len(amplitude) < 20:
        return 0.0
        
    # Look for oscillations after initial peak
    peak_idx = np.argmax(amplitude[:len(amplitude)//2])
    post_peak = amplitude[peak_idx:]
    
    if len(post_peak) < 10:
        return 0.0
        
    # Count zero crossings of derivative (oscillations)
    derivative = np.diff(post_peak)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(derivative))) > 0)
    
    return float(zero_crossings)


def extract_phy_features(
    iq_burst: np.ndarray,
    sample_rate: float,
    icao24: str = "unknown",
    timestamp: float = 0.0
) -> PHYFeatures:
    """
    Extract PHY-layer features from IQ burst.
    
    Args:
        iq_burst: Complex IQ samples
        sample_rate: Sample rate in Hz
        icao24: Aircraft ICAO24 address
        timestamp: Burst timestamp
        
    Returns:
        PHYFeatures dataclass
    """
    # Amplitude envelope
    amplitude = np.abs(iq_burst)
    
    # Phase
    phase = np.unwrap(np.angle(iq_burst))
    
    # Instantaneous frequency (CFO)
    if len(phase) > 1:
        freq_inst = np.diff(phase) * sample_rate / (2 * np.pi)
    else:
        freq_inst = np.array([0.0])
        
    # CFO features
    cfo_mean = float(np.mean(freq_inst))
    cfo_std = float(np.std(freq_inst))
    cfo_drift = float(freq_inst[-1] - freq_inst[0]) if len(freq_inst) > 1 else 0.0
    
    # Amplitude features
    amp_mean = float(np.mean(amplitude))
    amp_std = float(np.std(amplitude))
    
    if SCIPY_AVAILABLE:
        amp_skew = float(scipy_stats.skew(amplitude))
        amp_kurtosis = float(scipy_stats.kurtosis(amplitude))
    else:
        amp_skew = 0.0
        amp_kurtosis = 0.0
        
    # Preamble analysis (first 8us of Mode-S message)
    samples_per_us = sample_rate / 1e6
    preamble_samples = int(8 * samples_per_us)
    preamble = amplitude[:min(preamble_samples, len(amplitude))]
    
    preamble_rise_time = compute_rise_time(preamble, sample_rate)
    preamble_overshoot = compute_overshoot(preamble)
    preamble_ringing = compute_ringing(preamble)
    
    # Phase features
    phase_std = float(np.std(phase))
    phase_jitter = float(np.std(np.diff(phase))) if len(phase) > 1 else 0.0
    
    # Phase linearity (correlation with linear fit)
    if len(phase) > 2:
        t = np.arange(len(phase))
        coeffs = np.polyfit(t, phase, 1)
        linear_fit = np.polyval(coeffs, t)
        residuals = phase - linear_fit
        phase_linearity = 1.0 - float(np.std(residuals) / (np.std(phase) + 1e-10))
    else:
        phase_linearity = 1.0
        
    # Spectral features
    if SCIPY_AVAILABLE and len(iq_burst) > 10:
        f, psd = scipy_signal.welch(iq_burst, fs=sample_rate, nperseg=min(256, len(iq_burst)))
        psd = np.abs(psd)
        
        # Normalize PSD
        psd_sum = np.sum(psd)
        if psd_sum > 0:
            psd_norm = psd / psd_sum
            
            spectral_centroid = float(np.sum(f * psd_norm))
            spectral_spread = float(np.sqrt(np.sum((f - spectral_centroid)**2 * psd_norm)))
            
            # Spectral flatness (geometric mean / arithmetic mean)
            geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
            arithmetic_mean = np.mean(psd)
            spectral_flatness = float(geometric_mean / (arithmetic_mean + 1e-10))
        else:
            spectral_centroid = 0.0
            spectral_spread = 0.0
            spectral_flatness = 0.0
    else:
        spectral_centroid = 0.0
        spectral_spread = 0.0
        spectral_flatness = 0.0
        
    # SNR estimate
    signal_power = np.mean(amplitude**2)
    
    # Estimate noise from quiet portions (first/last 5%)
    noise_samples = max(1, len(amplitude) // 20)
    noise_power = (np.mean(amplitude[:noise_samples]**2) + 
                  np.mean(amplitude[-noise_samples:]**2)) / 2
    
    if noise_power > 0:
        snr_estimate = float(10 * np.log10(signal_power / noise_power))
    else:
        snr_estimate = 30.0  # Default high SNR
        
    return PHYFeatures(
        icao24=icao24,
        timestamp=timestamp,
        cfo_mean=cfo_mean,
        cfo_std=cfo_std,
        cfo_drift=cfo_drift,
        amp_mean=amp_mean,
        amp_std=amp_std,
        amp_skew=amp_skew,
        amp_kurtosis=amp_kurtosis,
        preamble_rise_time=preamble_rise_time,
        preamble_overshoot=preamble_overshoot,
        preamble_ringing=preamble_ringing,
        phase_std=phase_std,
        phase_jitter=phase_jitter,
        phase_linearity=phase_linearity,
        spectral_centroid=spectral_centroid,
        spectral_spread=spectral_spread,
        spectral_flatness=spectral_flatness,
        snr_estimate=snr_estimate
    )


class PHYFeatureExtractor:
    """Service for extracting PHY features from IQ bursts."""
    
    def __init__(self, sample_rate: float = 2e6):
        self.sample_rate = sample_rate
        self.feature_history: dict[str, list[PHYFeatures]] = {}
        self.max_history = 100
        
    def extract(
        self,
        iq_burst: np.ndarray,
        icao24: str,
        timestamp: float
    ) -> PHYFeatures:
        """Extract features and store in history."""
        features = extract_phy_features(
            iq_burst,
            self.sample_rate,
            icao24,
            timestamp
        )
        
        # Store in history
        if icao24 not in self.feature_history:
            self.feature_history[icao24] = []
            
        self.feature_history[icao24].append(features)
        
        # Trim history
        if len(self.feature_history[icao24]) > self.max_history:
            self.feature_history[icao24].pop(0)
            
        return features
        
    def get_fingerprint(self, icao24: str) -> Optional[np.ndarray]:
        """Get average feature vector as RF fingerprint."""
        if icao24 not in self.feature_history:
            return None
            
        features = self.feature_history[icao24]
        if len(features) < 5:
            return None
            
        vectors = np.array([f.to_vector() for f in features])
        return np.mean(vectors, axis=0)
        
    def compute_consistency(
        self,
        current: PHYFeatures,
        icao24: str
    ) -> float:
        """
        Compute consistency score between current transmission and history.
        
        Returns:
            Consistency score 0-1 (1 = consistent with history)
        """
        fingerprint = self.get_fingerprint(icao24)
        if fingerprint is None:
            return 1.0  # No history, assume consistent
            
        current_vector = current.to_vector()
        
        # Normalize vectors
        fp_norm = fingerprint / (np.linalg.norm(fingerprint) + 1e-10)
        cur_norm = current_vector / (np.linalg.norm(current_vector) + 1e-10)
        
        # Cosine similarity
        similarity = float(np.dot(fp_norm, cur_norm))
        
        return max(0.0, similarity)
