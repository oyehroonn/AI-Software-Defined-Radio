"""IQ sample capture service for PHY-layer analysis."""

import logging
import threading
import time
from datetime import datetime, UTC
from typing import Optional, Callable
from dataclasses import dataclass
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# Try to import GNU Radio components
try:
    from gnuradio import gr, blocks
    import osmosdr
    GNURADIO_AVAILABLE = True
except ImportError:
    GNURADIO_AVAILABLE = False
    logger.warning("GNU Radio not available, using mock IQ capture")


@dataclass
class IQBurst:
    """Captured IQ burst around a detected message."""
    timestamp: datetime
    center_freq: float
    sample_rate: float
    samples: np.ndarray
    trigger_offset: int
    icao24: Optional[str] = None


class RingBuffer:
    """Thread-safe ring buffer for IQ samples."""
    
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.complex64)
        self.write_idx = 0
        self.lock = threading.Lock()
        
    def write(self, samples: np.ndarray):
        """Write samples to ring buffer."""
        with self.lock:
            n = len(samples)
            if n >= self.size:
                self.buffer[:] = samples[-self.size:]
                self.write_idx = 0
            else:
                end_idx = (self.write_idx + n) % self.size
                if end_idx > self.write_idx:
                    self.buffer[self.write_idx:end_idx] = samples
                else:
                    first_part = self.size - self.write_idx
                    self.buffer[self.write_idx:] = samples[:first_part]
                    self.buffer[:end_idx] = samples[first_part:]
                self.write_idx = end_idx
                
    def read_window(self, offset_from_current: int, length: int) -> np.ndarray:
        """Read a window of samples relative to current position."""
        with self.lock:
            start = (self.write_idx - offset_from_current - length) % self.size
            
            if start + length <= self.size:
                return self.buffer[start:start + length].copy()
            else:
                first_part = self.size - start
                result = np.empty(length, dtype=np.complex64)
                result[:first_part] = self.buffer[start:]
                result[first_part:] = self.buffer[:length - first_part]
                return result


if GNURADIO_AVAILABLE:
    class IQCaptureFlowgraph(gr.top_block):
        """GNU Radio flowgraph for IQ capture."""
        
        def __init__(
            self,
            center_freq: float = 1090e6,
            sample_rate: float = 2e6,
            gain: float = 40,
            device_args: str = "rtl=0"
        ):
            gr.top_block.__init__(self, "IQ Capture")
            
            self.center_freq = center_freq
            self.sample_rate = sample_rate
            self.gain = gain
            
            # Buffer for ~1 second of samples
            buffer_size = int(sample_rate)
            self.ring_buffer = RingBuffer(buffer_size)
            
            # SDR source
            self.source = osmosdr.source(args=f"numchan=1 {device_args}")
            self.source.set_sample_rate(sample_rate)
            self.source.set_center_freq(center_freq)
            self.source.set_gain(gain)
            self.source.set_bandwidth(0)
            
            # Custom sink to feed ring buffer
            self.sink = blocks.vector_sink_c()
            
            self.connect(self.source, self.sink)
            
        def get_burst_window(
            self,
            trigger_time: float,
            pre_samples: int = 100,
            post_samples: int = 500
        ) -> Optional[IQBurst]:
            """Extract IQ window around a detected burst."""
            # Calculate sample offset from current time
            current_time = time.time()
            time_diff = current_time - trigger_time
            
            if time_diff < 0 or time_diff > 1.0:
                logger.warning(f"Trigger time out of buffer range: {time_diff}s")
                return None
                
            sample_offset = int(time_diff * self.sample_rate)
            total_samples = pre_samples + post_samples
            
            samples = self.ring_buffer.read_window(
                sample_offset - post_samples,
                total_samples
            )
            
            return IQBurst(
                timestamp=datetime.fromtimestamp(trigger_time, UTC),
                center_freq=self.center_freq,
                sample_rate=self.sample_rate,
                samples=samples,
                trigger_offset=pre_samples
            )


class MockIQCapture:
    """Mock IQ capture for testing without hardware."""
    
    def __init__(
        self,
        center_freq: float = 1090e6,
        sample_rate: float = 2e6
    ):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.running = False
        
    def start(self):
        self.running = True
        
    def stop(self):
        self.running = False
        
    def get_burst_window(
        self,
        trigger_time: float,
        pre_samples: int = 100,
        post_samples: int = 500
    ) -> IQBurst:
        """Generate mock IQ burst for testing."""
        total_samples = pre_samples + post_samples
        
        # Generate Mode-S like signal
        t = np.arange(total_samples) / self.sample_rate
        
        # 1090 MHz PPM modulation
        freq_offset = np.random.normal(0, 100)  # CFO
        phase_noise = np.random.normal(0, 0.1, total_samples)
        
        # Preamble + data pattern
        signal = np.zeros(total_samples, dtype=np.complex64)
        
        # Add preamble pulses at 0, 1, 3.5, 4.5 microseconds
        samples_per_us = self.sample_rate / 1e6
        preamble_positions = [0, 1, 3.5, 4.5]
        
        for pos in preamble_positions:
            idx = int((pre_samples + pos * samples_per_us))
            if idx < total_samples:
                pulse_len = int(0.5 * samples_per_us)
                signal[idx:idx + pulse_len] = 1.0
                
        # Add some noise
        noise = (np.random.randn(total_samples) + 
                1j * np.random.randn(total_samples)) * 0.1
        
        # Apply CFO
        cfo_phase = 2 * np.pi * freq_offset * t
        signal = signal * np.exp(1j * (cfo_phase + phase_noise))
        
        samples = (signal + noise).astype(np.complex64)
        
        return IQBurst(
            timestamp=datetime.fromtimestamp(trigger_time, UTC),
            center_freq=self.center_freq,
            sample_rate=self.sample_rate,
            samples=samples,
            trigger_offset=pre_samples
        )


class IQCaptureService:
    """Service for capturing IQ samples around detected ADS-B messages."""
    
    def __init__(
        self,
        center_freq: float = 1090e6,
        sample_rate: float = 2e6,
        gain: float = 40,
        use_mock: bool = False
    ):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        
        if use_mock or not GNURADIO_AVAILABLE:
            logger.info("Using mock IQ capture")
            self.capture = MockIQCapture(center_freq, sample_rate)
        else:
            self.capture = IQCaptureFlowgraph(
                center_freq=center_freq,
                sample_rate=sample_rate,
                gain=gain
            )
            
        self.burst_handlers: list[Callable[[IQBurst], None]] = []
        
    def add_handler(self, handler: Callable[[IQBurst], None]):
        """Add handler for captured bursts."""
        self.burst_handlers.append(handler)
        
    def start(self):
        """Start capture."""
        if hasattr(self.capture, 'start'):
            self.capture.start()
        logger.info("IQ capture service started")
        
    def stop(self):
        """Stop capture."""
        if hasattr(self.capture, 'stop'):
            self.capture.stop()
        logger.info("IQ capture service stopped")
        
    def capture_burst(
        self,
        trigger_time: float,
        icao24: Optional[str] = None,
        pre_us: int = 10,
        post_us: int = 130
    ) -> Optional[IQBurst]:
        """
        Capture IQ burst around a message detection.
        
        Args:
            trigger_time: Unix timestamp of message detection
            icao24: ICAO24 of detected aircraft
            pre_us: Microseconds before trigger to capture
            post_us: Microseconds after trigger (Mode-S long msg = 120us)
            
        Returns:
            IQBurst with captured samples
        """
        samples_per_us = self.sample_rate / 1e6
        pre_samples = int(pre_us * samples_per_us)
        post_samples = int(post_us * samples_per_us)
        
        burst = self.capture.get_burst_window(
            trigger_time,
            pre_samples,
            post_samples
        )
        
        if burst:
            burst.icao24 = icao24
            
            # Call handlers
            for handler in self.burst_handlers:
                try:
                    handler(burst)
                except Exception as e:
                    logger.error(f"Burst handler error: {e}")
                    
        return burst
