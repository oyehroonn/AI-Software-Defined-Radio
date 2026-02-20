"""VHF airband voice capture service."""

import logging
import threading
import time
import wave
import io
from datetime import datetime, UTC
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

# Try to import GNU Radio components
try:
    from gnuradio import gr, analog, audio, blocks, filter as gr_filter
    import osmosdr
    GNURADIO_AVAILABLE = True
except ImportError:
    GNURADIO_AVAILABLE = False
    logger.warning("GNU Radio not available, using mock VHF capture")


@dataclass
class VoiceRecording:
    """Captured voice transmission."""
    timestamp: datetime
    frequency_mhz: float
    duration_seconds: float
    audio_data: np.ndarray
    sample_rate: int
    sensor_id: str
    signal_level: float
    

class VoiceActivityDetector:
    """Simple voice activity detector based on energy threshold."""
    
    def __init__(
        self,
        threshold_db: float = -30,
        hold_time_sec: float = 0.5,
        sample_rate: int = 48000
    ):
        self.threshold_db = threshold_db
        self.hold_samples = int(hold_time_sec * sample_rate)
        self.sample_rate = sample_rate
        self.active = False
        self.samples_since_active = 0
        
    def process(self, audio_chunk: np.ndarray) -> bool:
        """Process audio chunk and return if voice is active."""
        # Calculate energy in dB
        energy = np.mean(audio_chunk ** 2)
        energy_db = 10 * np.log10(energy + 1e-10)
        
        if energy_db > self.threshold_db:
            self.active = True
            self.samples_since_active = 0
        else:
            self.samples_since_active += len(audio_chunk)
            if self.samples_since_active > self.hold_samples:
                self.active = False
                
        return self.active


if GNURADIO_AVAILABLE:
    class VHFAirbandFlowgraph(gr.top_block):
        """GNU Radio flowgraph for VHF airband reception."""
        
        def __init__(
            self,
            center_freq: float = 127.5e6,
            sample_rate: float = 2e6,
            audio_rate: int = 48000,
            gain: float = 40,
            squelch_db: float = -30
        ):
            gr.top_block.__init__(self, "VHF Airband")
            
            self.center_freq = center_freq
            self.sample_rate = sample_rate
            self.audio_rate = audio_rate
            
            # SDR source
            self.source = osmosdr.source(args="numchan=1")
            self.source.set_sample_rate(sample_rate)
            self.source.set_center_freq(center_freq)
            self.source.set_gain(gain)
            
            # Low-pass filter for channel selection
            channel_width = 25000  # 25 kHz AM channel
            filter_taps = gr_filter.firdes.low_pass(
                1.0,
                sample_rate,
                channel_width / 2,
                channel_width / 10
            )
            self.channel_filter = gr_filter.fir_filter_ccf(1, filter_taps)
            
            # Squelch
            self.squelch = analog.simple_squelch_cc(squelch_db, 1.0)
            
            # AM demodulation (airband uses AM)
            decim = int(sample_rate / audio_rate)
            self.demod = analog.am_demod_cf(
                channel_rate=sample_rate,
                audio_decim=decim,
                audio_pass=5000,
                audio_stop=5500
            )
            
            # Audio output sink
            self.audio_sink = blocks.vector_sink_f()
            
            # Connect blocks
            self.connect(self.source, self.channel_filter)
            self.connect(self.channel_filter, self.squelch)
            self.connect(self.squelch, self.demod)
            self.connect(self.demod, self.audio_sink)
            
        def get_audio(self, clear: bool = True) -> np.ndarray:
            """Get captured audio samples."""
            data = np.array(self.audio_sink.data())
            if clear:
                self.audio_sink.reset()
            return data
            
        def set_frequency(self, freq: float):
            """Change center frequency."""
            self.source.set_center_freq(freq)
            self.center_freq = freq


class MockVHFCapture:
    """Mock VHF capture for testing without hardware."""
    
    def __init__(
        self,
        center_freq: float = 127.5e6,
        audio_rate: int = 48000
    ):
        self.center_freq = center_freq
        self.audio_rate = audio_rate
        self.running = False
        self._thread = None
        self._audio_buffer = []
        
    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._generate_audio)
        self._thread.start()
        
    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()
            
    def _generate_audio(self):
        """Generate mock audio with occasional voice-like patterns."""
        while self.running:
            # Generate 1 second of audio
            t = np.linspace(0, 1, self.audio_rate)
            
            # Random chance of "voice" (AM modulated tone)
            if np.random.random() < 0.1:  # 10% chance of voice
                # Simulate voice with multiple tones
                audio = np.zeros(self.audio_rate)
                for _ in range(3):
                    freq = np.random.uniform(200, 800)
                    audio += np.sin(2 * np.pi * freq * t)
                audio = audio / 3 * 0.5
                audio *= np.random.uniform(0.8, 1.2, self.audio_rate)  # AM modulation
            else:
                # Just noise
                audio = np.random.randn(self.audio_rate) * 0.01
                
            self._audio_buffer.extend(audio.tolist())
            time.sleep(1)
            
    def get_audio(self, clear: bool = True) -> np.ndarray:
        """Get captured audio."""
        data = np.array(self._audio_buffer)
        if clear:
            self._audio_buffer.clear()
        return data
        
    def set_frequency(self, freq: float):
        self.center_freq = freq


class VHFCaptureService:
    """Service for capturing VHF airband voice communications."""
    
    # Common ATC frequencies (MHz)
    COMMON_FREQUENCIES = [
        118.0,    # Tower common
        119.1,    # Tower
        121.5,    # Emergency
        122.75,   # Unicom
        123.0,    # Unicom
        124.0,    # Approach
        125.0,    # Approach
        127.5,    # Departure
        128.0,    # Center
        132.0,    # Center
        134.0,    # ATIS
    ]
    
    def __init__(
        self,
        frequencies: Optional[list[float]] = None,
        audio_rate: int = 48000,
        use_mock: bool = False,
        sensor_id: str = "sensor-1",
        recordings_path: str = "/var/lib/aerosentry/voice"
    ):
        self.frequencies = frequencies or [121.5e6]  # Default: emergency
        self.audio_rate = audio_rate
        self.sensor_id = sensor_id
        self.recordings_path = Path(recordings_path)
        self.recordings_path.mkdir(parents=True, exist_ok=True)
        
        self.current_freq_idx = 0
        self.vad = VoiceActivityDetector(sample_rate=audio_rate)
        
        if use_mock or not GNURADIO_AVAILABLE:
            logger.info("Using mock VHF capture")
            self.capture = MockVHFCapture(audio_rate=audio_rate)
        else:
            self.capture = VHFAirbandFlowgraph(
                center_freq=self.frequencies[0],
                audio_rate=audio_rate
            )
            
        self.recording_handlers: list[Callable[[VoiceRecording], None]] = []
        self._running = False
        self._thread = None
        
        # Recording state
        self._recording = False
        self._recording_buffer = []
        self._recording_start = None
        
    def add_handler(self, handler: Callable[[VoiceRecording], None]):
        """Add handler for completed recordings."""
        self.recording_handlers.append(handler)
        
    def start(self):
        """Start capture service."""
        self._running = True
        
        if hasattr(self.capture, 'start'):
            self.capture.start()
            
        self._thread = threading.Thread(target=self._process_loop)
        self._thread.start()
        
        logger.info(f"VHF capture started on {self.frequencies[0]/1e6:.3f} MHz")
        
    def stop(self):
        """Stop capture service."""
        self._running = False
        
        if self._thread:
            self._thread.join()
            
        if hasattr(self.capture, 'stop'):
            self.capture.stop()
            
        logger.info("VHF capture stopped")
        
    def set_frequency(self, freq_mhz: float):
        """Change capture frequency."""
        freq_hz = freq_mhz * 1e6
        self.capture.set_frequency(freq_hz)
        logger.info(f"Frequency changed to {freq_mhz:.3f} MHz")
        
    def _process_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                # Get audio chunk
                audio = self.capture.get_audio()
                
                if len(audio) == 0:
                    time.sleep(0.1)
                    continue
                    
                # Process in chunks
                chunk_size = self.audio_rate // 10  # 100ms chunks
                
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        break
                        
                    voice_active = self.vad.process(chunk)
                    
                    if voice_active and not self._recording:
                        # Start recording
                        self._recording = True
                        self._recording_buffer = []
                        self._recording_start = datetime.now(UTC)
                        
                    if self._recording:
                        self._recording_buffer.extend(chunk.tolist())
                        
                    if not voice_active and self._recording:
                        # Stop recording and process
                        self._recording = False
                        self._process_recording()
                        
            except Exception as e:
                logger.error(f"VHF processing error: {e}")
                time.sleep(0.5)
                
    def _process_recording(self):
        """Process completed recording."""
        if len(self._recording_buffer) < self.audio_rate * 0.5:
            # Too short, discard
            return
            
        audio_data = np.array(self._recording_buffer)
        duration = len(audio_data) / self.audio_rate
        
        # Calculate signal level
        signal_level = 10 * np.log10(np.mean(audio_data ** 2) + 1e-10)
        
        freq_mhz = self.capture.center_freq / 1e6 if hasattr(self.capture, 'center_freq') else 0
        
        recording = VoiceRecording(
            timestamp=self._recording_start,
            frequency_mhz=freq_mhz,
            duration_seconds=duration,
            audio_data=audio_data,
            sample_rate=self.audio_rate,
            sensor_id=self.sensor_id,
            signal_level=signal_level
        )
        
        # Save to file
        filename = f"{self._recording_start.strftime('%Y%m%d_%H%M%S')}_{freq_mhz:.1f}MHz.wav"
        filepath = self.recordings_path / filename
        self._save_wav(filepath, audio_data)
        
        # Call handlers
        for handler in self.recording_handlers:
            try:
                handler(recording)
            except Exception as e:
                logger.error(f"Recording handler error: {e}")
                
        logger.info(f"Voice recording: {duration:.1f}s at {freq_mhz:.3f} MHz")
        
    def _save_wav(self, path: Path, audio: np.ndarray):
        """Save audio to WAV file."""
        # Normalize and convert to 16-bit
        audio_norm = audio / (np.max(np.abs(audio)) + 1e-10)
        audio_int16 = (audio_norm * 32767).astype(np.int16)
        
        with wave.open(str(path), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.audio_rate)
            wav.writeframes(audio_int16.tobytes())
