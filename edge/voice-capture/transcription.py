"""ASR transcription service using Whisper."""

import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Try to import Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TranscriptionResult:
    """Result of speech transcription."""
    text: str
    confidence: float
    language: str
    segments: list[dict]
    duration_seconds: float
    abstain: bool


class ATCTranscriber:
    """Speech-to-text transcriber optimized for ATC communications."""
    
    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        confidence_threshold: float = 0.7
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if WHISPER_AVAILABLE:
            if device is None:
                device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
            self.device = device
            
            logger.info(f"Loading Whisper model '{model_name}' on {device}")
            self.model = whisper.load_model(model_name, device=device)
        else:
            logger.warning("Whisper not available, transcription disabled")
            
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio samples (float32)
            sample_rate: Sample rate in Hz
            
        Returns:
            TranscriptionResult with text and metadata
        """
        if self.model is None:
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language="unknown",
                segments=[],
                duration_seconds=len(audio) / sample_rate,
                abstain=True
            )
            
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)
            
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        # Transcribe
        result = self.model.transcribe(
            audio,
            language="en",  # ATC is typically English
            task="transcribe",
            fp16=TORCH_AVAILABLE and torch.cuda.is_available()
        )
        
        # Extract text and segments
        text = result["text"].strip()
        segments = result.get("segments", [])
        
        # Calculate confidence from segment probabilities
        if segments:
            probs = [s.get("avg_logprob", -1) for s in segments]
            avg_logprob = np.mean(probs)
            # Convert log prob to confidence (rough approximation)
            confidence = float(np.exp(avg_logprob))
        else:
            confidence = 0.5
            
        # Determine if we should abstain due to low confidence
        abstain = confidence < self.confidence_threshold
        
        return TranscriptionResult(
            text=text,
            confidence=confidence,
            language=result.get("language", "en"),
            segments=[{
                "start": s["start"],
                "end": s["end"],
                "text": s["text"]
            } for s in segments],
            duration_seconds=len(audio) / 16000,
            abstain=abstain
        )
        
    def transcribe_file(self, filepath: str) -> TranscriptionResult:
        """Transcribe audio file."""
        if self.model is None:
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language="unknown",
                segments=[],
                duration_seconds=0,
                abstain=True
            )
            
        result = self.model.transcribe(
            filepath,
            language="en",
            task="transcribe"
        )
        
        text = result["text"].strip()
        segments = result.get("segments", [])
        
        if segments:
            probs = [s.get("avg_logprob", -1) for s in segments]
            confidence = float(np.exp(np.mean(probs)))
        else:
            confidence = 0.5
            
        duration = segments[-1]["end"] if segments else 0
            
        return TranscriptionResult(
            text=text,
            confidence=confidence,
            language=result.get("language", "en"),
            segments=[{
                "start": s["start"],
                "end": s["end"],
                "text": s["text"]
            } for s in segments],
            duration_seconds=duration,
            abstain=confidence < self.confidence_threshold
        )
        
    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
            
        # Simple linear interpolation resampling
        duration = len(audio) / orig_sr
        num_samples = int(duration * target_sr)
        
        indices = np.linspace(0, len(audio) - 1, num_samples)
        return np.interp(indices, np.arange(len(audio)), audio)


class BatchTranscriber:
    """Batch transcription service."""
    
    def __init__(
        self,
        transcriber: Optional[ATCTranscriber] = None,
        output_dir: str = "/var/lib/aerosentry/transcripts"
    ):
        self.transcriber = transcriber or ATCTranscriber()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_directory(
        self,
        input_dir: str,
        pattern: str = "*.wav"
    ) -> list[tuple[str, TranscriptionResult]]:
        """Process all audio files in directory."""
        results = []
        input_path = Path(input_dir)
        
        for audio_file in input_path.glob(pattern):
            logger.info(f"Transcribing {audio_file}")
            result = self.transcriber.transcribe_file(str(audio_file))
            results.append((str(audio_file), result))
            
            # Save transcript
            if result.text:
                transcript_path = self.output_dir / f"{audio_file.stem}.txt"
                with open(transcript_path, 'w') as f:
                    f.write(f"File: {audio_file.name}\n")
                    f.write(f"Confidence: {result.confidence:.3f}\n")
                    f.write(f"Duration: {result.duration_seconds:.1f}s\n")
                    f.write(f"Abstained: {result.abstain}\n")
                    f.write(f"\nTranscript:\n{result.text}\n")
                    
        return results
