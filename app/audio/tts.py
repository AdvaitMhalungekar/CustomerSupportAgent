"""Piper TTS wrapper — open source, runs locally on CPU.
Implements sentence-by-sentence streaming behind a TTSProvider interface.
Piper uses a two-stage ONNX model: phoneme encoder + mel decoder.
We call the piper binary via subprocess since the Python package requires
native build steps on Windows; this keeps install simple.

If the Piper binary is not installed we fall back to a silent stub that
returns empty audio so the rest of the pipeline still works for integration tests.
"""
import io
import json
import os
import re
import struct
import subprocess
import logging
import shutil
from abc import ABC, abstractmethod
from typing import Generator, Optional, List
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

PIPER_EXE_CANDIDATES = [
    os.path.join(settings.BASE_DIR, "piper", "piper.exe"),
    shutil.which("piper") or "",
]

FILLER_PHRASES = [
    "Let me check that for you.",
    "One moment please.",
    "Sure, just a second.",
    "I am looking that up for you now.",
]

# Sentence boundary splitter (handles ., ?, !, ... for natural streaming)
_SENT_SPLITTER = re.compile(r"(?<=[.!?…])\s+")


def split_into_sentences(text: str) -> List[str]:
    """Split a text into spoken sentences for streaming."""
    if not text:
        return []
    sents = _SENT_SPLITTER.split(text.strip())
    return [s.strip() for s in sents if s.strip()]


class TTSProvider(ABC):
    """Abstract TTS interface — swap Piper for any other provider here."""

    @abstractmethod
    def synthesize_sentence(self, text: str) -> np.ndarray:
        """Synthesize one sentence; returns int16 PCM @ 22050Hz."""
        pass

    def stream_sentences(self, text: str) -> Generator[np.ndarray, None, None]:
        """Yield PCM chunks sentence by sentence for low-latency streaming."""
        for sent in split_into_sentences(text):
            audio = self.synthesize_sentence(sent)
            if audio is not None and len(audio) > 0:
                yield audio

    def filler(self) -> np.ndarray:
        """Return audio for a filler phrase while tool runs."""
        import random
        return self.synthesize_sentence(random.choice(FILLER_PHRASES))


class PiperTTS(TTSProvider):
    """Piper TTS via subprocess. Falls back to stub if binary not found."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        self.model_path = model_path or os.path.join(settings.MODELS_DIR, "en_US-lessac-medium.onnx")
        self.config_path = config_path or f"{self.model_path}.json"
        self.sample_rate = 22050  # Piper default
        self._binary = self._find_binary()

        if self._binary:
            logger.info(f"Piper TTS ready: binary={self._binary}, model={self.model_path}")
        else:
            logger.warning("Piper binary not found. Using silent stub TTS (synthesize returns silence).")

    def _find_binary(self) -> Optional[str]:
        for cand in PIPER_EXE_CANDIDATES:
            if cand and os.path.isfile(cand):
                return cand
        return None

    def synthesize_sentence(self, text: str) -> np.ndarray:
        """Synthesize one sentence using Piper subprocess."""
        if not text or not text.strip():
            return np.array([], dtype=np.int16)

        if self._binary is None or not os.path.exists(self.model_path):
            return self._sapi_or_stub(text)

        try:
            cmd = [
                self._binary,
                "--model", self.model_path,
                "--config", self.config_path,
                "--output-raw",
                "--quiet",
            ]
            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=10.0,
            )
            if result.returncode != 0 or not result.stdout:
                logger.warning(f"Piper returned code {result.returncode} for: '{text}'")
                return self._sapi_or_stub(text)

            pcm = np.frombuffer(result.stdout, dtype=np.int16)
            return pcm
        except subprocess.TimeoutExpired:
            logger.warning(f"Piper synthesis timed out for: '{text}'")
            return self._sapi_or_stub(text)
        except Exception as e:
            logger.error(f"Piper synthesis failed: {e}")
            return self._sapi_or_stub(text)

    def _sapi_or_stub(self, text: str) -> np.ndarray:
        """Synthesize using Windows SAPI if available, otherwise return silent stub."""
        try:
            import tempfile
            import wave
            import win32com.client
            import pythoncom

            pythoncom.CoInitialize()
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            stream = win32com.client.Dispatch("SAPI.SpFileStream")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_path = tf.name

            stream.Open(tmp_path, 3, False)
            voice.AudioOutputStream = stream
            voice.Speak(text)
            stream.Close()

            with wave.open(tmp_path, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                pcm = np.frombuffer(frames, dtype=np.int16)
                if sr != self.sample_rate:
                    import audioop
                    resampled, _ = audioop.ratecv(pcm.tobytes(), 2, 1, sr, self.sample_rate, None)
                    pcm = np.frombuffer(resampled, dtype=np.int16)

            try:
                os.remove(tmp_path)
            except Exception:
                pass

            return pcm
        except Exception as err:
            logger.debug(f"SAPI fallback unavailable ({err}), using silent stub.")
            return self._stub_audio(text)

    def _stub_audio(self, text: str) -> np.ndarray:
        """Return silent PCM proportional to text length (for integration tests)."""
        num_words = len(text.split())
        duration_samples = int(self.sample_rate * num_words * 0.4)
        return np.zeros(duration_samples, dtype=np.int16)

    def pcm_to_wav_bytes(self, pcm: np.ndarray) -> bytes:
        """Wrap PCM in a WAV container for browser playback."""
        sr = self.sample_rate
        num_samples = len(pcm)
        data_bytes = pcm.tobytes()
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + len(data_bytes), b"WAVE",
            b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16,
            b"data", len(data_bytes),
        )
        return header + data_bytes


# Global default TTS provider
tts_provider: TTSProvider = PiperTTS()
