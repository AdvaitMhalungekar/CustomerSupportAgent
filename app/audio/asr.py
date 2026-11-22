"""Groq Whisper ASR client with retry and local faster-whisper fallback.

NOTE ON STREAMING: Groq's /v1/audio/transcriptions is utterance-level (not streaming).
We transcribe each VAD-segmented utterance as a complete audio chunk.
There are no true interim results — this is a known tradeoff documented in README.
Perceived latency is minimized by keeping utterances short via tight VAD endpointing.
"""
import io
import time
import logging
import random
import os
from typing import Optional, Tuple
import numpy as np

from app.config import settings
from app.audio.asr_normalizer import normalize_asr_transcript

logger = logging.getLogger(__name__)

BACKCHANNEL_TOKENS = {"hmm", "um", "uh", "okay", "ok", "yeah", "yes", "right", "sure", "mhm"}


def _is_backchannel(text: str) -> bool:
    """Detect backchannel fillers that shouldn't trigger a full re-plan."""
    words = set(text.lower().strip().split())
    return len(words) <= 2 and words.issubset(BACKCHANNEL_TOKENS)


class GroqASR:
    """Utterance-level ASR using Groq's Whisper endpoint."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = settings.GROQ_ASR_MODEL
        self.base_url = settings.GROQ_BASE_URL
        self.is_offline = not bool(self.api_key and self.api_key != "your_groq_api_key_here")
        self._client = None

        if not self.is_offline:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
                logger.info(f"Groq ASR client initialized (model: {self.model})")
            except Exception as e:
                logger.warning(f"Groq ASR init failed: {e}. Using local fallback.")
                self._init_local_fallback()
        else:
            logger.info("ASR operating in offline/stub mode (no GROQ_API_KEY).")

    def _init_local_fallback(self):
        """Attempt to load faster-whisper as local fallback."""
        try:
            from faster_whisper import WhisperModel
            self._local_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            logger.info("Local faster-whisper (tiny) loaded as ASR fallback.")
        except ImportError:
            self._local_model = None
            logger.warning("faster-whisper not installed. ASR fallback unavailable.")

    def transcribe(self, audio_pcm: np.ndarray, sample_rate: int = 16000, max_retries: int = 3) -> Tuple[str, bool]:
        """
        Transcribe a PCM audio array.
        Returns: (transcript_text, is_backchannel)
        Uses Groq Whisper if online, falls back to local faster-whisper or stub.
        """
        if audio_pcm is None or len(audio_pcm) == 0:
            return "", True

        if self.is_offline or self._client is None:
            return self._stub_transcribe(audio_pcm), False

        # Convert to WAV bytes for Groq HTTP upload
        wav_bytes = _pcm_to_wav_bytes(audio_pcm, sample_rate)
        delay = 1.0

        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                transcription = self._client.audio.transcriptions.create(
                    model=self.model,
                    file=("audio.wav", wav_bytes, "audio/wav"),
                    language="en",
                    response_format="json",
                )
                asr_ms = (time.perf_counter() - t0) * 1000
                raw_text = transcription.text.strip()
                normalized = normalize_asr_transcript(raw_text)
                logger.debug(f"ASR ({asr_ms:.0f}ms): '{raw_text}' -> '{normalized}'")
                return normalized, _is_backchannel(normalized)

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str:
                    jitter = random.uniform(0.1, 0.5)
                    sleep_time = delay + jitter
                    logger.warning(f"Groq ASR rate limit (attempt {attempt+1}). Retry in {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    delay *= 2
                else:
                    logger.error(f"Groq ASR error: {e}. Trying local fallback.")
                    break

        # Fallback to local model
        return self._local_fallback_transcribe(audio_pcm, sample_rate)

    def _local_fallback_transcribe(self, audio_pcm: np.ndarray, sample_rate: int) -> Tuple[str, bool]:
        """Use faster-whisper if available."""
        if hasattr(self, "_local_model") and self._local_model is not None:
            try:
                audio_f32 = audio_pcm.astype(np.float32) / 32768.0
                segments, _ = self._local_model.transcribe(audio_f32, language="en")
                text = " ".join(s.text for s in segments).strip()
                normalized = normalize_asr_transcript(text)
                return normalized, _is_backchannel(normalized)
            except Exception as e:
                logger.error(f"Local ASR fallback failed: {e}")
        return "[transcription unavailable]", True

    def _stub_transcribe(self, audio_pcm: np.ndarray) -> str:
        """Deterministic stub for offline testing."""
        rms = float(np.sqrt(np.mean(audio_pcm.astype(np.float32) ** 2)))
        if rms < 100:
            return ""
        return "[OFFLINE MODE: simulated transcript]"


def _pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert int16 PCM numpy array to WAV bytes in memory."""
    import struct
    num_samples = len(pcm)
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    header_size = 44

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        header_size + data_size - 8,
        b"WAVE",
        b"fmt ",
        16,            # PCM chunk size
        1,             # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm.astype(np.int16).tobytes()


# Global singleton
groq_asr = GroqASR()
