"""Twilio Audio Codec: mu-law 8kHz <-> linear PCM 16kHz conversion.

Twilio Media Streams send and receive 8kHz mu-law (PCMU) audio payload in base64.
Silero VAD and Whisper ASR operate on 16kHz 16-bit linear PCM (or float32).
Piper TTS outputs 22050Hz (or 16kHz) 16-bit linear PCM.
"""

import audioop
import base64
import numpy as np


def ulaw_to_pcm16_16k(ulaw_bytes: bytes) -> np.ndarray:
    """Convert Twilio 8kHz mu-law bytes to 16kHz 16-bit linear PCM numpy array.

    Args:
        ulaw_bytes: Raw bytes encoded in 8kHz mu-law (from Twilio media payload).

    Returns:
        np.ndarray: 1D int16 array sampled at 16000 Hz.
    """
    if not ulaw_bytes:
        return np.array([], dtype=np.int16)

    # 1. Decode mu-law to 16-bit linear PCM (sample width = 2 bytes, 8000 Hz)
    pcm_8k = audioop.ulaw2lin(ulaw_bytes, 2)

    # 2. Resample 8000 Hz -> 16000 Hz
    # ratecv(fragment, width, nchannels, inrate, outrate, state[, weightA[, weightB]])
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)

    # 3. Convert to numpy int16 array
    return np.frombuffer(pcm_16k, dtype=np.int16)


def pcm16_to_ulaw_8k(pcm_data: np.ndarray, orig_sr: int = 16000) -> bytes:
    """Convert linear PCM int16 numpy array to Twilio 8kHz mu-law bytes.

    Args:
        pcm_data: 1D int16 array.
        orig_sr: Input sample rate (typically 16000 or 22050 from TTS).

    Returns:
        bytes: 8kHz mu-law encoded bytes.
    """
    if pcm_data is None or len(pcm_data) == 0:
        return b""

    raw_pcm = pcm_data.astype(np.int16).tobytes()

    # 1. Resample from orig_sr -> 8000 Hz if needed
    if orig_sr != 8000:
        raw_pcm_8k, _ = audioop.ratecv(raw_pcm, 2, 1, orig_sr, 8000, None)
    else:
        raw_pcm_8k = raw_pcm

    # 2. Encode to mu-law
    return audioop.lin2ulaw(raw_pcm_8k, 2)


def ulaw_to_base64_payload(ulaw_bytes: bytes) -> str:
    """Encode mu-law bytes to base64 string for Twilio 'media' event."""
    return base64.b64encode(ulaw_bytes).decode("ascii")


def base64_payload_to_ulaw(payload_str: str) -> bytes:
    """Decode base64 string from Twilio 'media' event to mu-law bytes."""
    return base64.b64decode(payload_str)
