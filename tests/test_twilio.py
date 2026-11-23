"""Unit tests for Twilio integration (endpoints, codec, and TwiML)."""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.audio.twilio_codec import (
    ulaw_to_pcm16_16k,
    pcm16_to_ulaw_8k,
    ulaw_to_base64_payload,
    base64_payload_to_ulaw,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_twilio_codec_roundtrip():
    """Verify mu-law 8kHz to linear PCM 16kHz and back conversion."""
    # Generate 16kHz 16-bit linear PCM audio
    samples_16k = np.random.randint(-1500, 1500, size=1600, dtype=np.int16)

    # 16k -> 8k mu-law
    ulaw_bytes = pcm16_to_ulaw_8k(samples_16k, orig_sr=16000)
    assert len(ulaw_bytes) == 800  # 1600 samples at 16k = 800 samples at 8k

    # Base64 serialization
    b64 = ulaw_to_base64_payload(ulaw_bytes)
    assert isinstance(b64, str)
    ulaw_back = base64_payload_to_ulaw(b64)
    assert ulaw_back == ulaw_bytes

    # 8k mu-law -> 16k linear PCM
    pcm_back = ulaw_to_pcm16_16k(ulaw_back)
    assert len(pcm_back) in (1599, 1600)
    assert pcm_back.dtype == np.int16


def test_twiml_endpoint(client):
    """Verify /twilio/twiml returns valid XML with stream URL."""
    res = client.get("/twilio/twiml")
    assert res.status_code == 200
    assert "application/xml" in res.headers["content-type"]
    assert "<Response>" in res.text
    assert "<Stream url=" in res.text
    assert "/ws/twilio" in res.text


def test_outbound_call_endpoint_validation(client):
    """Verify /api/call validates input phone number and configs."""
    # When public_url contains 'your-ngrok', it should prompt the user
    res = client.post("/api/call", json={"phone_number": "+919876543210"})
    assert res.status_code in (400, 500)
    assert "PUBLIC_URL" in res.json().get("error", "") or "Twilio" in res.json().get("error", "")
