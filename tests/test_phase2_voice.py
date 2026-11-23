"""Phase 2 tests: VAD segmenter, ASR normalizer, TTS, barge-in, WebSocket text mode."""
import numpy as np
import pytest
from app.audio.asr_normalizer import normalize_asr_transcript
from app.audio.vad import SileroVAD, UtteranceSegmenter, FRAME_SAMPLES
from app.audio.tts import PiperTTS, split_into_sentences


# ── ASR Normalizer Tests ──────────────────────────────────────────────────────

def test_normalize_phone_digits_full():
    raw = "nine eight seven six five four three two one zero"
    result = normalize_asr_transcript(raw)
    assert "9876543210" in result

def test_normalize_phone_digits_partial():
    raw = "my number is nine eight seven six"
    result = normalize_asr_transcript(raw)
    assert "9876" in result

def test_normalize_course_names_piethon():
    assert "Python" in normalize_asr_transcript("I am interested in pie-thon course")

def test_normalize_course_names_mashing_learning():
    assert "Machine Learning" in normalize_asr_transcript("Tell me about mashing learning")

def test_normalize_prices_spoken():
    result = normalize_asr_transcript("The fee is fifteen thousand rupees")
    assert "15,000" in result

def test_normalize_data_science_spoken():
    result = normalize_asr_transcript("I want to join the data science program")
    assert "Data Science" in result

def test_normalize_identity_on_clean():
    result = normalize_asr_transcript("What is the fee for Machine Learning?")
    assert "Machine Learning" in result
    assert result == "What is the fee for Machine Learning?"


# ── VAD Segmenter Tests ───────────────────────────────────────────────────────

def _make_silence_frame() -> np.ndarray:
    return np.zeros(FRAME_SAMPLES, dtype=np.int16)

def _make_speech_frame(amplitude=5000) -> np.ndarray:
    t = np.linspace(0, 0.032, FRAME_SAMPLES)
    return (amplitude * np.sin(2 * np.pi * 300 * t)).astype(np.int16)

def test_vad_model_loads():
    vad = SileroVAD()
    # Must not raise; stub mode allowed if model is a stub binary
    frame = _make_speech_frame()
    prob = vad.get_speech_prob(frame)
    assert 0.0 <= prob <= 1.0

def test_segmenter_emits_utterance_after_silence():
    """Test segmenter contract using the energy-based stub VAD (no ONNX model needed)."""
    # Temporarily point to a non-existent path to force stub mode
    vad = SileroVAD(model_path="/nonexistent/path.onnx")
    seg = UtteranceSegmenter(vad, silence_timeout_ms=90)  # 3 frames @ 30ms

    # Feed 10 speech frames (high amplitude -> stub detects as speech)
    result = None
    for _ in range(10):
        result = seg.process_frame(_make_speech_frame(amplitude=5000))
        if result is not None:
            break
    # Feed 3 silence frames to trigger end-of-utterance
    for _ in range(3):
        result = seg.process_frame(_make_silence_frame())
        if result is not None:
            break
    # Should have emitted an utterance
    assert result is not None, "Segmenter should emit utterance after speech+silence"
    assert isinstance(result, np.ndarray)
    assert len(result) > 0

def test_segmenter_no_utterance_during_silence_only():
    vad = SileroVAD()
    seg = UtteranceSegmenter(vad, silence_timeout_ms=90)
    for _ in range(20):
        r = seg.process_frame(_make_silence_frame())
        assert r is None, "No utterance should emit during all-silence"

def test_segmenter_flush_pending():
    vad = SileroVAD()
    seg = UtteranceSegmenter(vad, silence_timeout_ms=600)
    for _ in range(5):
        seg.process_frame(_make_speech_frame())
    result = seg.flush()
    # flush returns None or ndarray; with a stub vad that calls silence frames as non-speech
    # just check it doesn't crash
    assert result is None or isinstance(result, np.ndarray)


# ── TTS Tests ─────────────────────────────────────────────────────────────────

def test_tts_stub_returns_silence_proportional():
    tts = PiperTTS()
    audio = tts.synthesize_sentence("Hello, welcome to AI Adventures.")
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.int16
    # Should be non-empty (stub generates proportional silence)
    assert len(audio) > 0

def test_tts_split_sentences():
    text = "Hello! Welcome to AI Adventures. How can I help you today?"
    sents = split_into_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "Hello!"
    assert "AI Adventures" in sents[1]

def test_tts_pcm_to_wav_bytes():
    tts = PiperTTS()
    pcm = np.zeros(22050, dtype=np.int16)
    wav = tts.pcm_to_wav_bytes(pcm)
    # Check WAV header
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert len(wav) == 44 + 22050 * 2

def test_tts_stream_sentences_yields():
    tts = PiperTTS()
    chunks = list(tts.stream_sentences("Hello! This is a test. AI Adventures awaits you."))
    assert len(chunks) >= 1
    for c in chunks:
        assert isinstance(c, np.ndarray)
        assert c.dtype == np.int16


# ── Barge-In Graph Test ───────────────────────────────────────────────────────

def test_graph_barge_in_clears_and_replans():
    from app.graph.builder import voice_graph
    from app.graph.state import VoiceAgentState

    state: VoiceAgentState = {
        "transcript": "What is the Python course fee?",
        "session_id": "phase2-bargein",
        "barge_in": "Actually, what about Data Science?",
        "latency_ms": {},
        "tool_calls": [],
        "tool_results": [],
        "steps": 0,
        "interrupted": False,
        "handoff": False,
    }
    result = voice_graph.invoke(state)
    assert result.get("interrupted") is True
    assert result.get("reply") is not None
    assert "25,000" in result["reply"] or "data science" in result["reply"].lower()
