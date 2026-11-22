import os
import json
import pytest
from app.config import settings

def test_directory_structure():
    base_dir = settings.BASE_DIR
    expected_dirs = [
        os.path.join(base_dir, "app"),
        os.path.join(base_dir, "app", "graph"),
        os.path.join(base_dir, "app", "nodes"),
        os.path.join(base_dir, "app", "tools"),
        os.path.join(base_dir, "app", "guardrails"),
        os.path.join(base_dir, "app", "audio"),
        os.path.join(base_dir, "app", "kb"),
        os.path.join(base_dir, "app", "llm"),
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "evals"),
        os.path.join(base_dir, "tests"),
    ]
    for d in expected_dirs:
        assert os.path.isdir(d), f"Missing directory: {d}"

def test_knowledge_base_seeded():
    kb_path = settings.KB_PATH
    assert os.path.isfile(kb_path), f"KB file missing: {kb_path}"
    with open(kb_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check strict seeded facts
    assert "AI Adventures" in content
    assert "Pune, Maharashtra" in content
    assert "hands-on, code-first" in content
    assert "15,000" in content
    assert "25,000" in content
    assert "TODO: confirm with AI Adventures" in content

def test_enquiries_json_seeded():
    enq_path = settings.ENQUIRIES_PATH
    assert os.path.isfile(enq_path), f"Enquiries file missing: {enq_path}"
    with open(enq_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "enquiries" in data
    assert len(data["enquiries"]) >= 3
    assert data["enquiries"][0]["enquiry_id"] == "ENQ-101"

def test_config_defaults():
    assert settings.GROQ_FAST_MODEL == "openai/gpt-oss-20b"
    assert settings.GROQ_ESCALATION_MODEL == "openai/gpt-oss-120b"
    assert settings.GROQ_ASR_MODEL == "whisper-large-v3-turbo"
    assert settings.SAMPLE_RATE == 16000
    assert settings.TTFA_TARGET_MS == 1500

def test_offline_models_downloaded():
    models_dir = settings.MODELS_DIR
    vad_path = os.path.join(models_dir, "silero_vad.onnx")
    piper_path = os.path.join(models_dir, "en_US-lessac-medium.onnx")
    piper_cfg = os.path.join(models_dir, "en_US-lessac-medium.onnx.json")

    assert os.path.isfile(vad_path) and os.path.getsize(vad_path) > 1000000
    assert os.path.isfile(piper_path) and os.path.getsize(piper_path) > 10000000
    assert os.path.isfile(piper_cfg) and os.path.getsize(piper_cfg) > 100

