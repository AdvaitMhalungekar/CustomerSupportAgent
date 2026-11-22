import os

# Ensure Phoenix uses a writable local path inside workspace
if "PHOENIX_WORKING_DIR" not in os.environ:
    os.environ["PHOENIX_WORKING_DIR"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".phoenix")

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Groq Settings (OpenAI-compatible)
    GROQ_API_KEY: str = Field(default="", description="Groq API Key from console.groq.com")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_FAST_MODEL: str = "openai/gpt-oss-20b"
    GROQ_ESCALATION_MODEL: str = "openai/gpt-oss-120b"
    GROQ_ASR_MODEL: str = "whisper-large-v3-turbo"

    # Audio & Turn Taking
    USE_SEMANTIC_TURN_DETECTOR: bool = False
    SILENCE_TIMEOUT_MS: int = 600
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    AUDIO_CHUNK_MS: int = 30  # 30ms frames for Silero VAD (480 samples @ 16kHz)

    # Observability
    PHOENIX_COLLECTOR_ENDPOINT: str = "http://localhost:6006"
    PHOENIX_PROJECT_NAME: str = "ai-adventures-voice-support"

    # File paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    KB_PATH: str = os.path.join(DATA_DIR, "knowledge_base.md")
    ENQUIRIES_PATH: str = os.path.join(DATA_DIR, "enquiries.json")

    # Budget & Limits
    MAX_REACT_STEPS: int = 4
    TTFA_TARGET_MS: int = 1500

settings = Settings()
