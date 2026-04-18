import os

from dotenv import load_dotenv


load_dotenv()


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip()
    return value or default


def _get_bool(name: str, default: bool = False) -> bool:
    value = _get_env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


API_KEY = _get_env("API_KEY") or _get_env("HACKATHON_API_KEY") or "temp-key"


# Provider API keys
OPENAI_API_KEY = _get_env("OPENAI_API_KEY")
GROQ_API_KEY = _get_env("GROQ_API_KEY")
CEREBRAS_API_KEY = _get_env("CEREBRAS_API_KEY")


# Voice provider secrets kept here for the upcoming service migration
TWILIO_ACCOUNT_SID = _get_env("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = _get_env("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = _get_env("TWILIO_PHONE_NUMBER")


# Database configuration
DATABASE_PATH = _get_env("DATABASE_PATH", "honeypot.db")
DATABASE_URL = _get_env("DATABASE_URL") or f"sqlite:///{DATABASE_PATH}"


# OpenAI-first model defaults
OPENAI_CHAT_MODEL = _get_env("OPENAI_CHAT_MODEL", _get_env("LLM_MODEL", "gpt-4o"))
OPENAI_STT_MODEL = _get_env("OPENAI_STT_MODEL", "whisper-1")
OPENAI_TTS_MODEL = _get_env("OPENAI_TTS_MODEL", "tts-1-hd")
OPENAI_TTS_VOICE = _get_env("OPENAI_TTS_VOICE", "alloy")
OPENAI_EMBEDDING_MODEL = _get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_VISION_MODEL = _get_env("OPENAI_VISION_MODEL", OPENAI_CHAT_MODEL)
OPENAI_REALTIME_MODEL = _get_env("OPENAI_REALTIME_MODEL", "")


# Runtime provider selection used by the current agents
LLM_PROVIDER = _get_env("LLM_PROVIDER", "openai").lower()
LLM_MODEL = _get_env("LLM_MODEL", OPENAI_CHAT_MODEL)
FALLBACK_PROVIDER = _get_env("FALLBACK_PROVIDER", "none").lower()
FALLBACK_MODEL = _get_env("FALLBACK_MODEL", OPENAI_CHAT_MODEL)
LLM_TIMEOUT_SECONDS = int(_get_env("LLM_TIMEOUT_SECONDS", "12"))


MODE = _get_env("MODE", "prod").lower()
CALLBACKS_ENABLED = not _get_bool("DISABLE_CALLBACKS", default=(MODE == "dev"))

if MODE == "dev":
    print("Running in DEV mode - callbacks disabled by default")
else:
    print("Running in PROD mode - callbacks enabled unless disabled explicitly")

print(f"Configuration loaded successfully (provider={LLM_PROVIDER}, model={LLM_MODEL})")
