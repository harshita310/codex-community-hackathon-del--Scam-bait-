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


# Primary provider credentials.
OPENAI_API_KEY = _get_env("OPENAI_API_KEY")


# Voice / telephony credentials retained for the existing Twilio architecture.
TWILIO_ACCOUNT_SID = _get_env("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = _get_env("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = _get_env("TWILIO_PHONE_NUMBER")


# Database configuration.
DATABASE_PATH = _get_env("DATABASE_PATH", "honeypot.db")
DATABASE_URL = _get_env("DATABASE_URL") or f"sqlite:///{DATABASE_PATH}"


# OpenAI-first runtime configuration.
# The roadmap originally referenced older GPT-4o / Whisper / TTS-1 aliases.
# We keep the compatibility variable names while defaulting to the current
# official model equivalents for each capability.
LLM_PROVIDER = "openai"
LLM_MODEL = _get_env("LLM_MODEL", _get_env("OPENAI_CHAT_MODEL", "gpt-5.4"))
OPENAI_CHAT_MODEL = LLM_MODEL
OPENAI_DETECTION_MODEL = _get_env("OPENAI_DETECTION_MODEL", _get_env("OPENAI_CLASSIFIER_MODEL", "gpt-5.4-mini"))
OPENAI_EXTRACTION_MODEL = _get_env("OPENAI_EXTRACTION_MODEL", LLM_MODEL)
WHISPER_MODEL = _get_env("WHISPER_MODEL", _get_env("OPENAI_STT_MODEL", "gpt-4o-transcribe"))
OPENAI_STT_MODEL = WHISPER_MODEL
TTS_MODEL = _get_env("TTS_MODEL", _get_env("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"))
OPENAI_TTS_MODEL = TTS_MODEL
TTS_VOICE = _get_env("TTS_VOICE", _get_env("OPENAI_TTS_VOICE", "nova"))
OPENAI_TTS_VOICE = TTS_VOICE
EMBEDDING_MODEL = _get_env("EMBEDDING_MODEL", _get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
OPENAI_EMBEDDING_MODEL = EMBEDDING_MODEL
OPENAI_VISION_MODEL = _get_env("OPENAI_VISION_MODEL", LLM_MODEL)
OPENAI_REALTIME_MODEL = _get_env("OPENAI_REALTIME_MODEL", "gpt-realtime")
OPENAI_REASONING_EFFORT = _get_env("OPENAI_REASONING_EFFORT", "none")
OPENAI_VERBOSITY = _get_env("OPENAI_VERBOSITY", "low")
LLM_TIMEOUT_SECONDS = int(_get_env("LLM_TIMEOUT_SECONDS", "12"))


# Legacy provider secrets are intentionally excluded from the runtime config.


MODE = _get_env("MODE", "prod").lower()
CALLBACKS_ENABLED = not _get_bool("DISABLE_CALLBACKS", default=(MODE == "dev"))

if MODE == "dev":
    print("Running in DEV mode - callbacks disabled by default")
else:
    print("Running in PROD mode - callbacks enabled unless disabled explicitly")

print(f"Configuration loaded successfully (provider={LLM_PROVIDER}, model={LLM_MODEL})")
