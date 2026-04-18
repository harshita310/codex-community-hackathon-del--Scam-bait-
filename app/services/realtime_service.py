from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_REALTIME_MODEL, TTS_VOICE
from app.utils import logger


_OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def build_realtime_session(voice: str | None = None) -> dict:
    return {
        "type": "realtime",
        "model": OPENAI_REALTIME_MODEL,
        "voice": voice or TTS_VOICE,
        "instructions": (
            "You are ScamBait AI speaking as a confused elderly target. "
            "Stay in character, ask clarifying questions, and never share real credentials."
        ),
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
    }


def create_realtime_client_secret(voice: str | None = None) -> dict:
    if _OPENAI_CLIENT is None:
        logger.warning("Realtime session requested but OPENAI_API_KEY is missing.")
        return {}

    try:
        response = _OPENAI_CLIENT.realtime.client_secrets.create(
            session=build_realtime_session(voice=voice),
        )
        return response.model_dump()
    except Exception as e:
        logger.error(f"Realtime client secret creation failed: {e}", exc_info=True)
        return {}
