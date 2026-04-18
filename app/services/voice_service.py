from pathlib import Path

from fastapi.concurrency import run_in_threadpool
from openai import OpenAI

from app.config import OPENAI_API_KEY, TTS_MODEL, TTS_VOICE, WHISPER_MODEL
from app.utils import logger


_OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def _require_client() -> OpenAI:
    if _OPENAI_CLIENT is None:
        raise RuntimeError("OPENAI_API_KEY is required for voice services.")
    return _OPENAI_CLIENT


def _transcribe_audio_sync(audio_file_path: str) -> str:
    client = _require_client()
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
            language="hi",
        )

    if isinstance(transcript, str):
        return transcript
    return getattr(transcript, "text", "")


async def transcribe_audio(audio_file_path: str) -> str:
    logger.info("Transcribing audio with model=%s from %s", WHISPER_MODEL, audio_file_path)
    return await run_in_threadpool(_transcribe_audio_sync, audio_file_path)


def _synthesize_speech_sync(text: str, output_path: str) -> str:
    client = _require_client()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
    ) as response:
        response.stream_to_file(str(output_file))

    return str(output_file)


async def synthesize_speech(text: str, output_path: str) -> str:
    logger.info("Synthesizing speech with model=%s voice=%s", TTS_MODEL, TTS_VOICE)
    return await run_in_threadpool(_synthesize_speech_sync, text, output_path)
