from pathlib import Path
import asyncio
import logging
import wave

from app.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_MODEL_ID,
    ELEVENLABS_VOICE_ID,
    OPENAI_API_KEY,
    OPENAI_STT_PROMPT,
    TTS_MODEL,
    TTS_PROVIDER,
    TTS_SPEED,
    TTS_VOICE,
    WHISPER_MODEL,
)

try:
    from app.utils import logger
except ModuleNotFoundError:
    logger = logging.getLogger(__name__)


_OPENAI_CLIENT = None


async def _run_in_threadpool(func, *args, **kwargs):
    try:
        from fastapi.concurrency import run_in_threadpool

        return await run_in_threadpool(func, *args, **kwargs)
    except ModuleNotFoundError:
        return await asyncio.to_thread(func, *args, **kwargs)


def _require_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required for voice services.")
        from openai import OpenAI

        _OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    return _OPENAI_CLIENT


def _build_transcription_kwargs(audio_file_path: str, audio_file=None) -> dict:
    return {
        "model": WHISPER_MODEL,
        "file": audio_file if audio_file is not None else audio_file_path,
        "prompt": OPENAI_STT_PROMPT,
    }


def _build_openai_speech_kwargs(text: str, speed: float = TTS_SPEED) -> dict:
    return {
        "model": TTS_MODEL,
        "voice": TTS_VOICE,
        "input": text,
        "response_format": "wav",
        "speed": speed,
    }


def _build_elevenlabs_payload(text: str, speed: float = TTS_SPEED) -> dict:
    return {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.55,
            "similarity_boost": 0.8,
            "style": 0.25,
            "use_speaker_boost": True,
            "speed": speed,
        },
    }


def _transcribe_audio_sync(audio_file_path: str) -> str:
    client = _require_client()
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            **_build_transcription_kwargs(audio_file_path, audio_file)
        )

    if isinstance(transcript, str):
        return transcript
    return getattr(transcript, "text", "")


async def transcribe_audio(audio_file_path: str) -> str:
    logger.info("Transcribing audio with model=%s from %s", WHISPER_MODEL, audio_file_path)
    return await _run_in_threadpool(_transcribe_audio_sync, audio_file_path)


def _synthesize_openai_speech_sync(text: str, output_path: str) -> str:
    client = _require_client()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        **_build_openai_speech_kwargs(text)
    ) as response:
        response.stream_to_file(str(output_file))

    return str(output_file)


def _synthesize_elevenlabs_speech_sync(text: str, output_path: str) -> str:
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is required when TTS_PROVIDER=elevenlabs.")
    import requests

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream",
        params={"output_format": "pcm_16000"},
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "accept": "audio/pcm",
            "content-type": "application/json",
        },
        json=_build_elevenlabs_payload(text),
        timeout=30,
    )
    response.raise_for_status()

    with wave.open(str(output_file), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(response.content)

    return str(output_file)


def _synthesize_speech_sync(text: str, output_path: str) -> str:
    if TTS_PROVIDER == "elevenlabs":
        return _synthesize_elevenlabs_speech_sync(text, output_path)
    return _synthesize_openai_speech_sync(text, output_path)


async def synthesize_speech(text: str, output_path: str) -> str:
    logger.info(
        "Synthesizing speech with provider=%s model=%s voice=%s speed=%.2f",
        TTS_PROVIDER,
        ELEVENLABS_MODEL_ID if TTS_PROVIDER == "elevenlabs" else TTS_MODEL,
        ELEVENLABS_VOICE_ID if TTS_PROVIDER == "elevenlabs" else TTS_VOICE,
        TTS_SPEED,
    )
    return await _run_in_threadpool(_synthesize_speech_sync, text, output_path)
