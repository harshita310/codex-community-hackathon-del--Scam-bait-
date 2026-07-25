import asyncio
import base64
import json
import tempfile
import time
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

from app.agents.persona import generate_persona_response
from app.services.voice_audio import is_probable_speech, twilio_mulaw_to_wav, wav_to_twilio_mulaw
from app.services.voice_service import synthesize_speech, transcribe_audio
from app.services.voice_turns import VoiceTurnBuffer
from app.utils import logger
from app.voice_greeting import INITIAL_VOICE_GREETING


class AudioOrchestrator:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.stream_sid: str | None = None
        self.processing_audio = False
        self.speaking = False
        self.conversation_history: list[dict] = []
        self.turn_buffer = VoiceTurnBuffer()
        self.temp_dir = Path(tempfile.gettempdir()) / "kaizen_voice"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def start(self):
        """Accept the websocket and process Twilio Media Stream events."""
        await self.websocket.accept()

        try:
            while True:
                message = await self.websocket.receive_text()
                data = json.loads(message)
                await self.handle_twilio_message(data)
        except WebSocketDisconnect:
            logger.info("Voice websocket disconnected")
        except Exception as e:
            logger.error(f"Error in AudioOrchestrator loop: {e}", exc_info=True)
        finally:
            await self.cleanup()

    async def handle_twilio_message(self, data: dict):
        """Handle incoming Twilio websocket events."""
        event = data.get("event")

        if event == "start":
            self.stream_sid = data["start"]["streamSid"]
            logger.info(f"Twilio stream started: {self.stream_sid}")
            initial_text = INITIAL_VOICE_GREETING
            self.conversation_history.append({"sender": "ai", "text": initial_text})
            await self.stream_tts(initial_text)
            return

        if event == "media":
            if self.speaking:
                return

            payload = data.get("media", {}).get("payload")
            if not payload:
                return

            chunk = self.turn_buffer.add(base64.b64decode(payload))
            if chunk and not self.processing_audio:
                asyncio.create_task(self.process_audio_chunk(chunk))
            return

        if event == "stop":
            logger.info("Twilio stream stopped")
            chunk = self.turn_buffer.flush()
            if chunk and not self.processing_audio:
                await self.process_audio_chunk(chunk)
            await self.cleanup()

    async def process_audio_chunk(self, audio_chunk: bytes):
        """Transcribe buffered caller audio, run the persona, and synthesize a reply."""
        if not audio_chunk:
            return
        if not is_probable_speech(audio_chunk):
            logger.info("Skipping low-energy voice chunk.")
            return

        self.processing_audio = True
        try:
            input_path = self._write_audio_chunk(audio_chunk)
            transcript = await transcribe_audio(input_path)
            if not transcript.strip():
                logger.info("Voice transcription returned empty text.")
                return

            logger.info(f"Voice transcript: {transcript}")
            self.conversation_history.append({"sender": "scammer", "text": transcript})

            response_text = await generate_persona_response(
                conversation_history=self.conversation_history,
                metadata={"source": "voice_call"},
            )
            logger.info(f"Voice persona response: {response_text}")
            self.conversation_history.append({"sender": "ai", "text": response_text})

            output_path = str(self.temp_dir / f"tts_{int(time.time() * 1000)}.wav")
            await synthesize_speech(response_text, output_path)
            await self._send_audio_file(output_path)
        except Exception as e:
            logger.error(f"Error processing voice chunk: {e}", exc_info=True)
        finally:
            self.processing_audio = False

    def _write_audio_chunk(self, audio_chunk: bytes) -> str:
        """Convert inbound Twilio mulaw audio to WAV for transcription."""
        input_path = self.temp_dir / f"input_{int(time.time() * 1000)}.wav"
        return twilio_mulaw_to_wav(audio_chunk, str(input_path))

    async def _send_audio_file(self, output_path: str):
        if not self.stream_sid:
            return

        audio_bytes = wav_to_twilio_mulaw(output_path)
        audio_payload = base64.b64encode(audio_bytes).decode("utf-8")
        media_message = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": audio_payload},
        }
        await self.websocket.send_json(media_message)

    async def stream_tts(self, text: str):
        output_path = str(self.temp_dir / f"greeting_{int(time.time() * 1000)}.wav")
        try:
            self.speaking = True
            await synthesize_speech(text, output_path)
            await self._send_audio_file(output_path)
        except Exception as e:
            logger.error(f"Error streaming TTS: {e}", exc_info=True)
        finally:
            self.turn_buffer.clear()
            self.speaking = False

    async def cleanup(self):
        self.processing_audio = False
        self.speaking = False
        self.turn_buffer.clear()
        logger.info("AudioOrchestrator cleaned up")
