import asyncio
import base64
import json
import tempfile
import time
import wave
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

from app.agents.persona import generate_persona_response
from app.services.voice_service import synthesize_speech, transcribe_audio
from app.utils import logger


class AudioOrchestrator:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.stream_sid: str | None = None
        self.processing_audio = False
        self.conversation_history: list[dict] = []
        self.input_buffer = bytearray()
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
            initial_text = "Hello? Who is this?"
            self.conversation_history.append({"sender": "ai", "text": initial_text})
            await self.stream_tts(initial_text)
            return

        if event == "media":
            payload = data.get("media", {}).get("payload")
            if not payload:
                return

            self.input_buffer.extend(base64.b64decode(payload))
            if not self.processing_audio and len(self.input_buffer) >= 16000:
                chunk = bytes(self.input_buffer)
                self.input_buffer.clear()
                asyncio.create_task(self.process_audio_chunk(chunk))
            return

        if event == "stop":
            logger.info("Twilio stream stopped")
            if self.input_buffer and not self.processing_audio:
                chunk = bytes(self.input_buffer)
                self.input_buffer.clear()
                await self.process_audio_chunk(chunk)
            await self.cleanup()

    async def process_audio_chunk(self, audio_chunk: bytes):
        """Transcribe buffered caller audio, run the persona, and synthesize a reply."""
        if not audio_chunk:
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
        """
        Wrap the inbound Twilio audio bytes in a WAV container so they can be
        handed to the transcription endpoint.
        """
        input_path = self.temp_dir / f"input_{int(time.time() * 1000)}.wav"
        with wave.open(str(input_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(1)
            wav_file.setframerate(8000)
            wav_file.writeframes(audio_chunk)
        return str(input_path)

    async def _send_audio_file(self, output_path: str):
        if not self.stream_sid:
            return

        audio_bytes = Path(output_path).read_bytes()
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
            await synthesize_speech(text, output_path)
            await self._send_audio_file(output_path)
        except Exception as e:
            logger.error(f"Error streaming TTS: {e}", exc_info=True)

    async def cleanup(self):
        self.processing_audio = False
        logger.info("AudioOrchestrator cleaned up")
