import os
import json
import base64
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from elevenlabs.client import ElevenLabs
from app.utils import logger
from app.agents.persona import generate_persona_response

class AudioOrchestrator:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.deepgram_client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
        self.elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.stream_sid = None
        self.deepgram_connection = None
        self.is_listening = False
        self.conversation_history = []

    async def start(self):
        """Initialize connections to Deepgram and start processing."""
        await self.websocket.accept()
        self.deepgram_connection = self.deepgram_client.listen.live.v("1")
        self.deepgram_connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)
        self.deepgram_connection.on(LiveTranscriptionEvents.Error, self.on_error)
        
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            encoding="mulaw",
            sample_rate=8000,
            channels=1,
            interim_results=False,
            vad_events=True,
            endpointing=500,
        )

        if not self.deepgram_connection.start(options):
            logger.error("Failed to start Deepgram connection")
            await self.websocket.close()
            return

        logger.info("Deepgram connection started")
        self.is_listening = True
        
        try:
            while True:
                message = await self.websocket.receive_text()
                data = json.loads(message)
                await self.handle_twilio_message(data)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error in AudioOrchestrator loop: {e}")
        finally:
            await self.cleanup()

    async def handle_twilio_message(self, data):
        """Handle incoming messages from Twilio via WebSocket."""
        event = data.get("event")

        if event == "start":
            self.stream_sid = data["start"]["streamSid"]
            logger.info(f"Twilio Stream started: {self.stream_sid}")
            initial_text = "Hello? Who is this?"
            self.conversation_history.append({"sender": "ai", "text": initial_text})
            asyncio.create_task(self.stream_tts(initial_text))
        
        elif event == "media":
            if self.is_listening and self.deepgram_connection:
                payload = data["media"]["payload"]
                audio_data = base64.b64decode(payload)
                self.deepgram_connection.send(audio_data)
        
        elif event == "stop":
            logger.info("Twilio Stream stopped")
            await self.cleanup()

    def on_transcript(self, sender, result, **kwargs):
        """Handle transcript received from Deepgram."""
        if not result or not result.channel.alternatives:
            return
            
        sentence = result.channel.alternatives[0].transcript
        if not sentence:
            return
            
        if result.is_final:
            logger.info(f"User said: {sentence}")
            self.conversation_history.append({"sender": "scammer", "text": sentence})
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.process_response(sentence))
            except RuntimeError:
                pass

    def on_error(self, sender, error, **kwargs):
        logger.error(f"Deepgram error: {error}")

    async def process_response(self, text):
        """Send text to Persona Agent and stream audio back."""
        try:
            metadata = {"source": "voice_call"}
            response_text = await generate_persona_response(
                conversation_history=self.conversation_history,
                metadata=metadata
            )
            logger.info(f"AI Response: {response_text}")
            self.conversation_history.append({"sender": "ai", "text": response_text})
            await self.stream_tts(response_text)
        except Exception as e:
            logger.error(f"Error in process_response: {e}")

    async def stream_tts(self, text):
        """Stream audio from ElevenLabs to Twilio."""
        try:
            audio_stream = self.elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id="eleven_turbo_v2",
                output_format="ulaw_8000"
            )

            for chunk in audio_stream:
                if chunk:
                    audio_payload = base64.b64encode(chunk).decode("utf-8")
                    media_message = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": audio_payload}
                    }
                    await self.websocket.send_json(media_message)
        except Exception as e:
            logger.error(f"Error streaming TTS: {e}")

    async def cleanup(self):
        """Close connections."""
        self.is_listening = False
        if self.deepgram_connection:
            self.deepgram_connection.finish()
        logger.info("AudioOrchestrator cleaned up")
