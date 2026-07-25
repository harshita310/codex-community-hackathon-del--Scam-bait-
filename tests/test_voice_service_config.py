import unittest

from app.services.voice_service import (
    _build_elevenlabs_payload,
    _build_openai_speech_kwargs,
    _build_transcription_kwargs,
)
from app.services import voice_service


class VoiceServiceConfigTests(unittest.TestCase):
    def test_transcription_uses_multilingual_prompt_without_forced_language(self):
        kwargs = _build_transcription_kwargs("input.wav")

        self.assertNotIn("language", kwargs)
        self.assertIn("Hindi", kwargs["prompt"])
        self.assertIn("Hinglish", kwargs["prompt"])

    def test_openai_speech_uses_configured_speed(self):
        kwargs = _build_openai_speech_kwargs("hello", 0.82)

        self.assertEqual(kwargs["speed"], 0.82)
        self.assertEqual(kwargs["response_format"], "wav")

    def test_elevenlabs_payload_uses_voice_speed(self):
        payload = _build_elevenlabs_payload("namaste", 0.82)

        self.assertEqual(payload["text"], "namaste")
        self.assertEqual(payload["voice_settings"]["speed"], 0.82)

    def test_elevenlabs_failure_falls_back_to_openai_tts(self):
        original_provider = voice_service.TTS_PROVIDER
        original_elevenlabs = voice_service._synthesize_elevenlabs_speech_sync
        original_openai = voice_service._synthesize_openai_speech_sync

        calls = []

        def fail_elevenlabs(text: str, output_path: str) -> str:
            calls.append(("elevenlabs", text, output_path))
            raise RuntimeError("401 unauthorized")

        def use_openai(text: str, output_path: str) -> str:
            calls.append(("openai", text, output_path))
            return output_path

        try:
            voice_service.TTS_PROVIDER = "elevenlabs"
            voice_service._synthesize_elevenlabs_speech_sync = fail_elevenlabs
            voice_service._synthesize_openai_speech_sync = use_openai

            result = voice_service._synthesize_speech_sync("hello", "fallback.wav")
        finally:
            voice_service.TTS_PROVIDER = original_provider
            voice_service._synthesize_elevenlabs_speech_sync = original_elevenlabs
            voice_service._synthesize_openai_speech_sync = original_openai

        self.assertEqual(result, "fallback.wav")
        self.assertEqual(
            calls,
            [
                ("elevenlabs", "hello", "fallback.wav"),
                ("openai", "hello", "fallback.wav"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
