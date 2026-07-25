import unittest

from app.services.voice_service import (
    _build_elevenlabs_payload,
    _build_openai_speech_kwargs,
    _build_transcription_kwargs,
)


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


if __name__ == "__main__":
    unittest.main()
