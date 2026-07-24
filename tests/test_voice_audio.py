import math
import tempfile
import unittest
import wave
from pathlib import Path

from app.services.voice_audio import is_probable_speech, twilio_mulaw_to_wav, wav_to_twilio_mulaw


class VoiceAudioConversionTests(unittest.TestCase):
    def test_wraps_twilio_mulaw_as_linear_pcm_wav_for_transcription(self):
        mulaw_payload = bytes([0xFF]) * 160

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "input.wav"
            twilio_mulaw_to_wav(mulaw_payload, str(output_path))

            with wave.open(str(output_path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 8000)
                self.assertEqual(wav_file.getnframes(), 160)

    def test_converts_wav_file_to_raw_twilio_mulaw_payload(self):
        sample_rate = 24000
        samples = []
        for index in range(sample_rate // 10):
            value = int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            samples.append(value.to_bytes(2, "little", signed=True))

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "tts.wav"
            with wave.open(str(input_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(b"".join(samples))

            mulaw_payload = wav_to_twilio_mulaw(str(input_path))

        self.assertIsInstance(mulaw_payload, bytes)
        self.assertGreater(len(mulaw_payload), 0)
        self.assertLess(len(mulaw_payload), len(b"".join(samples)))

    def test_detects_silent_mulaw_as_not_speech(self):
        self.assertFalse(is_probable_speech(bytes([0xFF]) * 8000))

    def test_detects_loud_mulaw_as_probable_speech(self):
        self.assertTrue(is_probable_speech(bytes([0x00, 0x80]) * 4000))


if __name__ == "__main__":
    unittest.main()
