import unittest

from app.services.voice_turns import VoiceTurnBuffer


class VoiceTurnBufferTests(unittest.TestCase):
    def test_ignores_silence_until_speech_arrives(self):
        buffer = VoiceTurnBuffer(min_speech_bytes=4, silence_bytes=4)

        self.assertIsNone(buffer.add(bytes([0xFF]) * 8))

    def test_waits_for_silence_before_returning_utterance(self):
        buffer = VoiceTurnBuffer(min_speech_bytes=4, silence_bytes=4)

        self.assertIsNone(buffer.add(bytes([0x00, 0x80, 0x00, 0x80])))
        utterance = buffer.add(bytes([0xFF]) * 4)

        self.assertEqual(utterance, bytes([0x00, 0x80, 0x00, 0x80]))

    def test_drops_tiny_noise_bursts(self):
        buffer = VoiceTurnBuffer(min_speech_bytes=8, silence_bytes=4)

        self.assertIsNone(buffer.add(bytes([0x00, 0x80])))
        self.assertIsNone(buffer.add(bytes([0xFF]) * 4))


if __name__ == "__main__":
    unittest.main()
