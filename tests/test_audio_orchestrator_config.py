import unittest

from app.voice_greeting import INITIAL_VOICE_GREETING


class AudioOrchestratorConfigTests(unittest.TestCase):
    def test_initial_voice_greeting_uses_hindi(self):
        self.assertEqual(INITIAL_VOICE_GREETING, "Aap kon?")


if __name__ == "__main__":
    unittest.main()
