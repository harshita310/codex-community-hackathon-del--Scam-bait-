import unittest

from app.persona_voice_style import detect_voice_reply_style, voice_reply_instruction


class PersonaVoiceStyleTests(unittest.TestCase):
    def test_voice_call_with_hindi_text_prefers_hindi(self):
        style = detect_voice_reply_style("mera account band ho gaya", {"source": "voice_call"})

        self.assertEqual(style, "HINDI")

    def test_voice_call_with_roman_hindi_prefers_hindi(self):
        style = detect_voice_reply_style("bhai jaldi otp bhejo", {"source": "voice_call"})

        self.assertEqual(style, "HINDI")

    def test_hindi_instruction_keeps_hindi_as_main_language(self):
        instruction = voice_reply_instruction("HINDI")

        self.assertIn("simple Hindi", instruction)
        self.assertIn("main language", instruction)


if __name__ == "__main__":
    unittest.main()
