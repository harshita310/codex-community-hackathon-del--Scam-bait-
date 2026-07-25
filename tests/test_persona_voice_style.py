import unittest

from app.persona_voice_style import detect_voice_reply_style, voice_reply_instruction


class PersonaVoiceStyleTests(unittest.TestCase):
    def test_voice_call_with_hindi_text_prefers_hinglish(self):
        style = detect_voice_reply_style("मेरा अकाउंट बंद हो गया", {"source": "voice_call"})

        self.assertEqual(style, "HINGLISH")

    def test_voice_call_with_roman_hindi_prefers_hinglish(self):
        style = detect_voice_reply_style("bhai jaldi otp bhejo", {"source": "voice_call"})

        self.assertEqual(style, "HINGLISH")

    def test_hinglish_instruction_avoids_pure_english(self):
        instruction = voice_reply_instruction("HINGLISH")

        self.assertIn("natural Hinglish", instruction)
        self.assertIn("Do not switch to pure English", instruction)


if __name__ == "__main__":
    unittest.main()
