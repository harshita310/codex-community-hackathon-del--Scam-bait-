import unittest

from app.message_content import message_content_to_text


class MessageContentTests(unittest.TestCase):
    def test_returns_plain_string_content(self):
        self.assertEqual(message_content_to_text("hello"), "hello")

    def test_extracts_text_from_responses_api_blocks(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "output_text", "text": " second"},
        ]

        self.assertEqual(message_content_to_text(content), "first second")

    def test_extracts_nested_output_text_block(self):
        content = [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}]

        self.assertEqual(message_content_to_text(content), "hello")


if __name__ == "__main__":
    unittest.main()
