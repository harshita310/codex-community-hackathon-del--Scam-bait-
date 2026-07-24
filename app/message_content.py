from typing import Any


def message_content_to_text(content: Any) -> str:
    """Normalize OpenAI/LangChain message content into plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(message_content_to_text(item) for item in content)

    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), (str, list, dict)):
            return message_content_to_text(content["content"])

    return ""
