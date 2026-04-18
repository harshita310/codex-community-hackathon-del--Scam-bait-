import asyncio

from openai import AsyncOpenAI

from app.config import LLM_MODEL, OPENAI_API_KEY


class LLMClient:
    """Thin async wrapper around the configured OpenAI chat model."""

    def __init__(self):
        self.api_key = OPENAI_API_KEY or ""
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None

    async def generate(self, prompt: str) -> str:
        if not self.client:
            return "I am confused..."

        try:
            result = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=15,
            )
            message = result.choices[0].message.content
            if isinstance(message, str):
                return message
            return "I am confused..."
        except asyncio.TimeoutError:
            return "Hello? Are you there?"
        except Exception:
            return "I am confused..."
