import asyncio
import os
# import groq

class LLMClient:
    """Wrapper for Groq / Cerebras API calls."""
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "dummy")
        # self.client = groq.AsyncGroq(api_key=self.api_key)
        
    async def generate(self, prompt: str) -> str:
        # Timeout handling for LLM calls over poor network
        try:
            # result = await asyncio.wait_for(self.client.chat.completions.create(...), timeout=15)
            # return result.choices[0].message.content
            return "I am confused..." # placeholder
        except asyncio.TimeoutError:
            return "Hello? Are you there?"
