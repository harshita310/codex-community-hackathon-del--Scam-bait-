import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("LLM_MODEL", "gpt-5.4")

if not api_key:
    print("Error: OPENAI_API_KEY not found")
    raise SystemExit(1)


client = OpenAI(api_key=api_key)

try:
    model = client.models.retrieve(model_name)
    print(f"Configured chat model is available: {model.id}")
except Exception as exc:
    print(f"Error retrieving model '{model_name}': {exc}")
