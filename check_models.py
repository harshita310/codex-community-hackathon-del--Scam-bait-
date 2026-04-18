import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not found")
    raise SystemExit(1)


client = OpenAI(api_key=api_key)

try:
    models = client.models.list()
    print("Available OpenAI models:")
    for model in models.data:
        print(f"- {model.id}")
except Exception as exc:
    print(f"Error listing models: {exc}")
