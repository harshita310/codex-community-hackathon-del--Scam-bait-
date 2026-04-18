from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("CEREBRAS_API_KEY")
if not api_key:
    print("Error: CEREBRAS_API_KEY not found")
    exit(1)

# Cerebras is OpenAI compatible!
client = OpenAI(
    api_key=api_key,
    base_url="https://api.cerebras.ai/v1"
)

try:
    print("Connecting to Cerebras API...")
    models = client.models.list()
    print("\nAvailable Models:")
    for model in models.data:
        print(f"- {model.id}")
except Exception as e:
    print(f"Error listing models: {e}")
