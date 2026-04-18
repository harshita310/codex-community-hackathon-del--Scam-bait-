from cerebras.cloud.sdk import Cerebras
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("CEREBRAS_API_KEY")
if not api_key:
    print("Error: CEREBRAS_API_KEY not found")
    exit(1)

try:
    client = Cerebras(api_key=api_key)
    models = client.models.list()
    print("Available Models:")
    for model in models.data:
        print(f"- {model.id}")
except Exception as e:
    print(f"Error listing models: {e}")
