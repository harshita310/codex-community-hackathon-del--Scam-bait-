

import os
from dotenv import load_dotenv


load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

API_KEY = os.getenv("API_KEY") or os.getenv("HACKATHON_API_KEY") or "temp-key"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "temp-key")


# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Fallback to local SQLite if no URL provided
    DB_PATH = os.getenv("DATABASE_PATH", "honeypot.db")
    DATABASE_URL = f"sqlite:///{DB_PATH}"

DATABASE_PATH = "honeypot.db" # Keep for potential legacy reference, but mostly unused now

#Primary LLM
LLM_PROVIDER = "cerebras"  # Primary
LLM_MODEL = "zai-glm-4.7"

# Fallback
FALLBACK_PROVIDER = "groq"
FALLBACK_MODEL = "llama-3.1-8b-instant"


MODE = os.getenv("MODE", "prod")
if MODE == "dev":
    print("Running in DEV mode - callbacks disabled")
else:
    print("Running in PROD mode - callbacks enabled")
print("Configuration loaded successfully")