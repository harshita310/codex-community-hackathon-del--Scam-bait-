"""
Bot Configuration
Stores all configuration for the Telegram bot demo service.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================
# TELEGRAM BOT SETTINGS
# ============================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")

# ============================================
# HONEYPOT API SETTINGS
# ============================================

# Get base API URL from environment (set by Render).
# Default to localhost for local development instead of a stale hosted URL.
DEFAULT_API_URL = "http://localhost:8000"
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_URL).rstrip("/")

print(f"Bot Configuration: Using API_BASE_URL={API_BASE_URL}")

# Allow direct endpoint configuration for deployments where only the full API
# path is known up front.
HONEYPOT_API_URL = os.getenv("HONEYPOT_API_URL", f"{API_BASE_URL}/api/v1/honeypot")

# Get API key
HONEYPOT_API_KEY = os.getenv("HONEYPOT_API_KEY") or os.getenv("API_KEY", "")

if not HONEYPOT_API_KEY:
    raise ValueError("HONEYPOT_API_KEY or API_KEY not found in environment variables")

# ============================================
# DEMO SETTINGS
# ============================================

# Maximum concurrent demo sessions per user
MAX_SESSIONS_PER_USER = 1

# Session timeout (in seconds)
SESSION_TIMEOUT = 600  # 10 minutes

# Rate limiting
MAX_MESSAGES_PER_MINUTE = 10

# ============================================
# VISUALIZATION SETTINGS
# ============================================

# Enable/disable visualization features
ENABLE_VISUALIZATIONS = True

# Chart colors
CHART_COLORS = {
    "scam": "#FF4444",
    "safe": "#44FF44",
    "suspicious": "#FFAA44"
}

# ============================================
# LOGGING
# ============================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "bot/logs/bot.log"
