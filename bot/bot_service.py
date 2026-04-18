"""
Main Telegram Bot Service
Acts as a realistic victim for the HoneyPot Scam Detection system.
Exports create_application() for use by run_bot.py.
"""

import asyncio
import logging
import sys
import os
import random
from datetime import datetime
from typing import Dict, Optional
import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.bot_config import (
    TELEGRAM_BOT_TOKEN,
    HONEYPOT_API_URL,
    HONEYPOT_API_KEY,
    LOG_LEVEL
)

# ============================================
# LOGGING SETUP
# ============================================

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, LOG_LEVEL),
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# ============================================
# SESSION MANAGEMENT
# ============================================

# Store active sessions: {user_id: {session_data}}
active_sessions: Dict[int, Dict] = {}

def get_or_create_session(user_id: int) -> str:
    """Get existing session or create a new one."""
    if user_id in active_sessions:
        return active_sessions[user_id]["session_id"]
    
    # Create new session
    session_id = f"telegram_{user_id}_{int(datetime.now().timestamp())}"
    active_sessions[user_id] = {
        "session_id": session_id,
        "started_at": datetime.now(),
        "messages": []
    }
    logger.info(f"Created new session {session_id} for user {user_id}")
    return session_id

def end_session(user_id: int):
    """End active session for user."""
    if user_id in active_sessions:
        session_id = active_sessions[user_id]["session_id"]
        del active_sessions[user_id]
        logger.info(f"Ended session {session_id} for user {user_id}")

# ============================================
# FALLBACK RESPONSES (Varied to avoid repetition)
# ============================================

FALLBACK_REPLIES = [
    "I can't read this without my glasses... wait.",
    "Is this the bank? My screen is very dark.",
    "My grandson usually handles these messages.",
    "Who is this? I don't have this number saved.",
    "I am pressing the buttons but nothing happens.",
    "Can you explain simply? I am not good with technology.",
    "What do you mean? I am very confused.",
    "I think I received the wrong message.",
    "One moment, let me find my cheque book.",
    "My phone is acting up again. The screen is flickering."
]

# ============================================
# API INTEGRATION
# ============================================

async def call_honeypot_api(session_id: str, message_text: str, retries: int = 1) -> Dict:
    """
    Call the HoneyPot API with a message.
    Includes retry logic to handle transient errors.
    """
    payload = {
        "sessionId": session_id,
        "message": {
            "sender": "scammer",
            "text": message_text,
            "timestamp": datetime.now().isoformat() + "Z"
        },
        "metadata": {
            "channel": "Telegram",
            "language": "English",
            "locale": "IN",
            "source": "REAL_MODE"
        }
    }
    
    headers = {
        "x-api-key": HONEYPOT_API_KEY,
        "Content-Type": "application/json"
    }
    
    last_error = None
    for attempt in range(1 + retries):
        try:
            async with httpx.AsyncClient(timeout=40.0) as client:
                response = await client.post(
                    HONEYPOT_API_URL,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            last_error = e
            logger.error(f"API call failed (attempt {attempt + 1}/{1 + retries}): {e}")
            if attempt < retries:
                await asyncio.sleep(2)
    
    # All retries exhausted — return varied fallback
    logger.error(f"API call failed after {1 + retries} attempts: {last_error}")
    return {
        "status": "error",
        "reply": random.choice(FALLBACK_REPLIES),
        "meta": {
            "agentState": "error",
            "sessionStatus": "active"
        }
    }

# ============================================
# HANDLERS
# ============================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await update.message.reply_text("Hello? Who is this? Do I know you?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    await update.message.reply_text("I'm sorry, I'm not good with this phone. Are you my grandson?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle ALL user messages.
    User = Scammer, Bot = AI Victim
    """
    user_id = update.effective_user.id
    message_text = update.message.text
    
    # 1. Get/Create Session
    session_id = get_or_create_session(user_id)
    
    # 2. Typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # 3. Call HoneyPot API (with retry)
    response = await call_honeypot_api(session_id, message_text)
    
    # 4. Process Response
    ai_reply = response.get("reply")
    if not ai_reply or not ai_reply.strip():
        logger.warning(f"API returned empty reply for session {session_id}. Using fallback.")
        ai_reply = random.choice(FALLBACK_REPLIES)
        
    agent_state = response.get("meta", {}).get("agentState", "engaging")
    session_status = response.get("meta", {}).get("sessionStatus", "active")
    
    # 5. Send Reply
    await update.message.reply_text(ai_reply)
    
    # 6. Check for termination
    if session_status == "closed" or agent_state == "completed":
        logger.info(f"Session {session_id} completed via API")
        end_session(user_id)

# ============================================
# APPLICATION FACTORY
# ============================================

def create_application() -> Application:
    """Create and configure the Telegram bot application with all handlers."""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    return application

# FIX: Ensure session_id uniquely ties to update.effective_chat.id to prevent cross-user session bleed
