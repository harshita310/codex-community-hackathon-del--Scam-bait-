"""
Telegram Bot Runner - WEBHOOK + POLLING dual mode.

WEBHOOK MODE (Production/Render):
    Activated when RENDER_EXTERNAL_URL or PRODUCTION_BOT_URL is set.
    Telegram pushes updates to our /webhook endpoint.
    No getUpdates calls = zero 409 conflicts.

POLLING MODE (Local development):
    Activated when no external URL is found.
    Uses traditional getUpdates polling.
"""

import sys
import os
import asyncio
import signal

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from aiohttp import web
from telegram import Update
from bot.bot_service import create_application, logger
from bot.bot_config import HONEYPOT_API_KEY

# ============================================
# CREATE THE BOT APPLICATION
# ============================================

application = create_application()

# Webhook secret (reuse API key)
WEBHOOK_SECRET = HONEYPOT_API_KEY


# ============================================
# WEBHOOK HANDLER
# ============================================

async def webhook_handler(request):
    """Handle incoming Telegram webhook updates with detailed logging."""
    logger.info(f"Received webhook request: {request.method} {request.path}")
    
    # Log headers for debugging (safely mask secret)
    headers = dict(request.headers)
    secret = headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    safe_headers = {k: v for k, v in headers.items() if k != "X-Telegram-Bot-Api-Secret-Token"}
    logger.info(f"Headers: {safe_headers}")

    # Verify the secret token
    if secret != WEBHOOK_SECRET:
        logger.warning(f"⚠️ Webhook secret mismatch! Expected: {WEBHOOK_SECRET[:4]}... Got: {secret[:4]}...")
        return web.Response(status=403, text="Forbidden")
    else:
        logger.info("✅ Webhook secret verified")

    try:
        data = await request.json()
        logger.info(f"Payload received (size: {len(str(data))} chars)")
        
        update = Update.de_json(data, application.bot)
        
        # Log update details
        if update.message:
            logger.info(f"Processing message from user {update.effective_user.id}: {update.message.text[:20]}...")
        
        # Process the update through python-telegram-bot handlers
        await application.process_update(update)
        logger.info("Update processed successfully")
        
    except Exception as e:
        logger.error(f"❌ Webhook processing error: {e}", exc_info=True)
        return web.Response(status=500, text="Internal Server Error")

    return web.Response(text="ok")


async def health_check(request):
    """Health check endpoint for Render."""
    return web.Response(text="Bot is running! (webhook mode)")


# ============================================
# LIFECYCLE HOOKS
# ============================================

async def on_startup(app_web):
    """Initialize bot and set webhook on server start."""
    await application.initialize()
    await application.start()

    # Determine the public URL
    bot_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("PRODUCTION_BOT_URL", "")

    if bot_url:
        webhook_url = f"{bot_url}/webhook"
        await application.bot.set_webhook(
            url=webhook_url,
            secret_token=WEBHOOK_SECRET,
            drop_pending_updates=True
        )
        logger.info(f"Webhook set to: {webhook_url}")
        logger.info("✅ PATCH APPLIED: Webhook will NOT be deleted on shutdown.")
    else:
        logger.warning("No external URL found — webhook NOT registered!")
        logger.warning("Set RENDER_EXTERNAL_URL or PRODUCTION_BOT_URL to enable webhook mode.")


async def on_shutdown(app_web):
    """Graceful shutdown: delete webhook, stop application."""
    # try:
    #     await application.bot.delete_webhook()
    #     logger.info("Webhook deleted.")
    # except Exception as e:
    #     logger.error(f"Error deleting webhook: {e}")

    try:
        await application.stop()
        await application.shutdown()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

    logger.info("Bot shutdown complete.")


# ============================================
# POLLING MODE (Local Development)
# ============================================

async def run_polling():
    """Run bot in polling mode for local development."""
    await application.initialize()
    await application.start()
    await application.updater.start_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )

    logger.info("Bot is polling! Press Ctrl+C to stop.")

    # Wait for shutdown signal
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def handle_signal(sig):
        logger.info(f"🛑 Received shutdown signal: {signal.strsignal(sig) if hasattr(signal, 'strsignal') else sig}")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))
        except NotImplementedError:
            # Windows fallback
            signal.signal(sig, lambda s, f: handle_signal(s))

    await shutdown_event.wait()
    await application.updater.stop()
    await application.stop()
    await application.shutdown()


# ============================================
# MAIN
# ============================================

def main():
    """Entry point: choose webhook or polling mode based on environment."""
    api_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
    bot_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("PRODUCTION_BOT_URL", "")
    
    # On Render, we MUST bind to PORT, regardless of mode.
    # If no URL is provided, we still need a dummy server to keep Render happy.
    port = int(os.getenv('PORT', 8080))
    
    use_webhook = bool(bot_url)

    print("=" * 70)
    print("Starting ScamBait AI Telegram Bot")
    print(f"  Mode:    {'WEBHOOK' if use_webhook else 'POLLING (local dev/fallback)'}")
    print(f"  API URL: {api_url}")
    print(f"  Port:    {port}")
    if use_webhook:
        print(f"  Bot URL: {bot_url}")
        print(f"  Webhook: {bot_url}/webhook")
    print("=" * 70)

    app = web.Application()
    app.router.add_get('/', health_check)
    app.router.add_get('/health', health_check)

    if use_webhook:
        # ── WEBHOOK MODE ──
        app.router.add_post('/webhook', webhook_handler)
        app.on_startup.append(on_startup)
        app.on_shutdown.append(on_shutdown)
    else:
        # ── POLLING MODE WRAPPED IN WEB SERVER ──
        # This allows running on Render/Heroku without a public URL yet
        # while satisfying the port binding requirement.
        
        async def start_polling_background(app):
            asyncio.create_task(run_polling())
            
        app.on_startup.append(start_polling_background)

    # web.run_app handles SIGTERM/SIGINT gracefully
    web.run_app(app, host='0.0.0.0', port=port)

if __name__ == "__main__":
    main()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
