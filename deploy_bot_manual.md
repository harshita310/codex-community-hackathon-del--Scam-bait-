# Deploy Telegram Bot to Render

Use this guide if you want to deploy the Telegram bot separately instead of via the full Render blueprint.

## Steps

### 1. Open Render
- Go to [Render Dashboard](https://dashboard.render.com/).

### 2. Create a New Web Service
- Click **New +** -> **Web Service**
- Connect the repository: `harshita310/KAIZEN`
- Branch: `main`

### 3. Configure the Service

| Field | Value |
|-------|-------|
| **Name** | `honey-bot` |
| **Region** | Singapore (or the same region as the API) |
| **Environment** | Python |
| **Build Command** | `pip install -r requirements.txt && pip install -r bot/requirements.txt` |
| **Start Command** | `python run_bot.py` |

### 4. Add Environment Variables

Set these in Render with your own values:

| Key | Value |
|-----|-------|
| `TELEGRAM_BOT_TOKEN` | Your BotFather token |
| `API_KEY` or `HONEYPOT_API_KEY` | The shared API key used by the backend |
| `API_BASE_URL` | Your deployed API base URL, for example `https://your-api-service.onrender.com` |
| `PRODUCTION_BOT_URL` | Your deployed bot URL, for example `https://your-bot-service.onrender.com` |
| `DATABASE_URL` | Optional if the bot should share the same hosted database |

### 5. Deploy and Verify
- Click **Create Web Service**
- Wait for the service to finish building
- Open `/health` on the bot URL and confirm it returns `Bot is running! (webhook mode)`
- Send `/start` to your Telegram bot and confirm it replies

## Troubleshooting

If the bot does not respond:

1. Check the Render logs for webhook or startup errors.
2. Confirm `PRODUCTION_BOT_URL` matches the live Render bot URL.
3. Confirm `API_BASE_URL` points to a reachable backend service.
4. Confirm the backend accepts the same `API_KEY` or `HONEYPOT_API_KEY`.
