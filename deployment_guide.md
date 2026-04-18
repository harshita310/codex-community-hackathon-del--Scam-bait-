# Deployment Guide: ScamBait AI on Render

This guide covers deploying the backend API, Telegram bot, and dashboard from this repository to Render.

## 1. Prepare the Repository

1. Make sure the latest changes are pushed to GitHub.
2. Confirm the repository is public if you need it for hackathon submission.
3. Confirm `.env` is not committed and only `.env.example` is tracked.

## 2. Required Environment Variables

Render will prompt for the variables marked as `sync: false` in [render.yaml](C:\Users\lenovo\OneDrive\Desktop\KAIZEN\render.yaml). At minimum, prepare:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required for persona, detection, extraction, vision, embeddings, speech, and realtime helpers |
| `API_KEY` | Shared backend auth key used by the bot and API |
| `DATABASE_URL` | Optional hosted database connection string; if omitted locally, SQLite is used |
| `TELEGRAM_BOT_TOKEN` | BotFather token for the Telegram service |
| `API_BASE_URL` or `HONEYPOT_API_URL` | Bot-to-backend endpoint |
| `PRODUCTION_BOT_URL` | Public Render URL for the bot webhook service |
| `TWILIO_ACCOUNT_SID` | Required for the Twilio voice route |
| `TWILIO_AUTH_TOKEN` | Required for the Twilio voice route |
| `TWILIO_PHONE_NUMBER` | Required for the Twilio voice route |
| `_CALLBACK_URL` | Optional external callback target |
| `VITE_API_URL` | Dashboard API base URL |

## 3. Deploy with Render Blueprint

1. Open [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** -> **Blueprint**.
3. Connect the repository `harshita310/KAIZEN`.
4. Let Render detect `render.yaml`.
5. Fill in the required environment variables.
6. Click **Apply**.

This blueprint deploys:

- `honey-api` for the FastAPI backend
- `honey-bot` for the Telegram bot webhook service
- `honey-dashboard` for the frontend dashboard

## 4. Verify the Deployment

After the build completes, verify these endpoints:

1. Backend health: `https://<your-api-service>.onrender.com/health`
2. Bot health: `https://<your-bot-service>.onrender.com/health`
3. Dashboard root: `https://<your-dashboard-service>.onrender.com/`

Then test the product flows:

1. Send `/start` to the Telegram bot.
2. Send a sample scam-style message and confirm the bot replies.
3. Open the dashboard and confirm it can reach the API.
4. If Twilio is configured, place a test call and confirm `/voice/incoming` responds.

## 5. Troubleshooting

- If the API fails to boot, check the Render logs and confirm `OPENAI_API_KEY` is set.
- If the bot fails to respond, confirm `PRODUCTION_BOT_URL`, `API_BASE_URL`, and `API_KEY` are all aligned.
- If the dashboard cannot load data, confirm `VITE_API_URL` points to the live API service.
- If voice fails, confirm the Twilio variables are present and that the public voice webhook points to the deployed backend.
