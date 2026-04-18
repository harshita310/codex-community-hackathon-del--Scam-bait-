# Deploy Telegram Bot to Render (Manual)

Since the API and Dashboard are already deployed, we just need to add the Bot as a Background Worker.

## Steps

### 1. Go to Render Dashboard
- Navigate to: https://dashboard.render.com/

### 2. Create New Background Worker
- Click **New +** → **Background Worker**

### 3. Connect Repository
- Select your repository: `somewherelostt/scam_bait_bot_tts_web`
- Branch: `main`

### 4. Configure Service
Fill in the following:

| Field | Value |
|-------|-------|
| **Name** | `honey-bot` |
| **Region** | Singapore (or same as your API) |
| **Environment** | Python |
| **Build Command** | `pip install -r requirements.txt && pip install -r bot/requirements.txt` |
| **Start Command** | `python run_bot.py` |

### 5. Add Environment Variables

Click **Add Environment Variable** for each:

| Key | Value |
|-----|-------|
| `DATABASE_URL` | `postgresql://postgres:eEK9zyukPFolHiNZ@db.tyjqrnemtihgwppkcsgs.supabase.co:5432/postgres` |
| `TELEGRAM_BOT_TOKEN` | `8229177567:AAFhlrMxT0JnKA9eoYie9OdmslUT-8NSs4k` |
| `API_BASE_URL` | `https://honey-api-wr74.onrender.com` |
| `GROQ_API_KEY` | Your Groq API key |

### 6. Deploy
- Click **Create Background Worker**
- Wait for build to complete

### 7. Verify
- Check the logs to ensure the bot started successfully
- Send `/start` to your Telegram bot to test

## Troubleshooting

If the bot doesn't respond:
1. Check Render logs for errors
2. Verify `TELEGRAM_BOT_TOKEN` is correct
3. Ensure `API_BASE_URL` points to your deployed API

## Troubleshooting
Check logs if webhook fails to bind.
