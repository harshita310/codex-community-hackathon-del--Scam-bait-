# Deployment Guide: HoneyPot Scam Detection

This guide walks you through deploying the entire system (Frontend, Backend, Bot) to **Render** using **Supabase** as the database.

## 1. Setup Database (Supabase)

1.  **Create Project**: Go to [Supabase](https://supabase.com/), sign in, and create a "New Project".
2.  **Set Password**: Create a strong database password and **save it**.
3.  **Get Connection String**:
    - Go to **Project Settings** (cog icon) -> **Database**.
    - Under **Connection String**, select **URI**.
    - Copy the string. It looks like:
        `postgresql://postgres.xxxx:[YOUR-PASSWORD]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres`
    - **Replace `[YOUR-PASSWORD]`** with the password you created in step 2.
    - **Note this URL** as your `DATABASE_URL`.

## 2. Prepare Code

1.  **Commit & Push**: Ensure all your local changes are committed and pushed to GitHub.
    ```bash
    git add .
    git commit -m "Prepare for deployment"
    git push origin main
    ```

## 3. Deploy to Render

1.  **Create Blueprint**:
    - Go to [Render Dashboard](https://dashboard.render.com/).
    - Click **New +** -> **Blueprint**.
    - Connect your GitHub repository (`HoneyPot-Scam-Detection`).
    - Give it a name (e.g., `scambait-system`).

2.  **Configure Environment Variables**:
    Render will detect `render.yaml` and ask for the following variables. Fill them in:

    | Variable | Value |
    | :--- | :--- |
    | `DATABASE_URL` | Paste the **Supabase Connection String** from Step 1. |
    | `TELEGRAM_BOT_TOKEN` | Your Bot Token from BotFather. |
    | `GROQ_API_KEY` | Your Groq API Key. |
    | `HACKATHON_API_KEY` | Your Hackathon Key (if applicable). |
    | `LLM_MODEL` | `llama-3.1-8b-instant` (or your preferred model). |

3.  **Apply**:
    - Click **Apply**. Render will start deploying 3 services:
        - `honey-api` (Backend)
        - `honey-bot` (Telegram Bot)
        - `honey-dashboard` (Frontend)

## 4. Final Verification

1.  **Wait for Build**: It may take a few minutes.
2.  **Check Services**:
    - **API**: Visit the URL for `honey-api` (e.g., `https://honey-api.onrender.com/docs`). You should see Swagger UI.
    - **Dashboard**: Visit the URL for `honey-dashboard`. It should load and show stats (initially 0).
    - **Bot**: Send `/start` to your Telegram bot. It should respond.

## Troubleshooting

- **Database Errors**: Check the `honey-api` logs. Ensure `DATABASE_URL` is correct and you replaced the password.
- **Bot Not Responding**: Check `honey-bot` logs. Ensure `TELEGRAM_BOT_TOKEN` is correct.
- **Frontend Issues**: Ensure `VITE_API_URL` was correctly set by Render (it should happen automatically via `render.yaml`). If not, you may need to manually add `VITE_API_URL` to the **Dashboard Service** environment variables pointing to your API URL.
