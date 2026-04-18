# ScamBait AI OpenAI Edition

ScamBait AI is an AI honeypot that engages scammers using an elderly persona, extracts their payment and contact infrastructure, and generates structured evidence for review. This repo preserves the original FastAPI + LangGraph + Telegram + Twilio architecture while replacing the old provider stack with OpenAI APIs.

## OpenAI APIs Used

| API | Purpose in this project |
| --- | --- |
| `Chat / Responses` | Persona replies, scam detection fallback, structured reasoning, vision analysis |
| `Function Calling` | Typed scam classification with `is_scam`, `scam_type`, confidence, and extracted entities |
| `Structured Outputs` | Normalized extraction of financial identifiers from messy scam conversations |
| `Speech-to-Text` | Voice-call transcription through the OpenAI transcription endpoint |
| `Text-to-Speech` | Generated scam-bait voice responses for the Twilio voice path |
| `Vision` | Analysis of QR codes, fake notices, spoofed logos, and image-based scam artifacts |
| `Embeddings` | Semantic scam-memory matching for scam variants that keyword rules miss |
| `Realtime API` | Bonus helper for low-latency voice session setup |

## Current OpenAI Runtime Defaults

The original migration brief referenced `gpt-4o`, `whisper-1`, and `tts-1-hd`. This repo keeps the same architecture and capability mapping, but the runtime defaults now point at current OpenAI equivalents through environment variables:

- `LLM_MODEL=gpt-5.4`
- `OPENAI_DETECTION_MODEL=gpt-5.4-mini`
- `WHISPER_MODEL=gpt-4o-transcribe`
- `TTS_MODEL=gpt-4o-mini-tts`
- `OPENAI_EMBEDDING_MODEL=text-embedding-3-small`
- `OPENAI_REALTIME_MODEL=gpt-realtime`

If you need the exact hackathon-era model names, you can override them in `.env`.

## Architecture

```text
Scammer Message / Call / Image
        |
        v
FastAPI entrypoint
        |
        v
LangGraph workflow
  load_session
    -> detection (rules + TF-IDF/SVM + OpenAI function calling)
    -> vision_check (optional image analysis)
    -> persona (elderly scam-bait response)
    -> extraction (regex + OpenAI structured outputs)
    -> save_session / callback
        |
        +-> Telegram bot
        +-> Twilio voice path
        +-> Dashboard / stats
```

## Migration Table

| Area | Old Stack | New Stack |
| --- | --- | --- |
| Persona LLM | Cerebras / Groq | OpenAI chat model via `langchain-openai` |
| Detection fallback | LangChain text fallback | OpenAI function calling |
| Extraction | Regex only | Regex + OpenAI structured outputs |
| STT | Deepgram | OpenAI transcription API |
| TTS | ElevenLabs | OpenAI speech API |
| Image analysis | None | OpenAI vision agent |
| Scam memory | Keyword-only heuristics | OpenAI embeddings similarity |
| Low-latency voice | None | Realtime session helper |

## Repository Layout

```text
app/
  agents/
    detection.py
    extraction.py
    persona.py
    timeline.py
    vision.py
  services/
    audio_orchestrator.py
    memory_service.py
    realtime_service.py
    voice_service.py
  workflow/
    graph.py
  config.py
  main.py
tests/
  test_api.py
  test_database.py
  test_openai_integrations.py
```

## Setup

1. Clone the repo:

```bash
git clone https://github.com/harshita310/KAIZEN.git
cd KAIZEN
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -r bot/requirements.txt
```

3. Create your env file:

```bash
copy .env.example .env
```

4. Set at least these variables:

- `OPENAI_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`

5. Start the API:

```bash
python run.py
```

6. Start the bot:

```bash
python run_bot.py
```

## Demo Guide

### Telegram persona

Send a scam-style message such as:

```text
Your account is blocked. Verify immediately.
```

Expected result: the persona replies in-character as a confused elderly target.

### Detection

Run the detection agent against:

```text
Transfer money to avoid arrest. Send to scammer@paytm immediately.
```

Expected result: `is_scam=true` with a scam type such as `DIGITAL_ARREST` or `UPI_SCAM`.

### Extraction

Pass conversation text containing obfuscated identifiers such as:

```text
Send 9 8 7 6 at paytm and transfer nine crore rupees
```

Expected result: normalized phone / UPI / amount extraction.

### Voice

Use the Twilio voice route and verify:

- inbound audio is transcribed
- the persona responds
- a speech file is generated for playback

### Vision

Call `analyze_scam_image()` with a QR code or fake notice image URL and inspect:

- `is_scam_image`
- `indicators_found`
- `confidence`
- `extracted_entities`

### Semantic memory

Test semantic matching with a paraphrase such as:

```text
Your SIM will be blocked if you do not verify right now.
```

Expected result: the memory service returns a high similarity to a known scam pattern.

## Tests

Run the OpenAI integration smoke tests with:

```bash
python -m pytest tests/test_openai_integrations.py -v
```

The tests are designed to monkeypatch OpenAI calls so they can validate parsing and wiring without hitting the live API.

## Notes

- `.env` is intentionally ignored by Git.
- Local SQLite files are ignored by Git.
- The current Twilio audio path now uses OpenAI voice services, but you may still want a stronger media-format adapter for production voice quality.
