import re


HINGLISH_WORDS = {
    "arre",
    "bank",
    "bhejo",
    "bhai",
    "haan",
    "ji",
    "jaldi",
    "karo",
    "kya",
    "madam",
    "mera",
    "mujhe",
    "nahi",
    "otp",
    "paise",
    "paisa",
    "sir",
    "theek",
    "tum",
    "upi",
}


def detect_voice_reply_style(last_message: str, metadata: dict | None = None) -> str:
    metadata = metadata or {}
    text = last_message or ""
    lower_words = set(re.findall(r"[a-z0-9@]+", text.lower()))

    if metadata.get("source") == "voice_call":
        return "HINGLISH"

    if any("\u0900" <= char <= "\u097f" for char in text):
        return "HINGLISH"

    if lower_words & HINGLISH_WORDS:
        return "HINGLISH"

    if metadata.get("language") == "Hindi":
        return "HINGLISH"

    return "ENGLISH"


def voice_reply_instruction(style: str) -> str:
    if style == "HINDI":
        return (
            "CONSTRAINT: Speak simple Hindi as the main language in a natural Indian phone-call style. "
            "Use Devanagari Hindi where possible, but keep common words like OTP, UPI, bank, Paytm, and Google Pay as-is. "
            "Do not switch to pure English. "
            "Keep it to one or two short sentences and ask only one question."
        )

    if style == "HINGLISH":
        return (
            "CONSTRAINT: Speak natural Hinglish in simple Indian phone-call style. "
            "Use easy words like arre, beta, ji, nahi, kya, thoda slowly. "
            "Do not switch to pure English or formal Hindi. "
            "Keep it to one or two short sentences and ask only one question."
        )

    return (
        "CONSTRAINT: Speak simple English, but keep an Indian elderly voice. "
        "Keep it to one or two short sentences and ask only one question."
    )


def extract_spoken_text(text: str) -> str:
    match = re.search(r"\[SPOKEN\]\s*(.*?)(?:\n\s*\[|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def is_repetitive_response(candidate: str, conversation_history: list[dict]) -> bool:
    normalized_candidate = _normalize(candidate)
    if not normalized_candidate:
        return True

    recent_ai_messages = [
        _normalize(msg.get("text", ""))
        for msg in conversation_history[-6:]
        if msg.get("sender") == "ai"
    ]
    return normalized_candidate in recent_ai_messages


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()
