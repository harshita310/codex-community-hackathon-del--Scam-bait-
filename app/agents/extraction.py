import re

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import OPENAI_API_KEY, OPENAI_EXTRACTION_MODEL
from app.utils import logger


_OPENAI_CLIENT = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


class ExtractedIntelligence(BaseModel):
    phone_numbers: list[str] = Field(default_factory=list)
    upi_ids: list[str] = Field(default_factory=list)
    bank_accounts: list[str] = Field(default_factory=list)
    crypto_wallets: list[str] = Field(default_factory=list)
    phishing_links: list[str] = Field(default_factory=list)
    aadhaar_numbers: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)


def normalize_before_extract(text: str) -> str:
    """Pre-process obfuscated intel before regex runs."""
    text = re.sub(r"\s+at\s+", "@", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+dot\s+", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)

    word_map = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }
    for word, digit in word_map.items():
        text = re.sub(r"\b" + word + r"\b", digit, text, flags=re.IGNORECASE)

    return text


def _join_conversation(conversation_history: list) -> str:
    return " ".join(msg.get("text", "") for msg in conversation_history if isinstance(msg, dict))


def _merge_values(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            normalized = item.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(normalized)
    return merged


async def extract_with_openai(conversation_history: list) -> ExtractedIntelligence:
    if _OPENAI_CLIENT is None:
        logger.warning("OpenAI extraction requested but OPENAI_API_KEY is missing.")
        return ExtractedIntelligence()

    conversation_text = _join_conversation(conversation_history)
    if not conversation_text.strip():
        return ExtractedIntelligence()

    try:
        response = await _OPENAI_CLIENT.beta.chat.completions.parse(
            model=OPENAI_EXTRACTION_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract and normalize all financial identifiers from the conversation. "
                        "Convert spoken numbers to digits. Resolve 'at' to '@' and 'dot' to '.'. "
                        "Return only the structured extraction result."
                    ),
                },
                {
                    "role": "user",
                    "content": conversation_text,
                },
            ],
            response_format=ExtractedIntelligence,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            logger.warning("OpenAI extraction returned no parsed payload.")
            return ExtractedIntelligence()
        return parsed
    except Exception as e:
        logger.error(f"OpenAI extraction failed: {e}", exc_info=True)
        return ExtractedIntelligence()


def extract_bank_accounts(text: str) -> list[str]:
    pattern = r"\b\d{9,18}\b"
    return list(set(re.findall(pattern, text)))[:5]


def extract_upi_ids(text: str) -> list[str]:
    pattern_std = r"\b[\w\.-]+@[\w\.-]+\b"
    pattern_text = r"\b[\w\.-]+\s+(?:at|@)\s+[\w\.-]+\s+(?:dot|\.)\s+(?:com|in)\b"

    found_std = re.findall(pattern_std, text)
    found_text = re.findall(pattern_text, text, re.IGNORECASE)

    normalized_text = []
    for item in found_text:
        normalized_text.append(
            item.lower().replace(" at ", "@").replace(" dot ", ".").replace(" ", "")
        )

    all_upis = found_std + normalized_text
    return list(set(u for u in all_upis if "@" in u))[:5]


def extract_links(text: str) -> list[str]:
    pattern = r"(?:https?://)?(?:www\.)?(?:bit\.ly|tinyurl\.com|goo\.gl|[a-zA-Z0-9-]+\.[a-zA-Z]{2,})/[^\s]*"
    return list(set(re.findall(pattern, text)))[:5]


def extract_phone_numbers(text: str) -> list[str]:
    patterns = [
        r"\+91[\s-]?\d{10}",
        r"\b\d{10}\b",
        r"\b\d{5}[\s-]\d{5}\b",
    ]
    phones: list[str] = []
    for pattern in patterns:
        phones.extend(re.findall(pattern, text))
    return list(set(phones))[:5]


def extract_emails(text: str) -> list[str]:
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return list(set(re.findall(pattern, text)))[:5]


def extract_apk_links(text: str) -> list[str]:
    pattern = r"https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.apk"
    return list(set(re.findall(pattern, text, re.IGNORECASE)))[:5]


def extract_crypto_wallets(text: str) -> list[str]:
    patterns = [
        r"\b(0x[a-fA-F0-9]{40})\b",
        r"\b(T[A-Za-z1-9]{33})\b",
        r"\b(1[a-km-zA-HJ-NP-Z1-9]{25,34})\b",
        r"\b(bc1[a-zA-HJ-NP-Z0-9]{39,59})\b",
    ]
    wallets: list[str] = []
    for pattern in patterns:
        wallets.extend(re.findall(pattern, text))
    return list(set(wallets))[:5]


def extract_social_handles(text: str) -> list[str]:
    pattern = r"(?<![\w.-])@([a-zA-Z0-9_]{3,25})\b"
    handles = re.findall(pattern, text)
    return [f"@{handle}" for handle in list(set(handles))][:5]


def extract_ifsc_codes(text: str) -> list[str]:
    pattern = r"\b[A-Z]{4}0[A-Z0-9]{6}\b"
    return list(set(re.findall(pattern, text)))[:3]


def extract_aadhaar_numbers(text: str) -> list[str]:
    pattern = r"\b\d{4}\s?\d{4}\s?\d{4}\b"
    return list(set(re.findall(pattern, text)))[:3]


def extract_amounts(text: str) -> list[str]:
    pattern = r"(?:rs\.?|inr)?\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:lakh|crore|k))?"
    return list(set(re.findall(pattern, text, re.IGNORECASE)))[:5]


def extract_keywords(text: str) -> list[str]:
    suspicious_keywords = [
        "urgent",
        "immediately",
        "blocked",
        "suspend",
        "verify",
        "otp",
        "upi",
        "bank account",
        "account",
        "kyc",
        "refund",
        "winner",
        "prize",
        "lottery",
        "congratulations",
        "click here",
        "link",
        "expire",
        "confirm",
        "apk",
        "download",
        "install",
        "cbi",
        "police",
        "arrest",
    ]
    text_lower = text.lower()
    found = [keyword for keyword in suspicious_keywords if keyword in text_lower]
    return list(set(found))[:10]


def extract_regex_intelligence(conversation_history: list) -> dict:
    all_text = _join_conversation(conversation_history)
    normalized = normalize_before_extract(all_text)

    return {
        "bankAccounts": _merge_values(extract_bank_accounts(all_text), extract_bank_accounts(normalized)),
        "upiIds": _merge_values(extract_upi_ids(all_text), extract_upi_ids(normalized)),
        "phishingLinks": _merge_values(extract_links(all_text), extract_links(normalized)),
        "phoneNumbers": _merge_values(extract_phone_numbers(all_text), extract_phone_numbers(normalized)),
        "emails": _merge_values(extract_emails(all_text), extract_emails(normalized)),
        "apkLinks": _merge_values(extract_apk_links(all_text), extract_apk_links(normalized)),
        "cryptoWallets": _merge_values(extract_crypto_wallets(all_text), extract_crypto_wallets(normalized)),
        "socialHandles": _merge_values(extract_social_handles(all_text), extract_social_handles(normalized)),
        "ifscCodes": _merge_values(extract_ifsc_codes(all_text), extract_ifsc_codes(normalized)),
        "aadhaarNumbers": _merge_values(extract_aadhaar_numbers(all_text), extract_aadhaar_numbers(normalized)),
        "amounts": _merge_values(extract_amounts(all_text), extract_amounts(normalized)),
        "suspiciousKeywords": extract_keywords(all_text),
    }


async def extract_intelligence(conversation_history: list) -> dict:
    regex_intelligence = extract_regex_intelligence(conversation_history)
    openai_intelligence = await extract_with_openai(conversation_history)

    merged = {
        "bankAccounts": _merge_values(
            regex_intelligence["bankAccounts"],
            openai_intelligence.bank_accounts,
        ),
        "upiIds": _merge_values(
            regex_intelligence["upiIds"],
            openai_intelligence.upi_ids,
        ),
        "phishingLinks": _merge_values(
            regex_intelligence["phishingLinks"],
            openai_intelligence.phishing_links,
        ),
        "phoneNumbers": _merge_values(
            regex_intelligence["phoneNumbers"],
            openai_intelligence.phone_numbers,
        ),
        "emails": regex_intelligence["emails"],
        "apkLinks": regex_intelligence["apkLinks"],
        "cryptoWallets": _merge_values(
            regex_intelligence["cryptoWallets"],
            openai_intelligence.crypto_wallets,
        ),
        "socialHandles": regex_intelligence["socialHandles"],
        "ifscCodes": regex_intelligence["ifscCodes"],
        "aadhaarNumbers": _merge_values(
            regex_intelligence["aadhaarNumbers"],
            openai_intelligence.aadhaar_numbers,
        ),
        "amounts": _merge_values(
            regex_intelligence["amounts"],
            openai_intelligence.amounts,
        ),
        "suspiciousKeywords": regex_intelligence["suspiciousKeywords"],
    }

    logger.info("Extraction results prepared: %s", merged)
    return merged
