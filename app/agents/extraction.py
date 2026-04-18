

import re

def normalize_before_extract(text: str) -> str:
    """Pre-process obfuscated intel before regex runs (Strategy 1: Silent Intel)"""
    
    # 1. "at" → "@", "dot" → "."
    text = re.sub(r'\s+at\s+', '@', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+dot\s+', '.', text, flags=re.IGNORECASE)
    
    # 2. Spaced characters: "9 8 7 6" → "9876"
    # Collapses single digits separated by spaces
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    
    # 3. Word numbers (partial)
    word_map = {
        "zero":"0","one":"1","two":"2","three":"3","four":"4",
        "five":"5","six":"6","seven":"7","eight":"8","nine":"9"
    }
    for word, digit in word_map.items():
        text = re.sub(r'\b' + word + r'\b', digit, text, flags=re.IGNORECASE)
    
    return text

def extract_intelligence(conversation_history: list) -> dict:
   
    # Combine all message texts into one string for easier searching
    all_text = " ".join([
        msg.get("text", "") 
        for msg in conversation_history
    ])
    
    # Run on BOTH original and normalized — merge results
    normalized = normalize_before_extract(all_text)
    
    intelligence = {
        "bankAccounts":  list(set(extract_bank_accounts(all_text) + extract_bank_accounts(normalized))),
        "upiIds":        list(set(extract_upi_ids(all_text)       + extract_upi_ids(normalized))),
        "phishingLinks": list(set(extract_links(all_text)         + extract_links(normalized))),
        "phoneNumbers":  list(set(extract_phone_numbers(all_text) + extract_phone_numbers(normalized))),
        "emails":        list(set(extract_emails(all_text)        + extract_emails(normalized))),
        "apkLinks":      list(set(extract_apk_links(all_text)     + extract_apk_links(normalized))),
        "cryptoWallets": list(set(extract_crypto_wallets(all_text))),
        "socialHandles": list(set(extract_social_handles(all_text))),
        "ifscCodes":     list(set(extract_ifsc_codes(all_text))),
        "suspiciousKeywords": extract_keywords(all_text)
    }
    
    print(f" Extraction Results:")
    for key, value in intelligence.items():
        if value:
            print(f"   {key}: {value}")
    
    return intelligence


def extract_bank_accounts(text: str) -> list:
    pattern = r'\b\d{9,18}\b'
    accounts = re.findall(pattern, text)
    return list(set(accounts))[:5]


def extract_upi_ids(text: str) -> list:
    pattern_std = r'\b[\w\.-]+@[\w\.-]+\b'
    pattern_text = r'\b[\w\.-]+\s+(?:at|@)\s+[\w\.-]+\s+(?:dot|\.)\s+(?:com|in)\b'
    
    found_std = re.findall(pattern_std, text)
    found_text = re.findall(pattern_text, text, re.IGNORECASE)
    
    normalized_text = []
    for t in found_text:
        t = t.lower()
        t = t.replace(" at ", "@").replace(" dot ", ".").replace(" ", "")
        normalized_text.append(t)
    
    all_upis = found_std + normalized_text
    upis = [u for u in all_upis if '@' in u]
    return list(set(upis))[:5]


def extract_links(text: str) -> list:
    pattern = r'(?:https?://)?(?:www\.)?(?:bit\.ly|tinyurl\.com|goo\.gl|[a-zA-Z0-9-]+\.[a-zA-Z]{2,})/[^\s]*'
    links = re.findall(pattern, text)
    return list(set(links))[:5]


def extract_phone_numbers(text: str) -> list:
    patterns = [
        r'\+91[\s-]?\d{10}',       # +91-1234567890
        r'\b\d{10}\b',              # 9876543210
        r'\b\d{5}[\s-]\d{5}\b'     # 12345-67890
    ]
    phones = []
    for pattern in patterns:
        found = re.findall(pattern, text)
        phones.extend(found)
    return list(set(phones))[:5]


def extract_emails(text: str) -> list:
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(pattern, text)
    return list(set(emails))[:5]


def extract_apk_links(text: str) -> list:
    pattern = r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.apk'
    links = re.findall(pattern, text, re.IGNORECASE)
    return list(set(links))[:5]


def extract_crypto_wallets(text: str) -> list:
    patterns = [
        r'\b(0x[a-fA-F0-9]{40})\b',  # Ethereum/BSC/Polygon
        r'\b(T[A-Za-z1-9]{33})\b',   # TRON
        r'\b(1[a-km-zA-HJ-NP-Z1-9]{25,34})\b', # Bitcoin (Legacy)
        r'\b(bc1[a-zA-HJ-NP-Z0-9]{39,59})\b'   # Bitcoin (Bech32)
    ]
    wallets = []
    for p in patterns:
        wallets.extend(re.findall(p, text))
    return list(set(wallets))[:5]


def extract_social_handles(text: str) -> list:
    # Matches @username, commonly for Telegram/Twitter
    # Avoids email parts by ensuring whitespace before @
    pattern = r'(?<![\w.-])@([a-zA-Z0-9_]{3,25})\b'
    handles = re.findall(pattern, text)
    return [f"@{h}" for h in list(set(handles))][:5]


def extract_ifsc_codes(text: str) -> list:
    pattern = r'\b[A-Z]{4}0[A-Z0-9]{6}\b'
    codes = re.findall(pattern, text)
    return list(set(codes))[:3]


def extract_keywords(text: str) -> list:
    suspicious_keywords = [
        'urgent', 'immediately', 'blocked', 'suspend', 'verify',
        'otp', 'upi', 'bank account', 'account', 'kyc', 'refund',
        'winner', 'prize', 'lottery', 'congratulations',
        'click here', 'link', 'expire', 'confirm',
        'apk', 'download', 'install', 'cbi', 'police', 'arrest'
    ]
    text_lower = text.lower()
    found = []
    for keyword in suspicious_keywords:
        if keyword in text_lower:
            found.append(keyword)
    return list(set(found))[:10]