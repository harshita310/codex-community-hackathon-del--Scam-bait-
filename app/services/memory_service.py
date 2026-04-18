import json
import os

import numpy as np
from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
from app.utils import logger


KNOWN_SCAM_PATTERNS = [
    {"name": "digital_arrest", "text": "Your Aadhaar was linked to a crime and you must transfer funds for verification."},
    {"name": "kyc_update", "text": "Complete your KYC immediately or your bank account will be blocked today."},
    {"name": "upi_refund", "text": "Share your UPI PIN to receive a refund that failed to credit."},
    {"name": "job_task", "text": "Earn daily income from home by completing Telegram tasks and paying a joining fee."},
    {"name": "lottery_win", "text": "You won a lucky draw prize and must pay a release fee to claim it."},
    {"name": "fake_support", "text": "Install this APK or remote support app so we can fix your banking issue."},
    {"name": "sim_block", "text": "Your SIM will be blocked unless you verify the OTP and full name immediately."},
    {"name": "parcel_customs", "text": "A parcel with illegal items was caught and customs requires urgent payment."},
    {"name": "sextortion", "text": "We recorded your private video and will leak it unless you send money today."},
    {"name": "impersonation", "text": "This is the bank investigation team, transfer money to the safe account now."},
]


_OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_known_scam_embeddings.json")


def embed_text(text: str) -> list:
    if _OPENAI_CLIENT is None:
        raise RuntimeError("OPENAI_API_KEY is required for embedding generation.")

    response = _OPENAI_CLIENT.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(a: list, b: list) -> float:
    if not a or not b:
        return 0.0

    a_vector = np.array(a)
    b_vector = np.array(b)
    denominator = float(np.linalg.norm(a_vector) * np.linalg.norm(b_vector))
    if denominator == 0:
        return 0.0
    return float(np.dot(a_vector, b_vector) / denominator)


def _load_cached_embeddings() -> dict[str, list] | None:
    if not os.path.exists(_CACHE_PATH):
        return None

    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as cache_file:
            cached = json.load(cache_file)
        if set(cached.keys()) == {pattern["name"] for pattern in KNOWN_SCAM_PATTERNS}:
            return cached
    except Exception as e:
        logger.warning(f"Unable to read embedding cache: {e}")
    return None


def _write_cached_embeddings(embeddings: dict[str, list]) -> None:
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as cache_file:
            json.dump(embeddings, cache_file)
    except Exception as e:
        logger.warning(f"Unable to write embedding cache: {e}")


def _warm_known_pattern_embeddings() -> dict[str, list]:
    cached = _load_cached_embeddings()
    if cached is not None:
        return cached

    if _OPENAI_CLIENT is None:
        logger.warning("Embedding warm-up skipped because OPENAI_API_KEY is missing.")
        return {}

    embeddings: dict[str, list] = {}
    for pattern in KNOWN_SCAM_PATTERNS:
        try:
            embeddings[pattern["name"]] = embed_text(pattern["text"])
        except Exception as e:
            logger.error(f"Embedding warm-up failed for {pattern['name']}: {e}", exc_info=True)

    if embeddings:
        _write_cached_embeddings(embeddings)
    return embeddings


KNOWN_SCAM_PATTERN_EMBEDDINGS = _warm_known_pattern_embeddings()


def find_closest_scam_pattern(message: str) -> tuple[str, float]:
    if not KNOWN_SCAM_PATTERN_EMBEDDINGS:
        return "", 0.0

    try:
        message_embedding = embed_text(message)
    except Exception as e:
        logger.error(f"Failed to embed message for scam memory: {e}", exc_info=True)
        return "", 0.0

    best_pattern = ""
    best_score = 0.0
    for pattern in KNOWN_SCAM_PATTERNS:
        pattern_name = pattern["name"]
        score = cosine_similarity(message_embedding, KNOWN_SCAM_PATTERN_EMBEDDINGS.get(pattern_name, []))
        if score > best_score:
            best_pattern = pattern_name
            best_score = score

    return best_pattern, best_score


def is_semantically_similar_to_scam(message: str, threshold: float = 0.75) -> bool:
    pattern_name, similarity_score = find_closest_scam_pattern(message)
    if pattern_name:
        logger.info("Semantic scam memory match: %s (%.3f)", pattern_name, similarity_score)
    return similarity_score > threshold
