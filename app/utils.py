"""
Shared logging, callback, and workflow utility helpers.

This module restores the helper surface that the existing LangGraph workflow
expects without changing the broader architecture.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Mapping
from typing import Any

import httpx

from app.config import API_KEY, CALLBACKS_ENABLED
from app.models import Callback, ExtractedIntelligence


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
CALLBACK_URL = os.getenv("_CALLBACK_URL", "").strip()
CALLBACK_TIMEOUT_SECONDS = float(os.getenv("CALLBACK_TIMEOUT_SECONDS", "10"))

ACTIONABLE_INTELLIGENCE_KEYS = (
    "bankAccounts",
    "upiIds",
    "phishingLinks",
    "phoneNumbers",
    "emails",
    "apkLinks",
    "cryptoWallets",
    "socialHandles",
    "ifscCodes",
)


def _build_logger() -> logging.Logger:
    logger_instance = logging.getLogger("kaizen")
    if not logger_instance.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger_instance.addHandler(handler)

    logger_instance.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger_instance.propagate = False
    return logger_instance


logger = _build_logger()


def get_session_logger(session_id: str) -> logging.Logger:
    """
    Return a child logger scoped to a single session id.
    """
    safe_session_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", session_id or "unknown")
    return logger.getChild(f"session.{safe_session_id}")


class PerformanceLogger:
    """
    Lightweight timing context manager used across the workflow nodes.
    """

    def __init__(self, label: str, active_logger: logging.Logger | None = None):
        self.label = label
        self.active_logger = active_logger or logger
        self.started_at: float | None = None

    def __enter__(self) -> "PerformanceLogger":
        self.started_at = time.perf_counter()
        self.active_logger.debug("[PERF] %s started", self.label)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed_ms = 0.0
        if self.started_at is not None:
            elapsed_ms = (time.perf_counter() - self.started_at) * 1000

        if exc is None:
            self.active_logger.info("[PERF] %s completed in %.1f ms", self.label, elapsed_ms)
        else:
            self.active_logger.error(
                "[PERF] %s failed after %.1f ms: %s",
                self.label,
                elapsed_ms,
                exc,
                exc_info=True,
            )

        return False


def _coerce_intelligence(intelligence: Any) -> dict[str, list[str]]:
    if not isinstance(intelligence, Mapping):
        return {}

    normalized: dict[str, list[str]] = {}
    for key, value in intelligence.items():
        if isinstance(value, list):
            normalized[str(key)] = [str(item) for item in value if item is not None]
        elif value:
            normalized[str(key)] = [str(value)]
        else:
            normalized[str(key)] = []
    return normalized


def _count_intelligence_items(intelligence: Mapping[str, Any], keys: tuple[str, ...]) -> int:
    return sum(
        len(value)
        for key, value in intelligence.items()
        if key in keys and isinstance(value, list)
    )


def log_intelligence(session_id: str, intelligence: Any) -> None:
    """
    Log a concise summary of extracted intelligence for observability.
    """
    normalized = _coerce_intelligence(intelligence)
    actionable_count = _count_intelligence_items(normalized, ACTIONABLE_INTELLIGENCE_KEYS)
    suspicious_count = len(normalized.get("suspiciousKeywords", []))

    if actionable_count == 0 and suspicious_count == 0:
        logger.info("[INTEL] Session %s: no intelligence extracted yet", session_id)
        return

    categories = [
        f"{key}={len(value)}"
        for key, value in normalized.items()
        if isinstance(value, list) and value
    ]
    logger.info(
        "[INTEL] Session %s: actionable=%s suspicious=%s (%s)",
        session_id,
        actionable_count,
        suspicious_count,
        ", ".join(categories),
    )


def should_send_callback(state: Mapping[str, Any]) -> bool:
    """
    Decide when a scam conversation has enough signal to finalize.

    The workflow keeps engaging until one of the following is true:
    - we extracted actionable infrastructure from the scammer
    - we gathered enough suspicious evidence after several turns
    - we hit a turn cap and should stop wasting resources
    """
    if not state.get("scamDetected", False):
        return False

    intelligence = _coerce_intelligence(state.get("extractedIntelligence", {}))
    actionable_count = _count_intelligence_items(intelligence, ACTIONABLE_INTELLIGENCE_KEYS)
    suspicious_count = len(intelligence.get("suspiciousKeywords", []))
    total_messages = int(state.get("totalMessages") or 0)

    if actionable_count >= 1 and total_messages >= 3:
        return True

    if actionable_count >= 2:
        return True

    if suspicious_count >= 2 and total_messages >= 6:
        return True

    if total_messages >= 10:
        return True

    return False


def send_final_callback(session_id: str, state: Mapping[str, Any]) -> bool:
    """
    Send the final callback payload if callbacks are enabled and configured.
    """
    if not CALLBACKS_ENABLED:
        logger.info("[CALLBACK] Disabled. Skipping final callback for session %s", session_id)
        return False

    if not CALLBACK_URL:
        logger.warning("[CALLBACK] _CALLBACK_URL missing. Skipping final callback for session %s", session_id)
        return False

    intelligence = _coerce_intelligence(state.get("extractedIntelligence", {}))

    payload = Callback(
        sessionId=session_id,
        scamDetected=bool(state.get("scamDetected", False)),
        totalMessagesExchanged=int(state.get("totalMessages") or 0),
        extractedIntelligence=ExtractedIntelligence(**intelligence),
        agentNotes=str(
            state.get("fullSummaryForCallback")
            or state.get("agentNotes")
            or ""
        ),
    ).model_dump(mode="json")

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY

    try:
        response = httpx.post(
            CALLBACK_URL,
            json=payload,
            headers=headers,
            timeout=CALLBACK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("[CALLBACK] Failed for session %s: %s", session_id, exc, exc_info=True)
        return False

    logger.info("[CALLBACK] Final callback sent for session %s", session_id)
    return True
