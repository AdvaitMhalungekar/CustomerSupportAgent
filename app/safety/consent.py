"""
Consent manager — Phase 4 Safety.

The agent's FIRST turn is always a consent notice. The caller must reply with
an explicit acknowledgement ("yes", "okay", "sure", etc.) before any personal
data (phone number, enquiry ID) is processed or stored.

Design:
  - ConsentState is kept per session (in-process dict; can be Redis for prod).
  - If consent is not yet given, every inbound transcript is checked.
  - On a positive ACK, consent is marked and the timestamp is logged.
  - On a negative or ambiguous response, the agent re-prompts once, then
    falls back to anonymous info-only mode (no tool calls that touch PII).
  - Offline / test mode: consent is auto-granted so tests run without user input.
"""
from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

CONSENT_NOTICE = (
    "Hello! You've reached AI Adventures, Pune's hands-on AI and machine learning "
    "training institute. This call may be monitored and recorded for quality and "
    "training purposes. By continuing, you consent to this recording. "
    "Do you agree to proceed?"
)

CONSENT_NAK_PATTERNS = [
    r"\bno\b(?!\s+(?:problem|worries|issue|issues|objection|objections))\b",
    r"\bnope\b",
    r"\bno thank(?:s)?\b",
    r"\brefuse\b",
    r"\b(?:don\'?t|do not)\s+(?:want|agree|consent)\b",
    r"\bdisagree\b",
    r"\bcancel\b",
]

CONSENT_ACK_PATTERNS = [
    r"\byes\b", r"\byeah\b", r"\bsure\b", r"\bokay\b", r"\bok\b",
    r"\bagree\b", r"\baccept\b", r"\bfine\b", r"\bgo ahead\b", r"\bproceed\b",
    r"\bof course\b", r"\babsolutely\b", r"\bcorrect\b", r"\bright\b",
    r"\bno problem\b", r"\bno worries\b",
]

_NAK_RE = [re.compile(p, re.IGNORECASE) for p in CONSENT_NAK_PATTERNS]
_ACK_RE = [re.compile(p, re.IGNORECASE) for p in CONSENT_ACK_PATTERNS]


@dataclass
class ConsentRecord:
    session_id: str
    given: bool = False
    refused: bool = False
    timestamp: float = 0.0
    transcript_snippet: str = ""
    attempts: int = 0


class ConsentManager:
    """
    Manages per-session consent state.
    Thread-safe for a single-process FastAPI server (asyncio, no threads crossing sessions).
    """

    def __init__(self, auto_grant_offline: bool = False):
        """
        auto_grant_offline: set True in test/offline mode so unit tests
        don't need to simulate a consent exchange.
        """
        self._sessions: Dict[str, ConsentRecord] = {}
        self._auto_grant = auto_grant_offline

    def get_or_create(self, session_id: str) -> ConsentRecord:
        if session_id not in self._sessions:
            rec = ConsentRecord(session_id=session_id)
            if self._auto_grant:
                rec.given = True
                rec.timestamp = time.time()
                rec.transcript_snippet = "[AUTO-GRANTED: offline mode]"
            self._sessions[session_id] = rec
        return self._sessions[session_id]

    def is_consent_given(self, session_id: str) -> bool:
        return self.get_or_create(session_id).given

    def process_consent_reply(self, session_id: str, text: str) -> str:
        """
        Parse a caller's reply to the consent notice.
        Returns one of: "granted" | "refused" | "re_prompt" | "already_given"
        """
        rec = self.get_or_create(session_id)
        if rec.given:
            return "already_given"

        rec.attempts += 1
        lower = text.lower().strip()

        # Check explicit refusal first (with exceptions like "no problem")
        for pat in _NAK_RE:
            if pat.search(lower):
                rec.refused = True
                rec.timestamp = time.time()
                rec.transcript_snippet = text[:120]
                logger.info(f"[{session_id}] Consent REFUSED: '{text[:60]}'")
                return "refused"

        # Check positive ACK
        for pat in _ACK_RE:
            if pat.search(lower):
                rec.given = True
                rec.timestamp = time.time()
                rec.transcript_snippet = text[:120]
                logger.info(f"[{session_id}] Consent GRANTED: '{text[:60]}'")
                return "granted"

        # Ambiguous — re-prompt once; after 2 attempts, default to info-only
        if rec.attempts >= 2:
            logger.warning(f"[{session_id}] Consent ambiguous after {rec.attempts} attempts. Info-only mode.")
            rec.given = False  # stays ungiven; PII tools locked out
            return "re_prompt"

        return "re_prompt"

    def consent_notice(self) -> str:
        return CONSENT_NOTICE

    def re_prompt_text(self) -> str:
        return (
            "I just need a quick yes or no — do you agree to this call being recorded "
            "for quality purposes?"
        )

    def refused_text(self) -> str:
        return (
            "No problem at all. I won't record this call. You can still ask me about "
            "our courses and fees. If you'd like to speak directly to our team, "
            "I can arrange that for you."
        )


# Global instance — auto_grant in offline/test mode
import os
_offline = not bool(os.environ.get("GROQ_API_KEY", ""))
consent_manager = ConsentManager(auto_grant_offline=_offline)
