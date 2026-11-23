"""
Input Guardrail — Phase 4 upgraded.

Responsibilities:
1. Prompt-injection detection (hardened pattern set)
2. PII redaction for LIVE transcript (lighter — preserves phone numbers for tool lookups)
3. PII redaction for STORED transcript (full — called before audit log writes)
4. Flags the safety_blocked field on the state so speak_gate can route correctly.
"""
import re
import time
import logging
from typing import Tuple, Optional

from app.graph.state import VoiceAgentState
from app.safety.pii import redact_for_storage, redact_for_live

logger = logging.getLogger(__name__)

# ── Injection Patterns (hardened for voice) ──────────────────────────────────
# Voice ASR may distort punctuation and casing — use loose word-boundary patterns.

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?|context)",
    r"disregard\s+(all\s+)?(previous|prior)\s+prompts?",
    r"system\s*prompt",
    r"you\s+are\s+now\s+(an?\s+)?unrestricted",
    r"jailbreak",
    r"developer\s+mode",
    r"bypass\s+.*?\b(rules?|guardrails?|safety|filters?|restrictions?)\b",
    r"do\s+anything\s+now",
    r"reveal\s+your\s+(hidden\s+)?instructions?",
    r"override\s+(the\s+)?safety",
    r"act\s+as\s+(if\s+you\s+are\s+)?(an?\s+)?.*?\b(ai|model|assistant|bot|agent)\b",
    r"pretend\s+(you\s+are|to\s+be)\s+(a\s+)?(different|unrestricted|evil|dan|jailbroken)",
    r"forget\s+(all\s+)?(your|previous|the)\s+(rules?|instructions?|training)",
    r"new\s+persona\s*[:=]",
    r"you\s+have\s+no\s+(restrictions?|limits?|rules?|guidelines?)",
    r"disable\s+(all\s+)?(safety|filter|guardrail)",
    r"switch\s+to\s+.*?\bmode\b",
    r"\b(admin|root)\s+mode\b",
    r"\b(admin|root)\s+access\b",
    r"\bunrestricted\b",
    # Social engineering variants that target refund/fee autonomy
    r"your\s+(manager|ceo|owner|boss)\s+(said|told|approved|authorized)",
    r"i\s+have\s+written\s+permission\s+to\s+bypass",
    r"special\s+(code|password|override)\s*[:=]\s*\S+",
]

_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def check_input_safety(text: str) -> Tuple[bool, Optional[str], str]:
    """
    Check user transcript for prompt injection and sanitize PII for live use.
    Returns: (is_safe, violation_reason, live_sanitized_text)
    """
    if not text or not text.strip():
        return True, None, text

    lower_text = text.lower()
    for pat in _COMPILED_INJECTION:
        if pat.search(lower_text):
            reason = f"Prompt injection detected: '{pat.pattern[:60]}'"
            logger.warning(f"[INPUT GUARD] Injection blocked: '{text[:80]}'")
            # Return original text for audit logging but flag as unsafe
            return False, reason, text

    # Light PII redaction for live context (preserves phone numbers for tools)
    sanitized = redact_for_live(text)
    return True, None, sanitized


def n_input_guardrail(state: VoiceAgentState) -> VoiceAgentState:
    start_t = time.perf_counter()
    transcript = state.get("transcript", "")
    session_id = state.get("session_id", "unknown")
    latencies = dict(state.get("latency_ms", {}))

    is_safe, violation, sanitized = check_input_safety(transcript)

    # Storage-level PII redaction for audit log
    redacted_for_storage = redact_for_storage(transcript)

    elapsed = (time.perf_counter() - start_t) * 1000
    latencies["input_guardrail"] = elapsed

    if not is_safe:
        logger.warning(f"[{session_id}] Input guardrail blocked: {violation}")
        return {
            "transcript": sanitized,
            "pii_redacted_transcript": redacted_for_storage,
            "safety_blocked": True,
            "safety_violation": violation,
            "handoff": True,
            "handoff_reason": f"Input safety triggered: {violation}",
            "handoff_category": "injection",
            "reply": "I am unable to process that instruction. Let me transfer you to our admissions desk.",
            "latency_ms": latencies,
        }

    return {
        "transcript": sanitized,
        "pii_redacted_transcript": redacted_for_storage,
        "safety_blocked": False,
        "safety_violation": None,
        "latency_ms": latencies,
    }
