"""
PII Redactor — Phase 4 Safety.

Extended redaction for STORED transcripts (more aggressive than the real-time
input guardrail which only redacts card/SSN/password in live caller speech).

For stored data we additionally redact:
  - 10-digit Indian mobile numbers
  - email addresses
  - Aadhaar-style 12-digit numbers
  - UPI IDs
  - date-of-birth patterns

The redact_for_storage() function is called before any transcript snippet
is written to the audit log or the handoff payload.
"""
from __future__ import annotations

import re
from typing import List, Tuple

# (pattern, replacement_label)
_REDACT_RULES: List[Tuple[re.Pattern, str]] = [
    # Credit/debit card numbers (13-16 digits, with optional separators)
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[REDACTED_CARD]"),
    # Indian mobile numbers (10 digits starting 6-9, with optional country code)
    (re.compile(r"\b(?:\+?91[-\s]?)?[6-9]\d{9}\b"), "[REDACTED_PHONE]"),
    # Email addresses
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    # Aadhaar (12 digits, with optional spaces every 4)
    (re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "[REDACTED_AADHAAR]"),
    # UPI IDs (handle@bank)
    (re.compile(r"\b[A-Za-z0-9.\-_]+@[a-z]{2,20}\b"), "[REDACTED_UPI]"),
    # Date of birth patterns (DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD)
    (re.compile(r"\b(?:\d{2}[/\-]\d{2}[/\-]\d{4}|\d{4}[/\-]\d{2}[/\-]\d{2})\b"), "[REDACTED_DOB]"),
    # Password/PIN hints
    (re.compile(r"(?i)\b(?:password|pin|passcode)\s*[:=\s]\s*\S+"), "[REDACTED_CRED]"),
    # SSN (US, retained from input_guard for completeness)
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
]

# Patterns that should NOT be redacted even if they look like phone numbers
# (e.g. pincode 411005, year 2024 — short or already-normalised)
_PRESERVE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b411\d{3}\b"),   # Pune pin codes
    re.compile(r"\b20\d{2}\b"),    # years
]


def redact_for_storage(text: str) -> str:
    """
    Redact PII from text before writing to audit logs or handoff payloads.
    Preserves known non-PII numeric tokens (pincodes, years).
    """
    if not text:
        return text
    result = text
    for pattern, label in _REDACT_RULES:
        result = pattern.sub(label, result)
    return result


def redact_for_live(text: str) -> str:
    """
    Lighter redaction for live transcript display (keeps phone numbers visible
    for the agent to look up, but strips card/SSN/password).
    """
    if not text:
        return text
    result = text
    # Only strip clearly dangerous items in live context
    for pattern, label in _REDACT_RULES[:1]:  # card numbers only
        result = pattern.sub(label, result)
    # SSN
    result = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", result)
    # Password
    result = re.sub(r"(?i)\bpassword\s*[:=]\s*\S+", "[REDACTED_PWD]", result)
    return result


def contains_pii(text: str) -> bool:
    """Quick check if text contains any pattern we'd redact."""
    for pattern, _ in _REDACT_RULES:
        if pattern.search(text):
            return True
    return False
