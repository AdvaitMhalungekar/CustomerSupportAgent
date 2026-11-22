import re
from typing import Tuple, Optional

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+prompts",
    r"system\s*prompt",
    r"you\s+are\s+now\s+(an?\s+)?unrestricted",
    r"jailbreak",
    r"developer\s+mode",
    r"bypass\s+all\s+rules",
    r"do\s+anything\s+now",
    r"reveal\s+your\s+hidden\s+instructions",
    r"override\s+(the\s+)?safety",
]

PII_PATTERNS = [
    (r"\b(?:\d[ -]*?){13,16}\b", "[REDACTED_CARD]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
    (r"(?i)\bpassword\s*[:=]\s*\S+", "[REDACTED_PWD]"),
]

def check_input_safety(text: str) -> Tuple[bool, Optional[str], str]:
    """Check user transcript for prompt injection and sanitize sensitive PII.
    Returns: (is_safe, violation_reason, sanitized_text)
    """
    if not text or not text.strip():
        return True, None, text

    # Check prompt injection
    lower_text = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower_text):
            return False, f"Prompt injection detected matching pattern: {pattern}", text

    # Redact sensitive PII if present in transcript
    sanitized = text
    for pattern, replacement in PII_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized)

    return True, None, sanitized
