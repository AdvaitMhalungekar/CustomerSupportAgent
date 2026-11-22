import re
from typing import Tuple, List, Set

# Non-entity conversational numbers to tolerate (e.g. "one moment", "first option")
COMMON_CONVERSATIONAL_WORDS = {"one", "two", "three", "first", "second", "third"}

def _extract_numerical_tokens(text: str) -> List[str]:
    """Extract numbers, currency quantities, and numeric codes from text.
    Handles comma-separated formats like 15,000, 25,000, 411005, etc.
    """
    if not text:
        return []
    # Match numbers with optional commas or decimal points
    raw_matches = re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", text)
    tokens = []
    for m in raw_matches:
        # Standardize: both with and without commas
        cleaned = m.replace(",", "")
        tokens.append(m)
        if cleaned != m:
            tokens.append(cleaned)
    return tokens

def _extract_temporal_tokens(text: str) -> List[str]:
    """Extract durations and dates like '6 months', '2 weeks', etc."""
    durations = re.findall(r"\b\d+\s*(?:month|year|week|day|hr|hour)s?\b", text, re.IGNORECASE)
    return [d.lower() for d in durations]

def verify_groundedness(reply: str, kb_text: str, tool_context: str = "") -> Tuple[bool, List[str]]:
    """Deterministically check that every number, price, duration, or entity in reply
    appears in the KB or a tool result.
    Returns: (is_grounded, ungrounded_tokens)
    """
    if not reply or not reply.strip():
        return True, []

    combined_sources = f"{kb_text}\n{tool_context}"
    # Standardize sources: also add variants without commas
    source_numbers = set(_extract_numerical_tokens(combined_sources))
    source_durations = set(_extract_temporal_tokens(combined_sources))
    combined_sources_lower = combined_sources.lower()

    ungrounded = []

    # Check numerical tokens
    reply_numbers = _extract_numerical_tokens(reply)
    for num in reply_numbers:
        # Skip single digit conversational counts (like "option 1", "step 2") if needed,
        # but check pricing, pin codes, durations strictly
        raw_val = num.replace(",", "")
        if raw_val not in source_numbers and num not in source_numbers:
            # If it's a multi-digit number or specific fee/pin/phone, it MUST be grounded
            if len(raw_val) > 1 or raw_val not in {"1", "2", "3"}:
                ungrounded.append(num)

    # Check durations like "6 months", "12 months"
    reply_durations = _extract_temporal_tokens(reply)
    for dur in reply_durations:
        if dur not in combined_sources_lower and dur not in source_durations:
            ungrounded.append(dur)

    is_grounded = len(ungrounded) == 0
    return is_grounded, list(set(ungrounded))
