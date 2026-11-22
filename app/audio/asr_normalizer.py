"""ASR normalization for Indian-accented English and Whisper homophones.
Handles:
  - Digit sequences spoken as words ("nine eight..." -> "98...")
  - Course name mishearing ("pie-thon" -> "Python", "mashing learning" -> "Machine Learning")
  - Number formats ("fifteen thousand" -> "15000")
  - Phone number concatenation ("nine eight seven six" -> "9876")
"""
import re
from typing import Optional

WORD_TO_DIGIT = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9",
}

NUMBER_WORDS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000, "lakh": 100000,
}

COURSE_NORMALIZATIONS = [
    (r"\bpie[\s-]?thon\b", "Python"),
    (r"\bpy[\s-]?thon\b", "Python"),
    (r"\bpythan\b", "Python"),
    (r"\bpython programming\b", "Python"),
    (r"\bmachine[\s-]?learning\b", "Machine Learning"),
    (r"\bmashing learning\b", "Machine Learning"),
    (r"\bml\b", "Machine Learning"),
    (r"\bdata[\s-]?science\b", "Data Science"),
    (r"\bdata science\b", "Data Science"),
    (r"\bds\b", "Data Science"),
    (r"\bdeep[\s-]?learning\b", "Deep Learning"),
    (r"\bdl\b", "Deep Learning"),
    (r"\bcomputer[\s-]?vision\b", "Computer Vision"),
    (r"\bcv\b", "Computer Vision"),
    (r"\bai adventures\b", "AI Adventures"),
    (r"\bay ai adventures\b", "AI Adventures"),
]

# Spoken price normalizations ("fifteen thousand" -> "15,000")
PRICE_PATTERNS = [
    (r"\bfifteen thousand\b", "15,000"),
    (r"\btwenty[\s-]?five thousand\b", "25,000"),
    (r"\bten thousand\b", "10,000"),
    (r"\btwenty thousand\b", "20,000"),
    (r"\bthirty thousand\b", "30,000"),
]


def normalize_phone_digits(text: str) -> str:
    """Convert consecutive spoken digit words to digit strings.
    'nine eight seven six five four three two one zero' -> '9876543210'
    """
    tokens = text.split()
    result_tokens = []
    digit_run: list[str] = []

    def flush_run():
        if len(digit_run) >= 3:
            result_tokens.append("".join(digit_run))
        else:
            result_tokens.extend(digit_run)
        digit_run.clear()

    for tok in tokens:
        clean = tok.lower().strip(".,!?")
        if clean in WORD_TO_DIGIT:
            digit_run.append(WORD_TO_DIGIT[clean])
        else:
            if digit_run:
                flush_run()
            result_tokens.append(tok)

    if digit_run:
        flush_run()

    return " ".join(result_tokens)


def normalize_course_names(text: str) -> str:
    """Fix common ASR course name mishearing."""
    result = text
    for pattern, replacement in COURSE_NORMALIZATIONS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def normalize_prices(text: str) -> str:
    """Normalize spoken price expressions to numeric form."""
    result = text
    for pattern, replacement in PRICE_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def normalize_asr_transcript(text: str) -> str:
    """Full normalization pipeline applied to raw Whisper ASR output."""
    if not text or not text.strip():
        return text
    t = text.strip()
    t = normalize_phone_digits(t)
    t = normalize_course_names(t)
    t = normalize_prices(t)
    # Clean double spaces
    t = re.sub(r"\s{2,}", " ", t)
    return t
