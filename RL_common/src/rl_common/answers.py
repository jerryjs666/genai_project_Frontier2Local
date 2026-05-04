from __future__ import annotations

import re
from decimal import Decimal, DivisionByZero, InvalidOperation


NUMBER_PATTERN = r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?"
ANSWER_IS_RE = re.compile(
    rf"(?i)\bthe\s+(?:final\s+)?answer\s+is\s*[:=]?\s*(?:\\\(|\$)?\s*({NUMBER_PATTERN})"
)
GSM8K_MARKER_RE = re.compile(rf"####\s*({NUMBER_PATTERN})")
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
NUMBER_RE = re.compile(NUMBER_PATTERN)
NO_ANSWER_RE = re.compile(
    r"(?i)\b(?:"
    r"error in (?:the )?(?:problem|question|problem statement)|"
    r"mistake in (?:the )?(?:problem|question|problem statement)|"
    r"not possible|cannot be determined|can't determine|"
    r"not enough information|insufficient information|ambiguous|does not make sense"
    r")\b"
)
CONCLUSION_CUE_RE = re.compile(
    r"(?i)\b(?:"
    r"so|therefore|thus|hence|finally|in total|total|altogether|together|overall|"
    r"result|equals?|comes to|gives|makes?|earns?|costs?|spends?|saves?|runs?|"
    r"needs?|left|remaining"
    r")\b"
)


def extract_gsm8k_gold_answer(answer_text: str) -> str:
    marker = "####"
    if marker not in answer_text:
        raise ValueError("GSM8K answer is missing final-answer marker '####'")
    return normalize_answer(answer_text.rsplit(marker, 1)[1]) or ""


def extract_answer(text: str | None) -> str | None:
    if not text:
        return None

    match = ANSWER_IS_RE.search(text)
    if match:
        return normalize_answer(match.group(1))

    match = GSM8K_MARKER_RE.search(text)
    if match:
        return normalize_answer(match.group(1))

    boxed_matches = BOXED_RE.findall(text)
    for boxed in reversed(boxed_matches):
        number = _last_number(boxed)
        if number is not None:
            return normalize_answer(number)

    number = _fallback_last_number(text)
    if number is not None:
        return normalize_answer(number)

    number = _fallback_conclusion_number(text)
    if number is not None:
        return normalize_answer(number)
    return None


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    normalized = str(answer).strip()
    if not normalized:
        return None

    normalized = normalized.strip(" \t\r\n`*_")
    normalized = normalized.removeprefix("$").strip()
    normalized = normalized.rstrip(".。").strip()
    normalized = normalized.replace(",", "")
    normalized = re.sub(r"\s+", "", normalized)
    if normalized.startswith("+"):
        normalized = normalized[1:]

    numeric = _to_decimal(normalized)
    if numeric is not None:
        return _decimal_to_string(numeric)
    return normalized.lower()


def answers_match(predicted: str | None, gold: str | None) -> bool:
    predicted_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    if predicted_norm is None or gold_norm is None:
        return False
    if predicted_norm == gold_norm:
        return True

    predicted_decimal = _to_decimal(predicted_norm)
    gold_decimal = _to_decimal(gold_norm)
    return predicted_decimal is not None and gold_decimal is not None and predicted_decimal == gold_decimal


def _last_number(text: str) -> str | None:
    matches = NUMBER_RE.findall(text)
    return matches[-1] if matches else None


def _fallback_conclusion_number(text: str) -> str | None:
    if NO_ANSWER_RE.search(text):
        return None

    tail = text.strip()[-1000:]
    if not tail:
        return None

    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", tail.replace("\n", " "))
        if item.strip()
    ]
    for sentence in reversed(sentences[-4:]):
        if sentence.endswith("?"):
            continue
        if not CONCLUSION_CUE_RE.search(sentence):
            continue
        number = _last_number(sentence)
        if number is not None:
            return number
    return None


def _fallback_last_number(text: str) -> str | None:
    stripped = text.strip()
    if NO_ANSWER_RE.search(stripped):
        return None
    if stripped.endswith("?"):
        return None
    return _last_number(stripped)


def _to_decimal(value: str) -> Decimal | None:
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return Decimal(numerator) / Decimal(denominator)
        return Decimal(value)
    except (DivisionByZero, InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _decimal_to_string(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f").rstrip("0").rstrip(".")
