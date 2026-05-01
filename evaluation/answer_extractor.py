"""
answer_extractor.py
-------------------
Robust answer extraction and correctness checking for GSM8K evaluation.

Fixes over the original notebook version:
  - Fractions like 5/6 are correctly parsed
  - Trailing punctuation like "26.0." is stripped before comparison
  - Comma-separated thousands like "1,000" are handled
"""

import re
from fractions import Fraction


def _clean(text: str) -> str:
    """Remove commas and strip trailing punctuation from a numeric string."""
    return text.replace(",", "").strip().rstrip(".").strip()


def _to_float(text: str):
    """
    Convert string to float. Supports plain numbers and fractions (e.g. 5/6).
    Returns None on failure.
    """
    text = _clean(text)
    try:
        if "/" in text:
            return float(Fraction(text))
        return float(text)
    except (ValueError, ZeroDivisionError):
        return None


def extract_answer(text: str):
    """
    Extract the final numeric answer from a model response.

    Priority:
      1. GSM8K standard marker : #### <number>
      2. Explicit phrase       : "The answer is <number>"
      3. Last number in text   : plain number or fraction

    Returns a cleaned string, or None if nothing found.
    """
    if not text:
        return None

    # 1. #### marker
    match = re.search(r'####\s*([\d,./\-]+)', text)
    if match:
        return _clean(match.group(1))

    # 2. "The answer is X"
    match = re.search(r'[Tt]he answer is[:\s]*([\d,./\-]+)', text)
    if match:
        return _clean(match.group(1))

    # 3. Last number-like token (supports fractions)
    numbers = re.findall(r'\d+/\d+|[\d,]+\.?\d*', text)
    if numbers:
        return _clean(numbers[-1])

    return None


def official_gold(answer_text: str) -> str:
    """Extract gold answer from GSM8K ground-truth answer string."""
    return extract_answer(answer_text)


def is_correct(predicted: str, gold: str) -> bool:
    """
    Check if predicted answer matches gold answer.
    Handles fractions, trailing punctuation, and float comparison.
    """
    if predicted is None or gold is None:
        return False

    pred_val = _to_float(predicted)
    gold_val = _to_float(gold)

    if pred_val is not None and gold_val is not None:
        return abs(pred_val - gold_val) < 1e-4

    # Fallback: string comparison
    return _clean(predicted).lower() == _clean(gold).lower()
