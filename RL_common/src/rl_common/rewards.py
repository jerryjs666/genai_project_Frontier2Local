from __future__ import annotations

import re
from statistics import mean, pstdev
from typing import Any, Callable

from .answers import answers_match, extract_answer
from .prompts import completion_to_text


STRICT_FORMAT_RE = re.compile(r"(?is).*the\s+(?:final\s+)?answer\s+is\s*[:=]?\s*[-+]?\$?\d[\d,]*(?:\.\d+)?\s*\.\s*$")


def score_completion(
    completion: str,
    gold_answer: str,
    reward_cfg: dict[str, Any] | None = None,
    completion_token_count: int | None = None,
) -> dict[str, Any]:
    cfg = reward_cfg or {}
    predicted = extract_answer(completion)
    correct = answers_match(predicted, gold_answer)
    parsed = predicted is not None

    answer_reward = float(cfg.get("answer_correct", 1.0)) if correct else float(cfg.get("answer_incorrect", 0.0))
    format_reward = float(cfg.get("format_reward", 0.2)) if has_strict_answer_format(completion) else 0.0
    penalty = 0.0
    if not parsed:
        penalty += float(cfg.get("parse_fail_penalty", -0.1))

    token_count = completion_token_count
    if token_count is None:
        token_count = len(completion.split())
    min_tokens = cfg.get("min_completion_tokens")
    max_tokens = cfg.get("max_completion_tokens")
    if min_tokens is not None and token_count < int(min_tokens):
        penalty += float(cfg.get("length_penalty", -0.05))
    if max_tokens is not None and token_count > int(max_tokens):
        penalty += float(cfg.get("length_penalty", -0.05))

    total_reward = answer_reward + format_reward + penalty
    return {
        "predicted_answer": predicted,
        "correct": correct,
        "parsed": parsed,
        "answer_reward": answer_reward,
        "format_reward": format_reward,
        "penalty": penalty,
        "reward": total_reward,
    }


def has_strict_answer_format(text: str) -> bool:
    return bool(STRICT_FORMAT_RE.match(text.strip()))


def make_answer_reward_func(correct_reward: float = 1.0, incorrect_reward: float = 0.0) -> Callable[..., list[float]]:
    def answer_reward(completions, gold_answer, log_extra=None, log_metric=None, **kwargs):
        texts = [completion_to_text(item) for item in completions]
        extracted = [extract_answer(text) for text in texts]
        rewards = [
            correct_reward if answers_match(predicted, gold) else incorrect_reward
            for predicted, gold in zip(extracted, gold_answer)
        ]
        if log_extra:
            log_extra("gold_answer", list(gold_answer))
            log_extra("extracted_answer", [item or "[none]" for item in extracted])
        if log_metric and rewards:
            log_metric("answer_exact_match", sum(1.0 for item in rewards if item == correct_reward) / len(rewards))
        return [float(item) for item in rewards]

    answer_reward.__name__ = "answer_reward"
    return answer_reward


def make_format_reward_func(format_reward_value: float = 0.2) -> Callable[..., list[float]]:
    def format_reward(completions, log_metric=None, **kwargs):
        texts = [completion_to_text(item) for item in completions]
        rewards = [format_reward_value if has_strict_answer_format(text) else 0.0 for text in texts]
        if log_metric and rewards:
            log_metric("format_exact_rate", sum(1.0 for item in rewards if item > 0) / len(rewards))
        return [float(item) for item in rewards]

    format_reward.__name__ = "format_reward"
    return format_reward


def make_penalty_reward_func(
    parse_fail_penalty: float = -0.1,
    length_penalty: float = -0.05,
    min_completion_tokens: int | None = None,
    max_completion_tokens: int | None = None,
) -> Callable[..., list[float]]:
    def penalty_reward(completions, completion_ids=None, log_metric=None, **kwargs):
        texts = [completion_to_text(item) for item in completions]
        rewards = []
        parse_fail_count = 0
        length_penalty_count = 0
        for idx, text in enumerate(texts):
            reward = 0.0
            if extract_answer(text) is None:
                reward += parse_fail_penalty
                parse_fail_count += 1
            token_count = len(completion_ids[idx]) if completion_ids is not None else len(text.split())
            too_short = min_completion_tokens is not None and token_count < min_completion_tokens
            too_long = max_completion_tokens is not None and token_count > max_completion_tokens
            if too_short or too_long:
                reward += length_penalty
                length_penalty_count += 1
            rewards.append(float(reward))
        if log_metric and rewards:
            log_metric("parse_fail_rate", parse_fail_count / len(rewards))
            log_metric("length_penalty_rate", length_penalty_count / len(rewards))
        return rewards

    penalty_reward.__name__ = "penalty_reward"
    return penalty_reward


def compute_group_diagnostics(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        return {
            "prompt_count": 0,
            "completion_count": 0,
            "all_correct_rate": 0.0,
            "all_wrong_rate": 0.0,
            "mixed_rate": 0.0,
            "parse_fail_rate": 0.0,
            "eos_stop_rate": 0.0,
            "avg_completion_length": 0.0,
            "reward_distribution": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
        }

    all_correct = 0
    all_wrong = 0
    mixed = 0
    parse_fail = 0
    rewards: list[float] = []
    lengths: list[int] = []
    completion_count = 0
    eos_stop_count = 0

    for group in groups:
        completions = group.get("completions", [])
        correctness = [bool(item.get("correct")) for item in completions]
        parsed = [bool(item.get("parsed")) for item in completions]
        if correctness and all(correctness):
            all_correct += 1
        elif correctness and not any(correctness):
            all_wrong += 1
        else:
            mixed += 1
        parse_fail += sum(1 for item in parsed if not item)
        completion_count += len(completions)
        eos_stop_count += sum(1 for item in completions if bool(item.get("stopped_by_eos", False)))
        rewards.extend(float(item.get("reward", 0.0)) for item in completions)
        lengths.extend(int(item.get("completion_tokens", item.get("completion_length", 0))) for item in completions)

    prompt_count = len(groups)
    return {
        "prompt_count": prompt_count,
        "completion_count": completion_count,
        "all_correct_rate": round(all_correct / prompt_count, 6),
        "all_wrong_rate": round(all_wrong / prompt_count, 6),
        "mixed_rate": round(mixed / prompt_count, 6),
        "parse_fail_rate": round(parse_fail / completion_count, 6) if completion_count else 0.0,
        "eos_stop_rate": round(eos_stop_count / completion_count, 6) if completion_count else 0.0,
        "avg_completion_length": round(mean(lengths), 6) if lengths else 0.0,
        "reward_distribution": {
            "mean": round(mean(rewards), 6) if rewards else 0.0,
            "std": round(pstdev(rewards), 6) if len(rewards) > 1 else 0.0,
            "min": round(min(rewards), 6) if rewards else 0.0,
            "max": round(max(rewards), 6) if rewards else 0.0,
        },
    }
