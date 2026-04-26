from __future__ import annotations

import json
import logging
import re
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "start_index": 0,
        "limit": 5000,
    },
    "teacher": {
        "model": "qwen3.5-397b-a17b",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "",
        "api_key_env": "DASHSCOPE_API_KEY",
        "use_thinking": False,
        "system_prompt_path": "prompts/teacher_system_prompt.txt",
        "max_concurrency": 10,
        "timeout_sec": 90,
        "max_retries": 2,
        "backoff_base_sec": 2,
    },
    "run": {
        "output_root": "outputs/runs",
        "run_name": None,
        "resume": True,
    },
}


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    config = deepcopy(DEFAULT_CONFIG)
    _deep_update(config, raw)
    return config


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def compute_run_dir(config: dict[str, Any]) -> Path:
    run_cfg = config["run"]
    run_name = run_cfg.get("run_name") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(run_cfg["output_root"]) / run_name


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("teacher_data_gen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def make_example_id(index: int) -> str:
    return f"gsm8k_train_{index:05d}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def scan_existing_outputs(run_dir: Path) -> tuple[set[str], set[str]]:
    done_ids: set[str] = set()
    response_keys: set[str] = set()
    success_path = run_dir / "success.jsonl"
    bad_path = run_dir / "bad.jsonl"

    for path in (success_path, bad_path):
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record_id = record.get("id")
                if record_id:
                    done_ids.add(record_id)
                response = record.get("response") or {}
                reasoning = str(response.get("reasoning") or "").strip()
                answer = str(response.get("answer") or "").strip()
                if reasoning or answer:
                    response_keys.add(make_response_key(reasoning, answer))
    return done_ids, response_keys


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def init_stats(total_target: int) -> dict[str, Any]:
    return {
        "total_target": total_target,
        "success_count": 0,
        "failed_count": 0,
        "filtered_count": 0,
        "skipped_count": 0,
        "resume_skipped_count": 0,
        "avg_latency_sec": 0.0,
        "failure_rate": 0.0,
        "usage": {
            "prompt_tokens_sum": 0,
            "completion_tokens_sum": 0,
            "total_tokens_sum": 0,
            "usage_missing_ratio": 0.0,
        },
        "teacher_cot_tokens_sum": 0,
        "teacher_cot_tokens_count": 0,
        "avg_teacher_cot_tokens": 0.0,
        "retry_histogram": {},
    }


def update_stats(
    stats: dict[str, Any],
    record_type: str,
    latency_sec: float,
    usage: dict[str, int] | None,
    attempt_count: int,
    teacher_cot_tokens: int | None = None,
) -> None:
    is_processed = record_type in {"success", "failed", "filtered"}
    if record_type == "success":
        stats["success_count"] += 1
    elif record_type == "failed":
        stats["failed_count"] += 1
    elif record_type == "filtered":
        stats["filtered_count"] += 1
    elif record_type == "skipped":
        stats["skipped_count"] += 1

    processed = stats["success_count"] + stats["failed_count"] + stats["filtered_count"]
    if is_processed and processed > 0 and latency_sec >= 0:
        old_total = stats["avg_latency_sec"] * (processed - 1)
        stats["avg_latency_sec"] = round((old_total + latency_sec) / processed, 6)

    if is_processed and usage is None:
        stats.setdefault("_usage_missing_count", 0)
        stats["_usage_missing_count"] += 1
    elif is_processed:
        stats["usage"]["prompt_tokens_sum"] += int(usage.get("prompt_tokens", 0))
        stats["usage"]["completion_tokens_sum"] += int(usage.get("completion_tokens", 0))
        stats["usage"]["total_tokens_sum"] += int(usage.get("total_tokens", 0))

    if is_processed:
        retry_count = max(0, attempt_count - 1)
        stats.setdefault("_retry_counter", Counter())
        stats["_retry_counter"][str(retry_count)] += 1

    if record_type == "success" and teacher_cot_tokens is not None:
        stats["teacher_cot_tokens_sum"] += teacher_cot_tokens
        stats["teacher_cot_tokens_count"] += 1
        stats["avg_teacher_cot_tokens"] = round(
            stats["teacher_cot_tokens_sum"] / stats["teacher_cot_tokens_count"],
            6,
        )

    denominator = processed if processed > 0 else 1
    stats["failure_rate"] = round(stats["failed_count"] / denominator, 6)
    missing_usage = stats.get("_usage_missing_count", 0)
    stats["usage"]["usage_missing_ratio"] = round(missing_usage / denominator, 6)
    stats["retry_histogram"] = dict(stats.get("_retry_counter", Counter()))


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    stats = deepcopy(stats)
    stats.pop("_usage_missing_count", None)
    retry_counter = stats.pop("_retry_counter", None)
    if retry_counter is not None:
        stats["retry_histogram"] = dict(retry_counter)
    return stats


def write_stats(path: Path, stats: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(finalize_stats(stats), f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_success_record(
    record_id: str,
    question: str,
    reasoning: str,
    answer: str,
    teacher_model: str,
    usage: dict[str, int] | None,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "question": question,
        "response": {
            "reasoning": reasoning,
            "answer": answer,
        },
        "created_at": utc_now_iso(),
        "teacher_model": teacher_model,
        "usage": usage,
    }


def build_bad_record(
    record_id: str,
    question: str,
    teacher_model: str,
    error: dict[str, Any] | None,
    filter_reason: str | None,
    usage: dict[str, int] | None,
    finish_reason: str | None = None,
    content_preview: str | None = None,
    generated_answer: str | None = None,
    ground_truth_answer: str | None = None,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "question": question,
        "created_at": utc_now_iso(),
        "teacher_model": teacher_model,
        "error": error,
        "filter_reason": filter_reason,
        "usage": usage,
        "finish_reason": finish_reason,
        "content_preview": content_preview,
        "generated_answer": generated_answer,
        "ground_truth_answer": ground_truth_answer,
    }


def classify_output(reasoning: str, answer: str, finish_reason: str | None, response_key: str, seen_response_keys: set[str]) -> str | None:
    if not reasoning.strip():
        return "empty_reasoning"
    if not answer.strip():
        return "empty_answer"
    if finish_reason == "length":
        return "truncated"
    if response_key in seen_response_keys:
        return "duplicate_response"
    return None


def make_response_key(reasoning: str, answer: str) -> str:
    return f"{reasoning.strip()}\n{answer.strip()}"


def extract_gsm8k_final_answer(answer_text: str) -> str:
    marker = "####"
    if marker not in answer_text:
        raise ValueError("GSM8K answer is missing final-answer marker")
    return answer_text.rsplit(marker, 1)[1].strip()


def answers_match(generated_answer: str, ground_truth_answer: str) -> bool:
    generated = _normalize_answer(generated_answer)
    ground_truth = _normalize_answer(ground_truth_answer)
    if generated == ground_truth:
        return True
    generated_decimal = _to_decimal(generated)
    ground_truth_decimal = _to_decimal(ground_truth)
    return (
        generated_decimal is not None
        and ground_truth_decimal is not None
        and generated_decimal == ground_truth_decimal
    )


def count_teacher_cot_tokens(reasoning: str) -> int:
    return len(re.findall(r"\d+\.\d+|\w+|[^\w\s]", reasoning, flags=re.UNICODE))


def _normalize_answer(answer: str) -> str:
    normalized = answer.strip()
    normalized = normalized.removeprefix("$").strip()
    normalized = normalized.removesuffix(".").strip()
    normalized = normalized.replace(",", "")
    return normalized


def _to_decimal(answer: str) -> Decimal | None:
    try:
        return Decimal(answer)
    except InvalidOperation:
        return None
