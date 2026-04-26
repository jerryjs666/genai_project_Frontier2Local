from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any
from collections import Counter

from datasets import load_dataset

from .provider import FatalProviderError, TeacherProvider, parse_response_json
from .utils import (
    answers_match,
    append_jsonl,
    build_bad_record,
    build_success_record,
    classify_output,
    count_teacher_cot_tokens,
    compute_run_dir,
    extract_gsm8k_final_answer,
    init_stats,
    load_config,
    make_example_id,
    make_response_key,
    scan_existing_outputs,
    setup_logger,
    update_stats,
    write_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher data from GSM8K questions.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def load_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_cfg = config["dataset"]
    dataset = load_dataset(
        dataset_cfg["name"],
        dataset_cfg["config"],
        split=dataset_cfg["split"],
    )
    start_index = int(dataset_cfg.get("start_index") or 0)
    if start_index < 0:
        raise ValueError("dataset.start_index must be non-negative")
    limit_value = dataset_cfg.get("limit")
    end_index = len(dataset) if limit_value is None else min(start_index + int(limit_value), len(dataset))
    if start_index > len(dataset):
        raise ValueError(f"dataset.start_index={start_index} is beyond dataset size {len(dataset)}")

    records: list[dict[str, Any]] = []
    for idx in range(start_index, end_index):
        row = dataset[idx]
        records.append(
            {
                "id": make_example_id(idx),
                "question": row["question"],
                "ground_truth_answer": extract_gsm8k_final_answer(row["answer"]),
            }
        )
    return records


async def run_pipeline(config: dict[str, Any], records: list[dict[str, Any]], provider: Any, run_dir: Path, logger: Any) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    success_path = run_dir / "success.jsonl"
    bad_path = run_dir / "bad.jsonl"
    stats_path = run_dir / "run_stats.json"

    if config["run"]["resume"]:
        done_ids, seen_response_keys = scan_existing_outputs(run_dir)
    else:
        done_ids, seen_response_keys = set(), set()
    logger.info("Resume scan complete. done_ids=%s", len(done_ids))

    total_target = len(records)
    stats = _load_resume_stats(run_dir, total_target) if config["run"]["resume"] else init_stats(total_target)
    stats["resume_skipped_count"] = 0
    teacher_model = config["teacher"]["model"]
    max_concurrency = int(config["teacher"]["max_concurrency"])
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_one(item: dict[str, str]) -> dict[str, Any]:
        async with semaphore:
            result = await provider.generate(item["question"])
            return {"item": item, "provider_result": result}

    tasks = []
    written_ids = set(done_ids)
    for item in records[:total_target]:
        if item["id"] in done_ids:
            stats["resume_skipped_count"] += 1
            continue
        tasks.append(asyncio.create_task(process_one(item)))

    fatal_error = None
    for task in asyncio.as_completed(tasks):
        try:
            payload = await task
        except FatalProviderError as exc:
            fatal_error = str(exc)
            logger.error("Fatal provider error: %s", fatal_error)
            for pending in tasks:
                if not pending.done():
                    pending.cancel()
            break

        item = payload["item"]
        result = payload["provider_result"]
        record_id = item["id"]
        if record_id in written_ids:
            continue

        usage = result.usage
        if result.error is not None:
            bad_record = build_bad_record(
                record_id=record_id,
                question=item["question"],
                teacher_model=teacher_model,
                error=result.error,
                filter_reason="api_failed",
                usage=usage,
            )
            append_jsonl(bad_path, bad_record)
            written_ids.add(record_id)
            update_stats(stats, "failed", result.latency_sec, usage, result.attempt_count)
            write_stats(stats_path, stats)
            continue

        try:
            parsed_reasoning, answer = parse_response_json(result.content)
        except json.JSONDecodeError as exc:
            bad_record = build_bad_record(
                record_id=record_id,
                question=item["question"],
                teacher_model=teacher_model,
                error={"kind": "invalid_json", "message": str(exc)},
                filter_reason="invalid_json",
                usage=usage,
                finish_reason=result.finish_reason,
                content_preview=result.content[:200],
            )
            append_jsonl(bad_path, bad_record)
            written_ids.add(record_id)
            update_stats(stats, "filtered", result.latency_sec, usage, result.attempt_count)
            write_stats(stats_path, stats)
            continue
        except KeyError as exc:
            filter_reason = "missing_reasoning" if "reasoning" in str(exc) else "missing_answer"
            bad_record = build_bad_record(
                record_id=record_id,
                question=item["question"],
                teacher_model=teacher_model,
                error={"kind": filter_reason, "message": str(exc)},
                filter_reason=filter_reason,
                usage=usage,
                finish_reason=result.finish_reason,
                content_preview=result.content[:200],
            )
            append_jsonl(bad_path, bad_record)
            written_ids.add(record_id)
            update_stats(stats, "filtered", result.latency_sec, usage, result.attempt_count)
            write_stats(stats_path, stats)
            continue
        except Exception as exc:
            bad_record = build_bad_record(
                record_id=record_id,
                question=item["question"],
                teacher_model=teacher_model,
                error={"kind": "parse_error", "message": str(exc)},
                filter_reason="invalid_json",
                usage=usage,
                finish_reason=result.finish_reason,
                content_preview=result.content[:200],
            )
            append_jsonl(bad_path, bad_record)
            written_ids.add(record_id)
            update_stats(stats, "filtered", result.latency_sec, usage, result.attempt_count)
            write_stats(stats_path, stats)
            continue

        final_reasoning = parsed_reasoning or result.reasoning
        response_key = make_response_key(final_reasoning, answer)
        filter_reason = classify_output(final_reasoning, answer, result.finish_reason, response_key, seen_response_keys)
        if filter_reason is not None:
            bad_record = build_bad_record(
                record_id=record_id,
                question=item["question"],
                teacher_model=teacher_model,
                error=None,
                filter_reason=filter_reason,
                usage=usage,
                finish_reason=result.finish_reason,
                content_preview=result.content[:200],
            )
            append_jsonl(bad_path, bad_record)
            written_ids.add(record_id)
            update_stats(stats, "filtered", result.latency_sec, usage, result.attempt_count)
            write_stats(stats_path, stats)
            continue

        ground_truth_answer = item.get("ground_truth_answer")
        if ground_truth_answer is not None and not answers_match(answer, ground_truth_answer):
            bad_record = build_bad_record(
                record_id=record_id,
                question=item["question"],
                teacher_model=teacher_model,
                error=None,
                filter_reason="answer_mismatch",
                usage=usage,
                finish_reason=result.finish_reason,
                generated_answer=answer,
                ground_truth_answer=ground_truth_answer,
            )
            append_jsonl(bad_path, bad_record)
            written_ids.add(record_id)
            update_stats(stats, "filtered", result.latency_sec, usage, result.attempt_count)
            write_stats(stats_path, stats)
            continue

        teacher_cot_tokens = count_teacher_cot_tokens(final_reasoning)

        success_record = build_success_record(
            record_id=record_id,
            question=item["question"],
            reasoning=final_reasoning,
            answer=answer,
            teacher_model=teacher_model,
            usage=usage,
        )
        append_jsonl(success_path, success_record)
        seen_response_keys.add(response_key)
        written_ids.add(record_id)
        update_stats(stats, "success", result.latency_sec, usage, result.attempt_count, teacher_cot_tokens)
        write_stats(stats_path, stats)

    if fatal_error:
        logger.error("Run stopped early because of a fatal provider error.")
    logger.info(
        "Run complete. success=%s failed=%s filtered=%s skipped=%s",
        stats["success_count"],
        stats["failed_count"],
        stats["filtered_count"],
        stats["skipped_count"],
    )
    write_stats(stats_path, stats)
    return stats


def _load_resume_stats(run_dir: Path, total_target: int) -> dict[str, Any]:
    stats = init_stats(total_target)
    for path in (run_dir / "success.jsonl", run_dir / "bad.jsonl"):
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                usage = record.get("usage")
                if path.name == "success.jsonl":
                    reasoning = str((record.get("response") or {}).get("reasoning") or "")
                    update_stats(stats, "success", 0.0, usage, 1, count_teacher_cot_tokens(reasoning))
                elif record.get("filter_reason") == "api_failed":
                    update_stats(stats, "failed", 0.0, usage, 1)
                else:
                    update_stats(stats, "filtered", 0.0, usage, 1)
    stats["total_target"] = total_target
    stats["_retry_counter"] = Counter({str(k): int(v) for k, v in stats.get("retry_histogram", {}).items()})
    return stats


async def async_main(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    run_dir = compute_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir)
    logger.info("Starting run in %s", run_dir)
    logger.info("Config summary: dataset_start_index=%s dataset_limit=%s model=%s resume=%s concurrency=%s use_thinking=%s",
                config["dataset"].get("start_index", 0),
                config["dataset"].get("limit"),
                config["teacher"]["model"],
                config["run"]["resume"],
                config["teacher"]["max_concurrency"],
                config["teacher"].get("use_thinking", False))
    records = load_records(config)
    provider = TeacherProvider(config)
    return await run_pipeline(config, records, provider, run_dir, logger)


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args.config))


if __name__ == "__main__":
    main()
