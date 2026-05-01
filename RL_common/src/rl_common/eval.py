from __future__ import annotations

import time
from statistics import mean
from typing import Any

import torch
from tqdm.auto import tqdm

from .answers import answers_match
from .prompts import render_generation_prompt
from .rewards import score_completion


def run_greedy_eval(
    model: Any,
    tokenizer: Any,
    examples: list[Any],
    prompt_cfg: dict[str, Any],
    eval_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = eval_cfg or {}
    batch_size = int(cfg.get("batch_size", 8))
    max_new_tokens = int(cfg.get("max_new_tokens", 256))
    max_prompt_length = cfg.get("max_prompt_length")
    reward_cfg = cfg.get("reward", {})
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    results = []
    correct = 0
    parse_success = 0
    output_tokens = []
    total_generated_tokens = 0
    start_all = time.perf_counter()

    for start in tqdm(range(0, len(examples), batch_size), desc=cfg.get("desc", "Evaluating")):
        batch = examples[start : start + batch_size]
        prompts = [render_generation_prompt(tokenizer, item.question, prompt_cfg) for item in batch]
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=max_prompt_length is not None,
            max_length=max_prompt_length,
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        input_len = tokenized["input_ids"].shape[1]
        start_batch = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        batch_latency = time.perf_counter() - start_batch
        completion_ids = generated[:, input_len:]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        for item, completion, ids in zip(batch, completions, completion_ids):
            token_count = int((ids != tokenizer.pad_token_id).sum().item()) if tokenizer.pad_token_id is not None else len(ids)
            score = score_completion(completion, item.gold_answer, reward_cfg, token_count)
            is_correct = answers_match(score["predicted_answer"], item.gold_answer)
            correct += int(is_correct)
            parse_success += int(score["parsed"])
            output_tokens.append(token_count)
            total_generated_tokens += token_count
            results.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "gold_answer": item.gold_answer,
                    "predicted_answer": score["predicted_answer"],
                    "correct": is_correct,
                    "parsed": score["parsed"],
                    "completion": completion,
                    "completion_tokens": token_count,
                    "batch_latency_s": round(batch_latency, 6),
                }
            )

    total_latency = time.perf_counter() - start_all
    total = len(examples)
    if was_training:
        model.train()
    return {
        "summary": {
            "total": total,
            "correct": correct,
            "exact_match": round(correct / total, 6) if total else 0.0,
            "parse_success_rate": round(parse_success / total, 6) if total else 0.0,
            "avg_output_tokens": round(mean(output_tokens), 6) if output_tokens else 0.0,
            "total_latency_s": round(total_latency, 6),
            "samples_per_second": round(total / total_latency, 6) if total_latency > 0 else 0.0,
            "tokens_per_second": round(total_generated_tokens / total_latency, 6) if total_latency > 0 else 0.0,
        },
        "results": results,
    }
