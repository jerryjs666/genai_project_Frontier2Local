from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
COMMON_SRC = ROOT / "RL_common" / "src"
if COMMON_SRC.exists() and str(COMMON_SRC) not in sys.path:
    sys.path.insert(0, str(COMMON_SRC))

from rl_common.config import format_run_name, load_config, make_run_dir, save_yaml
from rl_common.data import load_examples
from rl_common.model import load_policy_model, load_tokenizer
from rl_common.prompts import render_generation_prompt
from rl_common.rewards import compute_group_diagnostics, score_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sampling dry-run rollouts for GSM8K GRPO diagnostics.")
    parser.add_argument("--config", required=True, help="Path to rollout YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["run"]["resolved_name"] = format_run_name(config["run"]["name"], config)
    run_dir = make_run_dir(config["run"]["output_root"], config["run"]["resolved_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(run_dir / "resolved_config.yaml", config)

    model_cfg = config["model"]
    prompt_cfg = {
        "template": model_cfg.get("prompt_template", "qwen_chat"),
        "system_prompt": model_cfg.get("system_prompt", ""),
        "include_empty_system": model_cfg.get("include_empty_system", False),
    }
    generation_cfg = config.get("generation", {})
    reward_cfg = config.get("reward", {})

    print(f"Loading dataset: {config['dataset'].get('kind')}")
    examples = load_examples(config["dataset"])
    print(f"Loaded {len(examples)} prompts")

    print(f"Loading tokenizer: {model_cfg.get('tokenizer_name_or_path') or model_cfg['base_model_name_or_path']}")
    tokenizer = load_tokenizer(model_cfg)
    print(f"Loading model: {model_cfg['base_model_name_or_path']}")
    model = load_policy_model(model_cfg, is_trainable_adapter=False)
    model.eval()
    device = next(model.parameters()).device

    groups_by_id: dict[str, dict[str, Any]] = {
        item.id: {
            "id": item.id,
            "question": item.question,
            "gold_answer": item.gold_answer,
            "completions": [],
        }
        for item in examples
    }

    requests = []
    num_generations = int(generation_cfg.get("num_generations", 4))
    for item in examples:
        prompt = render_generation_prompt(tokenizer, item.question, prompt_cfg)
        for gen_idx in range(num_generations):
            requests.append((item, gen_idx, prompt))

    batch_size = int(generation_cfg.get("batch_size", 8))
    max_prompt_length = generation_cfg.get("max_prompt_length")
    for start in tqdm(range(0, len(requests), batch_size), desc="Sampling rollouts"):
        batch = requests[start : start + batch_size]
        prompts = [item[2] for item in batch]
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=max_prompt_length is not None,
            max_length=max_prompt_length,
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        input_len = tokenized["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                do_sample=bool(generation_cfg.get("do_sample", True)),
                temperature=float(generation_cfg.get("temperature", 0.9)),
                top_p=float(generation_cfg.get("top_p", 0.95)),
                max_new_tokens=int(generation_cfg.get("max_new_tokens", 256)),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[:, input_len:]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        for (example, gen_idx, _), completion, ids in zip(batch, completions, completion_ids):
            token_count = int((ids != tokenizer.pad_token_id).sum().item()) if tokenizer.pad_token_id is not None else len(ids)
            stopped_by_eos = bool((ids == tokenizer.eos_token_id).any().item()) if tokenizer.eos_token_id is not None else False
            score = score_completion(completion, example.gold_answer, reward_cfg, token_count)
            groups_by_id[example.id]["completions"].append(
                {
                    "generation_index": gen_idx,
                    "completion": completion,
                    "completion_tokens": token_count,
                    "stopped_by_eos": stopped_by_eos,
                    **score,
                }
            )

    groups = list(groups_by_id.values())
    summary = compute_group_diagnostics(groups)
    _write_jsonl(run_dir / "rollouts.jsonl", groups)
    _write_json(run_dir / "summary.json", summary)
    _write_markdown_samples(run_dir / "sampled_outputs.md", groups, int(config.get("inspection", {}).get("sample_count", 20)))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote rollout outputs to {run_dir}")


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_markdown_samples(path: Path, groups: list[dict[str, Any]], sample_count: int) -> None:
    lines = ["# Dry-run Sampled Outputs", ""]
    for group in groups[:sample_count]:
        lines.extend(
            [
                f"## {group['id']}",
                "",
                f"**Question**: {group['question']}",
                "",
                f"**Gold**: `{group['gold_answer']}`",
                "",
            ]
        )
        for item in group["completions"]:
            lines.extend(
                [
                    f"### generation {item['generation_index']}",
                    "",
                    f"- predicted: `{item['predicted_answer']}`",
                    f"- correct: `{item['correct']}`",
                    f"- reward: `{item['reward']}`",
                    "",
                    "```text",
                    item["completion"].strip(),
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
