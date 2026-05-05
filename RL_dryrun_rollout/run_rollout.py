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


QWEN_BASE_CHAT_TEMPLATE = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""


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
    _ensure_qwen_chat_template(tokenizer, model_cfg)
    eos_token_ids = _resolve_eos_token_ids(tokenizer, model_cfg)
    _inject_eos_generation_config(generation_cfg, eos_token_ids)
    save_yaml(run_dir / "resolved_config.yaml", config)

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
                eos_token_id=generation_cfg["eos_token_id"],
            )
        completion_ids = generated[:, input_len:]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        for (example, gen_idx, _), completion, ids in zip(batch, completions, completion_ids):
            token_count = int((ids != tokenizer.pad_token_id).sum().item()) if tokenizer.pad_token_id is not None else len(ids)
            stopped_by_eos = _stopped_by_eos(ids, eos_token_ids)
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


def _ensure_qwen_chat_template(tokenizer: Any, model_cfg: dict[str, Any]) -> None:
    if model_cfg.get("prompt_template", "qwen_chat") != "qwen_chat":
        return
    if getattr(tokenizer, "chat_template", None):
        return
    tokenizer.chat_template = QWEN_BASE_CHAT_TEMPLATE


def _resolve_eos_token_ids(tokenizer: Any, model_cfg: dict[str, Any]) -> list[int]:
    eos_tokens = model_cfg.get("eos_tokens")
    if eos_tokens is None:
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Tokenizer has no eos_token_id and model.eos_tokens is not configured")
        return [int(eos_token_id)]
    if not isinstance(eos_tokens, list) or not eos_tokens:
        raise TypeError("model.eos_tokens must be a non-empty list of token strings")

    token_ids: list[int] = []
    for token in eos_tokens:
        if not isinstance(token, str):
            raise TypeError("Every item in model.eos_tokens must be a string")
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            raise ValueError(f"Configured eos token is not in the tokenizer vocabulary: {token!r}")
        token_ids.append(int(token_id))
    return list(dict.fromkeys(token_ids))


def _inject_eos_generation_config(generation_cfg: dict[str, Any], eos_token_ids: list[int]) -> None:
    configured = generation_cfg.get("eos_token_id")
    expected = eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0]
    if configured is not None and configured != expected:
        raise ValueError(
            "generation.eos_token_id conflicts with model.eos_tokens. "
            f"Expected {expected!r}, got {configured!r}."
        )
    generation_cfg["eos_token_id"] = expected


def _stopped_by_eos(ids: Any, eos_token_ids: list[int]) -> bool:
    eos_tensor = torch.tensor(eos_token_ids, device=ids.device, dtype=ids.dtype)
    return bool(torch.isin(ids, eos_tensor).any().item())


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
