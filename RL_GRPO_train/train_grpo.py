from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

ROOT = Path(__file__).resolve().parents[1]
COMMON_SRC = ROOT / "RL_common" / "src"
if COMMON_SRC.exists() and str(COMMON_SRC) not in sys.path:
    sys.path.insert(0, str(COMMON_SRC))

from rl_common.config import format_run_name, load_config, make_run_dir, save_yaml
from rl_common.data import build_trl_dataset, load_examples
from rl_common.eval import run_greedy_eval
from rl_common.model import load_policy_model, load_tokenizer, save_adapter_or_model
from rl_common.rewards import make_answer_reward_func, make_format_reward_func, make_penalty_reward_func, make_eos_reward_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GSM8K GRPO baseline with TRL.")
    parser.add_argument("--config", required=True, help="Path to GRPO YAML config.")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and run greedy evaluation. Uses eval_dataset unless --final-test is set.",
    )
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="With --eval-only, evaluate final_eval_dataset. Do not use this during training.",
    )
    return parser.parse_args()


class GreedyEvalAndBestSaveCallback(TrainerCallback):
    def __init__(
        self,
        *,
        tokenizer: Any,
        eval_examples: list[Any],
        prompt_cfg: dict[str, Any],
        eval_cfg: dict[str, Any],
        run_dir: Path,
    ) -> None:
        self.tokenizer = tokenizer
        self.eval_examples = eval_examples
        self.prompt_cfg = prompt_cfg
        self.eval_cfg = eval_cfg
        self.run_dir = run_dir
        self.eval_steps = int(eval_cfg.get("every_steps", 50))
        self.metric_for_best = eval_cfg.get("metric_for_best", "exact_match")
        self.best_metric: float | None = None
        self.best_step: int | None = None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return control
        if state.global_step <= 0 or state.global_step % self.eval_steps != 0:
            return control
        if model is None:
            return control
        self.evaluate_and_save_if_best(model, state.global_step)
        return control

    def evaluate_and_save_if_best(self, model: Any, step: int) -> dict[str, Any]:
        result = run_greedy_eval(
            model=model,
            tokenizer=self.tokenizer,
            examples=self.eval_examples,
            prompt_cfg=self.prompt_cfg,
            eval_cfg={**self.eval_cfg, "desc": f"Greedy eval step {step}"},
        )
        summary = result["summary"]
        summary["step"] = step
        _write_json(self.run_dir / "latest_eval_results.json", result)
        _try_swanlab_log(summary, step)

        metric = float(summary[self.metric_for_best])
        if self.best_metric is None or metric > self.best_metric:
            self.best_metric = metric
            self.best_step = step
            final_adapter = self.run_dir / "final_adapter"
            if final_adapter.exists():
                shutil.rmtree(final_adapter)
            save_adapter_or_model(model, final_adapter)
            _write_json(self.run_dir / "best_eval_results.json", result)
        return result


class JsonlTrainerLogCallback(TrainerCallback):
    """Mirror every Trainer/TRL log row to a local JSONL file.

    TRL's GRPOTrainer emits reward/KL metrics through the normal Trainer
    logging path when those metrics are available. Saving each on_log payload
    makes the reward/KL values recoverable even when they are not included in
    custom summary files.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
        if not state.is_world_process_zero:
            return control

        row = dict(logs)
        row["step"] = state.global_step

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return control


class PerResponseRewardLogger:
    """Wrap reward functions and save per-completion rewards to JSONL.

    GRPOTrainer calls each reward function with a batch of completions. This
    wrapper records the reward assigned to each completion by each reward
    component. The resulting file can be grouped by reward_call_id and
    sample_index_in_batch to inspect the total reward for each sampled response.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.call_id = 0

    def wrap(self, name: str, reward_func: Any):
        def wrapped_reward_func(*args, **kwargs):
            rewards = reward_func(*args, **kwargs)
            call_id = self.call_id
            self.call_id += 1

            prompts = kwargs.get("prompts")
            completions = kwargs.get("completions")
            completion_ids = kwargs.get("completion_ids")
            answers = (
                kwargs.get("answer")
                or kwargs.get("answers")
                or kwargs.get("gold_answer")
                or kwargs.get("gold_answers")
            )

            rows = []
            for i, reward in enumerate(rewards):
                row: dict[str, Any] = {
                    "reward_call_id": call_id,
                    "reward_name": name,
                    "sample_index_in_batch": i,
                    "reward": _json_safe_scalar(reward),
                }

                if prompts is not None and i < len(prompts):
                    row["prompt"] = _json_safe_value(prompts[i])
                if completions is not None and i < len(completions):
                    row["completion"] = _json_safe_value(completions[i])
                if completion_ids is not None and i < len(completion_ids):
                    row["completion_ids"] = _json_safe_value(completion_ids[i])
                if answers is not None and i < len(answers):
                    row["answer"] = _json_safe_value(answers[i])

                rows.append(row)

            with self.log_path.open("a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            return rewards

        return wrapped_reward_func


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["run"]["resolved_name"] = format_run_name(config["run"]["name"], config)
    if "grpo" in config:
        config["grpo"]["run_name"] = format_run_name(
            config["grpo"].get("run_name") or config["run"]["resolved_name"],
            config,
        )
    run_dir = make_run_dir(config["run"]["output_root"], config["run"]["resolved_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(run_dir / "resolved_config.yaml", config)

    model_cfg = config["model"]
    prompt_cfg = {
        "template": model_cfg.get("prompt_template", "qwen_chat"),
        "system_prompt": model_cfg.get("system_prompt", ""),
        "include_empty_system": model_cfg.get("include_empty_system", False),
    }
    reward_cfg = config["reward"]

    tokenizer = load_tokenizer(model_cfg)
    model = load_policy_model(model_cfg, is_trainable_adapter=not args.eval_only)

    if args.eval_only:
        dataset_cfg = config["final_eval_dataset"] if args.final_test else config["eval_dataset"]
        examples = load_examples(dataset_cfg)
        result = run_greedy_eval(model, tokenizer, examples, prompt_cfg, {**config["eval"], "reward": reward_cfg})
        output_name = "final_test_eval_results.json" if args.final_test else "eval_only_results.json"
        _write_json(run_dir / output_name, result)
        print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
        return

    from trl import GRPOConfig, GRPOTrainer

    train_examples = load_examples(config["train_dataset"])
    eval_examples = load_examples(config["eval_dataset"])
    train_dataset = build_trl_dataset(train_examples, prompt_cfg)

    per_response_reward_logger = PerResponseRewardLogger(run_dir / "per_response_rewards.jsonl")
    raw_reward_funcs = [
        (
            "answer_reward",
            make_answer_reward_func(
                correct_reward=float(reward_cfg.get("answer_correct", 1.0)),
                incorrect_reward=float(reward_cfg.get("answer_incorrect", 0.0)),
            ),
        ),
        (
            "format_reward",
            make_format_reward_func(float(reward_cfg.get("format_reward", 0.2))),
        ),
        (
            "eos_reward",
            make_eos_reward_func(float(reward_cfg.get("eos_reward", 0.2))),
        ),
        (
            "penalty_reward",
            make_penalty_reward_func(
                parse_fail_penalty=float(reward_cfg.get("parse_fail_penalty", -0.1)),
                length_penalty=float(reward_cfg.get("length_penalty", -0.05)),
                min_completion_tokens=reward_cfg.get("min_completion_tokens"),
                max_completion_tokens=reward_cfg.get("max_completion_tokens"),
                trailing_text_penalty=float(reward_cfg.get("trailing_text_penalty", -0.5)),
            ),
        ),
    ]
    reward_funcs = [
        per_response_reward_logger.wrap(name, func)
        for name, func in raw_reward_funcs
    ]

    grpo_args = _build_grpo_config(GRPOConfig, config["grpo"], run_dir)
    callback = GreedyEvalAndBestSaveCallback(
        tokenizer=tokenizer,
        eval_examples=eval_examples,
        prompt_cfg=prompt_cfg,
        eval_cfg={**config["eval"], "reward": reward_cfg},
        run_dir=run_dir,
    )

    metrics_callback = JsonlTrainerLogCallback(run_dir / "trainer_metrics.jsonl")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[callback, metrics_callback],
    )
    train_output = trainer.train()

    trainer.state.save_to_json(str(run_dir / "trainer_state.json"))
    _write_json(run_dir / "trainer_log_history.json", trainer.state.log_history)
    jsonl_log_history = _read_jsonl(run_dir / "trainer_metrics.jsonl")
    combined_log_history = _merge_log_rows(trainer.state.log_history, jsonl_log_history)
    reward_kl_summary = _summarize_reward_kl(combined_log_history)
    _write_json(run_dir / "reward_kl_summary.json", reward_kl_summary)

    per_response_reward_rows = _read_jsonl(run_dir / "per_response_rewards.jsonl")
    per_response_reward_summary = _summarize_per_response_rewards(per_response_reward_rows)
    _write_json(run_dir / "per_response_reward_summary.json", per_response_reward_summary)

    if callback.best_metric is None:
        callback.evaluate_and_save_if_best(trainer.model, trainer.state.global_step)

    train_summary = {
        "global_step": trainer.state.global_step,
        "train_metrics": train_output.metrics,
        "reward_kl_metrics": reward_kl_summary,
        "per_response_reward_metrics": per_response_reward_summary,
        "best_step": callback.best_step,
        "best_metric": callback.best_metric,
        "best_metric_name": f"sft_val/{callback.metric_for_best}",
    }
    _write_json(run_dir / "train_summary.json", train_summary)
    print(json.dumps(train_summary, indent=2, ensure_ascii=False))


def _build_grpo_config(grpo_config_cls: Any, raw_cfg: dict[str, Any], run_dir: Path):
    cfg = dict(raw_cfg)
    cfg["output_dir"] = str(run_dir / "trainer_state")
    signature = inspect.signature(grpo_config_cls.__init__)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if not accepts_kwargs:
        accepted = set(signature.parameters)
        ignored = sorted(key for key in cfg if key not in accepted)
        if ignored:
            print(f"Ignoring unsupported GRPOConfig keys: {ignored}")
        cfg = {key: value for key, value in cfg.items() if key in accepted}
    return grpo_config_cls(**cfg)


def _try_swanlab_log(summary: dict[str, Any], step: int) -> None:
    try:
        import swanlab

        swanlab.log({f"sft_val/{key}": value for key, value in summary.items() if isinstance(value, (int, float))}, step=step)
    except Exception:
        return




def _json_safe_scalar(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().tolist()
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    return _json_safe_scalar(value)


def _summarize_per_response_rewards(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize component and total rewards saved by PerResponseRewardLogger."""
    numeric_rows = [
        row for row in rows
        if isinstance(row.get("reward"), (int, float))
    ]

    component_metrics: dict[str, dict[str, float | int]] = {}
    reward_names = sorted({str(row.get("reward_name")) for row in numeric_rows})
    for reward_name in reward_names:
        values = [
            float(row["reward"])
            for row in numeric_rows
            if str(row.get("reward_name")) == reward_name
        ]
        if values:
            component_metrics[reward_name] = {
                "count": len(values),
                "last": values[-1],
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }

    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    component_count = len(reward_names) or 1
    for row in numeric_rows:
        call_id = int(row.get("reward_call_id", -1))
        sample_idx = int(row.get("sample_index_in_batch", -1))
        response_group_id = call_id // component_count
        key = (response_group_id, sample_idx)
        group = grouped.setdefault(
            key,
            {
                "response_group_id": response_group_id,
                "sample_index_in_batch": sample_idx,
                "total_reward": 0.0,
                "components": {},
            },
        )
        reward_name = str(row.get("reward_name"))
        reward_value = float(row["reward"])
        group["components"][reward_name] = reward_value
        group["total_reward"] += reward_value
        for optional_key in ("prompt", "completion", "answer"):
            if optional_key in row and optional_key not in group:
                group[optional_key] = row[optional_key]

    total_rewards = [float(group["total_reward"]) for group in grouped.values()]
    total_reward_metrics = {}
    if total_rewards:
        total_reward_metrics = {
            "count": len(total_rewards),
            "last": total_rewards[-1],
            "mean": sum(total_rewards) / len(total_rewards),
            "min": min(total_rewards),
            "max": max(total_rewards),
        }

    recent_totals = sorted(
        grouped.values(),
        key=lambda row: (row["reward_call_id"], row["sample_index_in_batch"]),
    )[-10:]

    return {
        "num_component_reward_rows": len(rows),
        "num_numeric_component_reward_rows": len(numeric_rows),
        "num_response_reward_rows": len(grouped),
        "component_metrics": component_metrics,
        "total_reward_metrics": total_reward_metrics,
        "recent_response_rewards": recent_totals,
    }

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _merge_log_rows(*sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge log rows while removing exact duplicates."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rows in sources:
        for row in rows:
            key = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def _summarize_reward_kl(log_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = sorted(
        {
            key
            for row in log_rows
            for key in row.keys()
            if ("reward" in key.lower()) or ("kl" in key.lower())
        }
    )

    rows_with_reward_or_kl = [
        row for row in log_rows if any(key in row for key in metric_keys)
    ]

    metrics: dict[str, dict[str, float | int]] = {}
    for key in metric_keys:
        values = [
            float(row[key])
            for row in log_rows
            if key in row and isinstance(row[key], (int, float))
        ]
        if values:
            metrics[key] = {
                "count": len(values),
                "last": values[-1],
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }

    return {
        "num_log_rows_total": len(log_rows),
        "num_log_rows_with_reward_or_kl": len(rows_with_reward_or_kl),
        "metric_keys": metric_keys,
        "metrics": metrics,
        "recent_rows": rows_with_reward_or_kl[-10:],
    }


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
