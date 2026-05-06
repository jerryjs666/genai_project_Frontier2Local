from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from transformers import TrainerCallback

ROOT = Path(__file__).resolve().parents[1]
COMMON_SRC = ROOT / "RL_common" / "src"
if COMMON_SRC.exists() and str(COMMON_SRC) not in sys.path:
    sys.path.insert(0, str(COMMON_SRC))

from rl_common.config import format_run_name, load_config, make_run_dir, save_yaml
from rl_common.data import build_trl_dataset, load_examples
from rl_common.eval import run_greedy_eval
from rl_common.model import load_policy_model, load_tokenizer, save_adapter_or_model
from rl_common.rewards import (
    make_answer_reward_func,
    make_format_reward_func,
    make_penalty_reward_func,
    score_completion,
)


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


CORE_GRPO_LOG_KEYS = {
    "loss",
    "grad_norm",
    "learning_rate",
    "epoch",
    "reward",
    "reward_std",
    "frac_reward_zero_std",
    "rewards/answer_reward/mean",
    "rewards/answer_reward/std",
    "rewards/format_reward/mean",
    "rewards/penalty_reward/mean",
    "rewards/soft_overlong_punishment/mean",
    "rewards/soft_overlong_punishment/std",
    "answer_exact_match",
    "format_exact_rate",
    "parse_fail_rate",
    "length_penalty_rate",
    "completions/mean_length",
    "completions/clipped_ratio",
    "entropy",
    "kl",
    "clip_ratio/region_mean",
    "sampling/sampling_logp_difference/mean",
    "sampling/importance_sampling_ratio/mean",
}

VALIDATION_METRIC_PREFIX = "validation"


CORE_VALIDATION_LOG_KEYS = {
    "exact_match",
    "reward_mean",
    "reward_std",
    "parse_success_rate",
    "format_exact_rate",
    "length_penalty_rate",
    "avg_output_tokens",
    "avg_output_tokens_correct",
    "avg_output_tokens_wrong",
    "best_metric",
    "best_exact_match",
    "best_step",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GSM8K GRPO/GSPO/DAPO run with TRL.")
    parser.add_argument("--config", required=True, help="Path to RL YAML config.")
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


class EosCompatibleTokenizer:
    def __init__(self, tokenizer: Any, eos_token_ids: list[int]) -> None:
        self._tokenizer = tokenizer
        self._eos_token_ids = eos_token_ids

    @property
    def eos_token_id(self) -> list[int]:
        return self._eos_token_ids

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)


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
        result = _run_enriched_greedy_eval(
            model=model,
            tokenizer=self.tokenizer,
            examples=self.eval_examples,
            prompt_cfg=self.prompt_cfg,
            eval_cfg={**self.eval_cfg, "desc": f"Greedy eval step {step}"},
        )
        summary = result["summary"]
        summary["step"] = step

        metric = float(summary[self.metric_for_best])
        if self.best_metric is None or metric > self.best_metric:
            self.best_metric = metric
            self.best_step = step
            final_adapter = self.run_dir / "final_adapter"
            if final_adapter.exists():
                shutil.rmtree(final_adapter)
            save_adapter_or_model(model, final_adapter)
            summary["best_metric"] = self.best_metric
            summary[f"best_{self.metric_for_best}"] = self.best_metric
            summary["best_step"] = self.best_step
            _write_json(self.run_dir / "best_eval_results.json", result)
        else:
            summary["best_metric"] = self.best_metric
            summary[f"best_{self.metric_for_best}"] = self.best_metric
            summary["best_step"] = self.best_step
        _write_json(self.run_dir / "latest_eval_results.json", result)
        _try_swanlab_log(summary, step)
        return result


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

    model_cfg = config["model"]
    prompt_cfg = {
        "template": model_cfg.get("prompt_template", "qwen_chat"),
        "system_prompt": model_cfg.get("system_prompt", ""),
        "include_empty_system": model_cfg.get("include_empty_system", False),
    }
    reward_cfg = config["reward"]

    tokenizer = load_tokenizer(model_cfg)
    _ensure_qwen_chat_template(tokenizer, model_cfg)
    eos_token_ids = _resolve_eos_token_ids(tokenizer, model_cfg)
    _inject_eos_generation_kwargs(config["grpo"], eos_token_ids)
    save_yaml(run_dir / "resolved_config.yaml", config)

    eval_tokenizer = EosCompatibleTokenizer(tokenizer, eos_token_ids)
    model = load_policy_model(model_cfg, is_trainable_adapter=not args.eval_only)

    if args.eval_only:
        if not model_cfg.get("device_map") and torch.cuda.is_available():
            model.to("cuda")
        print(f"eval device = {next(model.parameters()).device}")
        dataset_cfg = config["final_eval_dataset"] if args.final_test else config["eval_dataset"]
        examples = load_examples(dataset_cfg)
        result = _run_enriched_greedy_eval(
            model=model,
            tokenizer=eval_tokenizer,
            examples=examples,
            prompt_cfg=prompt_cfg,
            eval_cfg={**config["eval"], "reward": reward_cfg},
        )
        output_name = "final_test_eval_results.json" if args.final_test else "eval_only_results.json"
        _write_json(run_dir / output_name, result)
        print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
        return

    from trl import GRPOConfig, GRPOTrainer

    CompactGRPOTrainer = _make_compact_grpo_trainer_class(GRPOTrainer)

    train_examples = load_examples(config["train_dataset"])
    eval_examples = load_examples(config["eval_dataset"])
    train_dataset = build_trl_dataset(train_examples, prompt_cfg)

    reward_funcs = _build_reward_funcs(reward_cfg, config["grpo"])
    grpo_args = _build_grpo_config(GRPOConfig, config["grpo"], run_dir)
    callback = GreedyEvalAndBestSaveCallback(
        tokenizer=eval_tokenizer,
        eval_examples=eval_examples,
        prompt_cfg=prompt_cfg,
        eval_cfg={**config["eval"], "reward": reward_cfg},
        run_dir=run_dir,
    )

    trainer = CompactGRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[callback],
    )
    train_output = trainer.train()

    if callback.best_metric is None:
        callback.evaluate_and_save_if_best(trainer.model, trainer.state.global_step)

    train_summary = {
        "global_step": trainer.state.global_step,
        "train_metrics": train_output.metrics,
        "best_step": callback.best_step,
        "best_metric": callback.best_metric,
        "best_metric_name": f"{VALIDATION_METRIC_PREFIX}/{callback.metric_for_best}",
    }
    _write_json(run_dir / "train_summary.json", train_summary)
    print(json.dumps(train_summary, indent=2, ensure_ascii=False))


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


def _inject_eos_generation_kwargs(grpo_cfg: dict[str, Any], eos_token_ids: list[int]) -> None:
    generation_kwargs = grpo_cfg.setdefault("generation_kwargs", {})
    if generation_kwargs is None:
        generation_kwargs = {}
        grpo_cfg["generation_kwargs"] = generation_kwargs
    if not isinstance(generation_kwargs, dict):
        raise TypeError("grpo.generation_kwargs must be a mapping when set")

    # TRL passes generation_kwargs directly into vLLM SamplingParams when
    # use_vllm=true. vLLM SamplingParams does not accept eos_token_id, so
    # use stop_token_ids for Qwen chat/EOS stopping instead.
    configured_eos = generation_kwargs.pop("eos_token_id", None)
    expected_eos = eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0]
    if configured_eos is not None and configured_eos != expected_eos:
        raise ValueError(
            "grpo.generation_kwargs.eos_token_id conflicts with model.eos_tokens. "
            f"Expected {expected_eos!r}, got {configured_eos!r}."
        )

    configured_stop_ids = generation_kwargs.get("stop_token_ids")
    if configured_stop_ids is not None:
        if isinstance(configured_stop_ids, int):
            configured_stop_ids = [configured_stop_ids]
        if not isinstance(configured_stop_ids, list) or not all(
            isinstance(token_id, int) for token_id in configured_stop_ids
        ):
            raise TypeError("grpo.generation_kwargs.stop_token_ids must be an int or a list of ints")
        generation_kwargs["stop_token_ids"] = list(dict.fromkeys(configured_stop_ids + eos_token_ids))
    else:
        generation_kwargs["stop_token_ids"] = eos_token_ids

    # Also stop if the decoded chat template marker appears as text. This is
    # accepted by vLLM and is harmless for non-vLLM generation.
    generation_kwargs.setdefault("stop", ["<|im_end|>"])


def _run_enriched_greedy_eval(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[Any],
    prompt_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    result = run_greedy_eval(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        prompt_cfg=prompt_cfg,
        eval_cfg=eval_cfg,
    )
    _enrich_eval_result(result, eval_cfg.get("reward", {}))
    return result


def _enrich_eval_result(result: dict[str, Any], reward_cfg: dict[str, Any]) -> None:
    rewards = []
    answer_rewards = []
    format_rewards = []
    penalties = []
    correct_tokens = []
    wrong_tokens = []
    format_success = 0
    length_penalty_count = 0

    for row in result["results"]:
        token_count = int(row.get("completion_tokens", 0))
        score = score_completion(row.get("completion", ""), row.get("gold_answer", ""), reward_cfg, token_count)
        min_completion_tokens = reward_cfg.get("min_completion_tokens")
        max_completion_tokens = reward_cfg.get("max_completion_tokens")
        too_short = min_completion_tokens is not None and token_count < int(min_completion_tokens)
        too_long = max_completion_tokens is not None and token_count > int(max_completion_tokens)

        row["answer_reward"] = score["answer_reward"]
        row["format_reward"] = score["format_reward"]
        row["penalty"] = score["penalty"]
        row["reward"] = score["reward"]

        format_success += int(float(score["format_reward"]) > 0.0)
        length_penalty_count += int(too_short or too_long)
        rewards.append(float(score["reward"]))
        answer_rewards.append(float(score["answer_reward"]))
        format_rewards.append(float(score["format_reward"]))
        penalties.append(float(score["penalty"]))
        if row.get("correct"):
            correct_tokens.append(token_count)
        else:
            wrong_tokens.append(token_count)

    total = int(result["summary"].get("total", len(result["results"])))
    result["summary"].update(
        {
            "format_exact_rate": round(format_success / total, 6) if total else 0.0,
            "length_penalty_rate": round(length_penalty_count / total, 6) if total else 0.0,
            "reward_mean": round(mean(rewards), 6) if rewards else 0.0,
            "reward_std": round(pstdev(rewards), 6) if len(rewards) > 1 else 0.0,
            "answer_reward_mean": round(mean(answer_rewards), 6) if answer_rewards else 0.0,
            "format_reward_mean": round(mean(format_rewards), 6) if format_rewards else 0.0,
            "penalty_mean": round(mean(penalties), 6) if penalties else 0.0,
            "avg_output_tokens_correct": round(mean(correct_tokens), 6) if correct_tokens else 0.0,
            "avg_output_tokens_wrong": round(mean(wrong_tokens), 6) if wrong_tokens else 0.0,
        }
    )


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


def _build_reward_funcs(reward_cfg: dict[str, Any], grpo_cfg: dict[str, Any]) -> list[Any]:
    reward_funcs = [
        make_answer_reward_func(
            correct_reward=float(reward_cfg.get("answer_correct", 1.0)),
            incorrect_reward=float(reward_cfg.get("answer_incorrect", 0.0)),
        ),
        make_format_reward_func(float(reward_cfg.get("format_reward", 0.2))),
        make_penalty_reward_func(
            parse_fail_penalty=float(reward_cfg.get("parse_fail_penalty", -0.1)),
            length_penalty=float(reward_cfg.get("length_penalty", -0.05)),
            min_completion_tokens=reward_cfg.get("min_completion_tokens"),
            max_completion_tokens=reward_cfg.get("max_completion_tokens"),
        ),
    ]

    soft_overlong_cfg = reward_cfg.get("soft_overlong_punishment")
    if soft_overlong_cfg is None:
        return reward_funcs
    if not isinstance(soft_overlong_cfg, dict):
        raise TypeError("reward.soft_overlong_punishment must be a mapping when set")
    if not bool(soft_overlong_cfg.get("enabled", True)):
        return reward_funcs

    max_completion_len = int(
        soft_overlong_cfg.get(
            "max_completion_len",
            grpo_cfg.get("max_completion_length"),
        )
    )
    soft_punish_cache = int(soft_overlong_cfg["soft_punish_cache"])
    if soft_punish_cache <= 0 or soft_punish_cache >= max_completion_len:
        raise ValueError("soft_punish_cache must be positive and smaller than max_completion_len")

    try:
        from trl.rewards import get_soft_overlong_punishment
    except ImportError as exc:
        raise RuntimeError("The installed TRL package does not provide get_soft_overlong_punishment") from exc

    soft_overlong_reward = get_soft_overlong_punishment(
        max_completion_len=max_completion_len,
        soft_punish_cache=soft_punish_cache,
    )
    soft_overlong_reward.__name__ = "soft_overlong_punishment"
    reward_funcs.append(soft_overlong_reward)
    return reward_funcs


def _make_compact_grpo_trainer_class(base_cls: Any):
    class CompactGRPOTrainer(base_cls):
        def log(self, logs: dict[str, float], *args, **kwargs):
            mode = "train" if self.model.training else "eval"
            if hasattr(self, "_metrics") and mode in self._metrics:
                self._metrics[mode] = defaultdict(
                    list,
                    {
                        key: value
                        for key, value in self._metrics[mode].items()
                        if _is_core_grpo_log_key(key)
                    },
                )
            return super().log(_compact_grpo_logs(logs), *args, **kwargs)

    return CompactGRPOTrainer


def _compact_grpo_logs(logs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in logs.items() if _is_core_grpo_log_key(key)}


def _is_core_grpo_log_key(key: str) -> bool:
    normalized = key.replace("reward/", "rewards/", 1)
    return normalized in CORE_GRPO_LOG_KEYS


def _try_swanlab_log(summary: dict[str, Any], step: int) -> None:
    try:
        import swanlab

        swanlab.log(
            {
                f"{VALIDATION_METRIC_PREFIX}/{key}": value
                for key, value in summary.items()
                if key in CORE_VALIDATION_LOG_KEYS and isinstance(value, (int, float))
            },
            step=step,
        )
    except Exception:
        return


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
