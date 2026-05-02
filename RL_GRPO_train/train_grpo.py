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
from rl_common.rewards import make_answer_reward_func, make_format_reward_func, make_penalty_reward_func


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

    grpo_args = _build_grpo_config(GRPOConfig, config["grpo"], run_dir)
    callback = GreedyEvalAndBestSaveCallback(
        tokenizer=tokenizer,
        eval_examples=eval_examples,
        prompt_cfg=prompt_cfg,
        eval_cfg={**config["eval"], "reward": reward_cfg},
        run_dir=run_dir,
    )

    trainer = GRPOTrainer(
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


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
