# RL_dryrun_rollout

This module runs sampling diagnostics before RL training. It does not train the model. It loads the LLM project's Qwen2.5-3B base SFT LoRA adapter, samples multiple completions for GSM8K prompts, and checks whether each prompt has useful reward variance for GRPO-style training.

## Default Model

The default dry-run starts from the same base SFT checkpoint used by `RL_GRPO_train`:

```yaml
model:
  base_model_name_or_path: Qwen/Qwen2.5-3B
  adapter_path: SFT_train/outputs/qwen25_3b_base_gsm8k_lora_sft_full
  tokenizer_name_or_path: SFT_train/outputs/qwen25_3b_base_gsm8k_lora_sft_full
  prompt_template: qwen_chat
  system_prompt: ""
  include_empty_system: false
  eos_tokens:
    - <|im_end|>
    - <|endoftext|>
```

Do not switch this to the Instruct SFT adapter unless the experiment target changes.

## EOS Compatibility

`run_rollout.py` resolves every token listed in `model.eos_tokens` at startup. If any token is missing from the tokenizer vocabulary, it fails immediately with a clear error.

The resolved EOS ids are used for generation and for the `stopped_by_eos` diagnostic. This matters for the base SFT tokenizer because chat turns stop on `<|im_end|>`, while `<|endoftext|>` is also present as the pad token.

If the tokenizer has no chat template, the script injects the Qwen chat template used by the base SFT adapter, with default system text:

```text
You are a helpful assistant.
```

## Run

From the project root:

```bash
python RL_dryrun_rollout/run_rollout.py \
  --config RL_dryrun_rollout/configs/qwen25_3b_sft_dryrun.yaml
```

The default run name is:

```yaml
run:
  name: qwen25_3b_base_sft_dryrun_g{generation.num_generations}_train{dataset.limit}
```

With `generation.num_generations: 16` and `dataset.limit: 200`, outputs go to:

```text
RL_dryrun_rollout/outputs/qwen25_3b_base_sft_dryrun_g16_train200/
```

## Outputs

- `resolved_config.yaml`: the resolved config, including generated EOS ids.
- `rollouts.jsonl`: one line per prompt, with all sampled completions and reward fields.
- `summary.json`: group-level diagnostics such as `mixed_rate`, `all_correct_rate`, `all_wrong_rate`, `parse_fail_rate`, `eos_stop_rate`, length, and reward distribution.
- `sampled_outputs.md`: a small human-readable sample for inspection.

The most important pre-RL diagnostic is usually `mixed_rate`: if too many prompt groups are all correct or all wrong, group-relative RL has little useful reward variance.
