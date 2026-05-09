# RL_common

Shared utilities for all GSM8K RL post-training stages (rollout diagnostics, GRPO, GSPO, DAPO). Extracted as a standalone installable package so reward functions, evaluation, data loading, and model loading stay consistent across modules.

## Contents

- `answers.py`: answer extraction and normalization. Priority order: `The answer is <number>` → `\boxed{...}` → last numeric expression in the output.
- `data.py`: loads the GSM8K train/test splits, maps official gold answers, and cross-references SFT train/val IDs to prevent data leakage between SFT and RL validation.
- `rewards.py`: reward components — answer correctness reward, format reward, parse/length penalty, and dry-run group diagnostics.
- `eval.py`: general-purpose evaluation loop, usable for both validation and final test evaluation.
- `model.py`: loads a base model, tokenizer, and optional LoRA adapter from a YAML config.
- `prompts.py`: prompt templates and chat format utilities.
- `config.py`: config dataclasses and YAML loading helpers.

## Installation

```bash
cd /content/project   # or the repo root
pip install -e RL_common
```

## Model Configuration

All RL modules configure the model through a YAML block. The default for the main pipeline:

```yaml
model:
  base_model_name_or_path: Qwen/Qwen2.5-3B
  adapter_path: SFT_train/outputs/qwen25_3b_base_gsm8k_lora_sft_full
  tokenizer_name_or_path: SFT_train/outputs/qwen25_3b_base_gsm8k_lora_sft_full
  prompt_template: qwen_chat
  system_prompt: ""
  include_empty_system: false
```

To run without a LoRA adapter (base model only), leave `adapter_path` empty or omit it.

## Tests

```bash
pytest RL_common/tests/
```

