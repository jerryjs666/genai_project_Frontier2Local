# evaluation

This module provides standalone GSM8K evaluation for all project checkpoints: base model, SFT adapter, and RL adapters (GRPO, GSPO, DAPO).

## Files

- `evaluate.py`: main evaluation script. Loads a model + optional LoRA adapter, runs greedy decoding on the GSM8K test set, and writes a results JSON.
- `answer_extractor.py`: answer extraction and normalization shared with `RL_common`. Prioritizes `The answer is <number>`, then `\boxed{...}`, then the last numeric expression.
- `config.yaml`: evaluation configuration (model path, adapter path, dataset split, output path).
- `run_eval_colab.ipynb`: Colab notebook wrapper for running evaluations interactively.
- `results/`: saved evaluation outputs, one subfolder per checkpoint.

## Results Summary

| Checkpoint | Split | Accuracy |
|---|---|---|
| `3b_base` | test | 66.49% |
| `3b_base_sft` | test | 72.78% |
| `3b_base_sft_grpo_g8_trainall` | test | 80.67% |
| `3b_base_sft_gspo_g16_trainall` | test | 82.03% |

Full per-question outputs are stored as JSON in the corresponding `results/` subfolder.

## Usage

### Via config file

```bash
pip install -e ../RL_common
python evaluate.py --config config.yaml
```

Edit `config.yaml` to point `model.adapter_path` at the checkpoint you want to evaluate.

### Via Colab

Open `run_eval_colab.ipynb` and set `PROJECT_DIR` to the repo root. The notebook installs dependencies and runs the evaluation with the default config.

## Configuration

`config.yaml` uses the same model config schema as `RL_common`:

```yaml
model:
  base_model_name_or_path: Qwen/Qwen2.5-3B
  adapter_path: ../SFT_train/outputs/qwen25_3b_base_gsm8k_lora_sft_full
  tokenizer_name_or_path: ../SFT_train/outputs/qwen25_3b_base_gsm8k_lora_sft_full
  prompt_template: qwen_chat
  system_prompt: ""
  include_empty_system: false

dataset:
  split: test  # or val

output:
  path: results/my_run/test_results.json
```

## Evaluation Protocol

- Greedy decoding (temperature = 0, no sampling).
- Identical system prompt across all checkpoints.
- Exact-match accuracy on the extracted final numeric answer.
- GSM8K test set: 1,319 held-out questions, zero overlap with the SFT training data.
