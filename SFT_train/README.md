# SFT_train

This folder contains the supervised fine-tuning (SFT) setup and results for training a LoRA adapter on GSM8K-style math reasoning data distilled from the Qwen3.5-397B teacher model.

## What Is Inside

- `qwen25_3b_base_llamafactory_lora_sft_colab_train.ipynb`: Colab notebook for the **base model** training run (recommended — this is the checkpoint used by all RL stages).
- `qwen25_3b_llamafactory_lora_sft_colab_train.ipynb`: Colab notebook for the Instruct model SFT run (experimental, not used in the main pipeline).
- `train_qwen25_3b_lora_sft.yaml`: LLaMA-Factory training configuration.
- `data/`: Training and validation data in LLaMA-Factory dataset format.
- `outputs/qwen25_3b_base_gsm8k_lora_sft_full/`: Final LoRA adapter, tokenizer files, logs, plots, and training metrics for the base model run.
- `outputs/qwen25_3b_gsm8k_lora_sft_full/`: Outputs for the Instruct model run (experimental).

## Model And Method

- Base model: `Qwen/Qwen2.5-3B` (**base model**, not Instruct)
- Fine-tuning method: LoRA via PEFT
- Training framework: LLaMA-Factory / Transformers
- Task: GSM8K math problem solving via chain-of-thought distillation
- Hardware: Google Colab A100, ~40 minutes

> **Note:** The student is the base model (`Qwen/Qwen2.5-3B`), not the Instruct variant. Using the base model avoids conflicts between the Instruct model's existing RLHF alignment and the custom chat format injected during SFT. All downstream RL stages depend on the base model SFT adapter at `outputs/qwen25_3b_base_gsm8k_lora_sft_full`.

## LoRA Configuration

Defined in `train_qwen25_3b_lora_sft.yaml`:

| Parameter | Value |
|---|---|
| LoRA rank | 32 |
| LoRA alpha | 64 (scale = 2.0) |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj |
| Epochs | 4 |
| Learning rate | 2e-5 |
| LR scheduler | cosine |
| Batch size per device | 16 |
| Optimizer | adamw_torch (bf16) |
| Max sequence length | 512 tokens |

## Data

The `data/` folder contains three dataset files:

- `gsm8k_sft_all.json`: full SFT dataset (7,256 verified teacher traces)
- `gsm8k_sft_train.json`: training split
- `gsm8k_sft_val.json`: validation split

`dataset_info.json` maps these files into the format expected by LLaMA-Factory:

- `instruction` → system prompt
- `input` → question
- `output` → teacher reasoning + answer

## Training

Open the recommended notebook in Colab:

```text
qwen25_3b_base_llamafactory_lora_sft_colab_train.ipynb
```

The notebook installs LLaMA-Factory and its dependencies, mounts Drive, and launches training with:

```text
train_qwen25_3b_lora_sft.yaml
```

The config assumes the project is placed at:

```text
/content/project/SFT_train
```

If the folder is moved, update `dataset_dir` and `output_dir` in the YAML before training.

## Results

The final adapter for the base model run is at:

```text
outputs/qwen25_3b_base_gsm8k_lora_sft_full/
```

Key training metrics:

| Metric | Value |
|---|---|
| Final eval loss | 0.2762 |
| Train loss | 0.2601 |
| Runtime | ~2,420 seconds |
| Epochs | 4 |
| GSM8K test accuracy | 72.78% (+6.29 pp over base) |

Useful output files:

- `adapter_model.safetensors`: trained LoRA adapter weights
- `adapter_config.json`: LoRA adapter configuration
- `trainer_log.jsonl`: step-by-step training log
- `training_loss.png` / `training_eval_loss.png`: loss curves
- `all_results.json`, `train_results.json`, `eval_results.json`: summary metrics

## Notes

- Use `qwen25_3b_base_llamafactory_lora_sft_colab_train.ipynb` for the main pipeline.
- The output is a LoRA adapter, not a merged model. Load it with PEFT on top of `Qwen/Qwen2.5-3B`.
- Evaluation results for both base and SFT models are in `evaluation/results/`.
