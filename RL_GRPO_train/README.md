# RL_GRPO_train

This module runs GSM8K RL training with TRL on top of the LLM project's Qwen2.5-3B base SFT LoRA adapter.

The RL starting point is:

```yaml
model:
  base_model_name_or_path: Qwen/Qwen2.5-3B
  adapter_path: SFT_train/outputs/qwen25_3b_base_gsm8k_lora_sft_full
  tokenizer_name_or_path: SFT_train/outputs/qwen25_3b_base_gsm8k_lora_sft_full
```

Do not replace these with the Instruct SFT adapter unless the experiment is explicitly changed. The base SFT tokenizer uses `<|im_end|>` as the chat stop token and `<|endoftext|>` as the pad token, so the RL configs register both as compatible EOS tokens.

## Files

- `configs/qwen25_3b_base_sft_grpo.yaml`: GRPO baseline.
- `configs/qwen25_3b_base_sft_gspo.yaml`: GSPO-style sequence-level importance sampling.
- `configs/qwen25_3b_base_sft_dapo.yaml`: DAPO-style loss, truncated-completion masking, asymmetric clipping, and soft overlong punishment.
- `train_grpo.py`: shared training and eval entrypoint.
- `colab_grpo_train.ipynb`, `colab_gspo_train.ipynb`, `colab_dapo_train.ipynb`: Colab runners for the three configs.

Training outputs are written to RL_GRPO_train/outputs/{run.name}.

## Colab Setup

Mount Drive, open one of the notebooks, and point `PROJECT_DIR` at the LLM project root. The notebooks look for common locations such as:

```text
/content/drive/MyDrive/LLM_project
/content/drive/MyDrive/LLM_project/project
/content/project
```

Required secrets:

- `HF_TOKEN`, if Hugging Face access requires it.
- `SWANLAB_API_KEY`, for SwanLab logging.

The notebooks install:

```bash
pip install -q -U "trl[vllm]>=0.29.0" "accelerate>=1.4.0" swanlab bitsandbytes sentencepiece
pip install -q -e RL_common
```

## Training

Run one config with:

```bash
accelerate launch --num_processes 1 RL_GRPO_train/train_grpo.py \
  --config RL_GRPO_train/configs/qwen25_3b_base_sft_grpo.yaml
```

Swap the config path for GSPO or DAPO:

```bash
RL_GRPO_train/configs/qwen25_3b_base_sft_gspo.yaml
RL_GRPO_train/configs/qwen25_3b_base_sft_dapo.yaml
```

All three configs keep the HPML RL hyperparameters aligned: 1000 steps, learning rate `2.0e-6`, `num_generations: 16`, generation batch size `512`, gradient accumulation `2`, cosine scheduler, vLLM colocate mode, and seed `42`.

## Algorithm Differences

GRPO is the baseline. It samples multiple completions per prompt and compares rewards inside each prompt group. This avoids a separate value model and uses group-relative advantages.

GSPO keeps the same reward, model, data, and training parameters as GRPO, but adds:

```yaml
grpo:
  loss_type: grpo
  importance_sampling_level: sequence
```

This makes importance sampling operate at the sequence level, so the policy update treats each full completion as the unit of comparison.

DAPO changes the optimization details:

```yaml
grpo:
  loss_type: dapo
  mask_truncated_completions: true
  epsilon: 0.2
  epsilon_high: 0.28
```

The DAPO config intentionally does not set `beta`. In TRL, `loss_type: dapo` selects the DAPO loss, while a nonzero `beta` separately enables reference-model KL regularization. Keeping `beta` unset preserves the KL-free DAPO setup.

It also adds TRL's soft overlong reward:

```yaml
reward:
  soft_overlong_punishment:
    enabled: true
    max_completion_len: 256
    soft_punish_cache: 51
```

This begins penalizing completions after roughly `256 - 51 = 205` tokens and excludes hard-truncated samples from the DAPO loss.

## EOS Compatibility

Each config includes:

```yaml
model:
  eos_tokens:
    - <|im_end|>
    - <|endoftext|>
```

`train_grpo.py` resolves both token IDs at startup and fails fast if either token is missing. It injects the IDs into TRL generation kwargs for training and uses the same EOS set for greedy validation and final test evaluation. This is important because the LLM project uses a base-model SFT adapter with Qwen chat formatting, not the Instruct model defaults.

If the tokenizer has no chat template, `train_grpo.py` injects the same Qwen-style chat template used by the base SFT adapter, with default system text:

```text
You are a helpful assistant.
```

## Metrics And Checkpoints

TRL training metrics are reported to SwanLab via `report_to: swanlab`. The custom greedy validation callback logs compact `validation/*` metrics, including:

- `validation/exact_match`
- `validation/reward_mean`
- `validation/reward_std`
- `validation/format_exact_rate`
- `validation/length_penalty_rate`
- `validation/avg_output_tokens`
- `validation/best_exact_match`

The best adapter on `eval_dataset` is saved to:

```text
RL_GRPO_train/outputs/{run.name}/final_adapter
```

Only the current best adapter is kept.

## Final Test

The default official GSM8K test path is still the notebook eval path:

```bash
python RL_GRPO_train/train_grpo.py \
  --config /content/{algo}_final_eval.yaml \
  --eval-only \
  --final-test
```

The notebook creates that temporary eval config by changing only:

```yaml
model:
  adapter_path: RL_GRPO_train/outputs/{run.name}/final_adapter
```

It keeps `tokenizer_name_or_path` pointing at `SFT_train/outputs/qwen25_3b_base_gsm8k_lora_sft_full`, so tokenizer, chat template, and EOS behavior remain compatible.

The repository-level `evaluation/` module has a separate GSM8K eval path, but it is not the default final RL eval path for these notebooks.
