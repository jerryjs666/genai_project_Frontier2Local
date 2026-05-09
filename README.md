# Frontier2Local

**Distilling Reasoning Ability from Qwen3.5-397B into a Compact 3B Student**

> STAT GR5293 · GenAI Final Project · Columbia University, May 2026  
> Shan Jiang · Leyao Chen · Jinbo Li

---

## Overview

Frontier2Local is an end-to-end pipeline for transferring math reasoning ability from a frontier MoE model (Qwen3.5-397B) into a compact, locally deployable student (Qwen2.5-3B) using supervised fine-tuning (SFT) distillation followed by reinforcement learning (RL) alignment. The entire training workflow runs on a single Google Colab A100.

**The core result:** Starting from a Qwen2.5-3B base model at 66.49% GSM8K accuracy, a two-stage SFT + GSPO pipeline reaches **82.03%** — a **+15.54 pp** improvement and near parity with Qwen2.5-7B-Instruct (≈83%) at less than half the parameters.

| Stage | Model | GSM8K Test Accuracy |
|---|---|---|
| Baseline | Qwen2.5-3B (base, prompted) | 66.49% |
| Stage 1 | + SFT on 7,256 teacher traces | 72.78% (+6.3 pp) |
| Stage 2 | + GRPO | 80.67% |
| Stage 2 | + DAPO | 81.12% |
| Stage 2 ★ | + GSPO | **82.03%** |

---

## Research Question

> *Can text-level distillation from a 397B teacher significantly enhance a 3B student's reasoning accuracy on GSM8K?*

The answer is yes. Even without access to teacher logits, structured API prompting with answer-match filtering captures enough reasoning signal to produce strong and measurable accuracy gains — gains attributable to distillation, not prompting (both baseline and student use identical system prompts at eval time).

---

## Repository Structure

```
Frontier2Local/
├── SFT_data_generation/       # Teacher data generation via DashScope API
│   ├── configs/               # YAML configs for the generation job
│   ├── prompts/               # System prompt for the teacher model
│   ├── src/teacher_data_gen/  # Generation, filtering, and utility code
│   │   ├── main.py            # Entry point
│   │   ├── provider.py        # DashScope API provider
│   │   └── utils.py           # Answer verification and output helpers
│   ├── outputs/               # Generated success.jsonl, bad.jsonl, run_stats.json
│   └── tests/                 # Smoke tests
│
├── SFT_train/                 # Supervised fine-tuning (LoRA via LLaMA-Factory)
│   ├── data/                  # GSM8K SFT dataset (train / val / all splits)
│   ├── outputs/               # Trained LoRA adapters and training logs
│   ├── train_qwen25_3b_lora_sft.yaml        # LLaMA-Factory training config
│   ├── qwen25_3b_llamafactory_lora_sft_colab_train.ipynb      # Instruct SFT notebook
│   └── qwen25_3b_base_llamafactory_lora_sft_colab_train.ipynb # Base SFT notebook
│
├── RL_GRPO_train/             # RL post-training (GRPO / GSPO / DAPO)
│   ├── configs/               # Per-algorithm YAML configs
│   │   ├── qwen25_3b_base_sft_grpo.yaml
│   │   ├── qwen25_3b_base_sft_gspo.yaml
│   │   └── qwen25_3b_base_sft_dapo.yaml
│   ├── outputs/               # Best adapters, eval results, training summaries
│   ├── train_grpo.py          # Shared training + eval entrypoint
│   ├── colab_grpo_train.ipynb
│   ├── colab_gspo_train.ipynb
│   └── colab_dapo_train.ipynb
│
├── RL_common/                 # Shared utilities for all RL stages
│   └── src/rl_common/
│       ├── answers.py         # Answer extraction and normalization
│       ├── data.py            # GSM8K dataset loading and gold-answer mapping
│       ├── rewards.py         # Reward functions (answer, format, penalties)
│       ├── eval.py            # Shared evaluation loop
│       ├── model.py           # Model + tokenizer + LoRA loading from YAML
│       └── prompts.py         # Prompt formatting utilities
│
└── evaluation/                # Standalone GSM8K evaluation module
    ├── evaluate.py            # Batched greedy-decoding evaluation loop
    ├── answer_extractor.py    # Answer parsing (matches rl_common logic)
    ├── config.yaml            # Evaluation config (model, splits, output path)
    ├── run_eval_colab.ipynb   # Colab notebook for running evaluation
    └── results/               # Per-model evaluation results (JSON)
```

---

## Pipeline

### Stage 0 — Teacher Data Generation (`SFT_data_generation/`)

The teacher model, `qwen3.5-397b-a17b` (a 397B MoE with 17B active parameters), is accessed via the Alibaba DashScope API. For each of the 7,473 GSM8K training questions, the pipeline:

1. Prompts the teacher in JSON output mode (no built-in thinking mode — avoids excessively long, repetitive outputs).
2. Parses the structured `{ reasoning, answer }` response.
3. Filters against GSM8K ground-truth answers; mismatches go to `bad.jsonl`, matches go to `success.jsonl`.

**Filter statistics:**

| Metric | Value |
|---|---|
| Total API calls | 7,473 |
| Passed filter (success.jsonl) | 7,256 (97.1%) |
| Filtered out (answer mismatch) | 217 (2.9%) |
| Avg tokens per API call | ~1,836 |
| Total tokens consumed | ~13.3M |

**Run:**
```bash
export DASHSCOPE_API_KEY=YOUR_KEY_HERE
python -m teacher_data_gen.main --config configs/teacher_gsm8k.yaml
```

**Output format** (`success.jsonl`):
```json
{
  "id": "<assigned_id>",
  "question": "...",
  "response": { "reasoning": "...", "answer": "..." },
  "created_at": "...",
  "teacher_model": "qwen3.5-397b-a17b",
  "usage": { ... }
}
```

---

### Stage 1 — SFT Distillation (`SFT_train/`)

A LoRA adapter is trained on top of `Qwen/Qwen2.5-3B` (base) using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) with the 7,256 verified teacher traces.

**Training configuration:**

| Hyperparameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B` |
| Method | LoRA (PEFT) |
| LoRA rank / alpha | 32 / 64 |
| LoRA dropout | 0.05 |
| Target modules | attention + MLP projections (`up_proj`, `down_proj`, `gate_proj`) |
| Epochs | 4 |
| Learning rate | 2e-5 (cosine scheduler, warmup ratio 0.03) |
| Batch size | 16 per device |
| Optimizer | AdamW (bf16) |
| Context length | 512 tokens |
| Framework | LLaMA-Factory + SwanLab |

**Training results:**

| Metric | Value |
|---|---|
| Final eval loss | 0.2762 |
| Train loss | 0.2601 |
| Training runtime | ~2,420 s |
| GSM8K test accuracy | **72.78%** (vs. 66.49% base) |

**To reproduce (Colab):**

Open `SFT_train/qwen25_3b_base_llamafactory_lora_sft_colab_train.ipynb`, mount Drive, and point the config's `dataset_dir` and `output_dir` to `/content/project/SFT_train`.

---

### Stage 2 — RL Alignment (`RL_GRPO_train/`)

Three RL algorithms are applied as a second stage on top of the SFT LoRA checkpoint. All use the same reward function, dataset, hyperparameters, and evaluation protocol.

**Shared hyperparameters:**

| Hyperparameter | Value |
|---|---|
| Steps | 1,000 |
| Learning rate | 2.0e-6 |
| `num_generations` | 16 |
| Generation batch size | 512 |
| Gradient accumulation steps | 2 |
| Scheduler | Cosine |
| vLLM mode | Colocate |
| Seed | 42 |

**Algorithm differences:**

**GRPO** (baseline) — Group Relative Policy Optimization. Samples multiple completions per prompt and computes group-relative advantages. No separate value model.

**GSPO** — Adds sequence-level importance sampling (`importance_sampling_level: sequence`), treating each full completion as the unit of comparison. More stable than GRPO and achieves the best final accuracy.

**DAPO** — Asymmetric clipping (`epsilon: 0.2`, `epsilon_high: 0.28`), `mask_truncated_completions: true`, and soft overlong punishment (penalty starts after ~205 tokens). KL-free (no `beta`). Sustains longer chains of thought.

**Training results:**

| Algorithm | Best Val Exact Match | Best Step | Runtime |
|---|---|---|---|
| GRPO | 92.69% | 620 | ~3.95 h |
| GSPO | 91.31% | 1,120 | ~3.96 h |
| DAPO | 91.86% | 600 | ~3.45 h |

**To train:**
```bash
# GRPO
accelerate launch --num_processes 1 RL_GRPO_train/train_grpo.py \
  --config RL_GRPO_train/configs/qwen25_3b_base_sft_grpo.yaml

# GSPO
accelerate launch --num_processes 1 RL_GRPO_train/train_grpo.py \
  --config RL_GRPO_train/configs/qwen25_3b_base_sft_gspo.yaml

# DAPO
accelerate launch --num_processes 1 RL_GRPO_train/train_grpo.py \
  --config RL_GRPO_train/configs/qwen25_3b_base_sft_dapo.yaml
```

Or open the corresponding Colab notebook (`colab_grpo_train.ipynb`, etc.) and point `PROJECT_DIR` at the repo root.

**Required secrets (Colab):**
- `HF_TOKEN` — for gated HuggingFace models, if needed
- `SWANLAB_API_KEY` — for SwanLab experiment tracking

**Install dependencies:**
```bash
pip install -q -U "trl[vllm]>=0.29.0" "accelerate>=1.4.0" swanlab bitsandbytes sentencepiece
pip install -q -e RL_common
```

---

## Reward Function (`RL_common/src/rl_common/rewards.py`)

Each completion is scored by:

| Component | Condition | Value (default) |
|---|---|---|
| Answer reward | Correct final answer | +1.0 |
| Answer reward | Incorrect | 0.0 |
| Format reward | Matches `"The answer is <number>."` | +0.2 |
| Parse penalty | No numeric answer extracted | −0.1 |
| Length penalty | Below `min_tokens` or above `max_tokens` | −0.05 |

Answer extraction uses a three-tier fallback: `"The answer is <number>"` → `\boxed{...}` → last numeric expression in the response.

---

## Evaluation (`evaluation/`)

The standalone evaluation module runs batched greedy-decoding on any base model or LoRA checkpoint against the GSM8K test or train split.

**Configure** `evaluation/config.yaml`:
```yaml
model_id: "Qwen/Qwen2.5-3B-Instruct"
lora_checkpoint: "SFT_train/outputs/qwen25_3b_gsm8k_lora_sft_full/checkpoint-1200"  # or null for base model
dataset_name: "openai/gsm8k"
splits: [test]
max_new_tokens: 512
eval_batch_size: 256
output_dir: "evaluation/results"
output_prefix: "my_run"
```

Results are saved as `evaluation/results/{output_prefix}_{split}.json`, containing a top-level `summary` (accuracy, totals, latency) and a `results` list with per-example predictions.

RL final-test evaluation is handled directly by `train_grpo.py --eval-only --final-test` within the Colab notebooks, using a temporary YAML that points `adapter_path` at the best RL checkpoint.

---

## Shared Utilities (`RL_common/`)

The `rl-common` package is a lightweight Python package used by both RL training and evaluation. Install with:
```bash
pip install -e RL_common
```

| Module | Purpose |
|---|---|
| `answers.py` | Answer extraction with 3-tier fallback; numeric normalization |
| `data.py` | GSM8K train/test loading; SFT split ID alignment |
| `rewards.py` | Scoring functions for RL training |
| `eval.py` | Reusable evaluation loop (validation and test) |
| `model.py` | Load base model + tokenizer + LoRA adapter from YAML config |
| `prompts.py` | Chat template formatting |

Model configuration is controlled entirely through the caller's YAML:
```yaml
model:
  base_model_name_or_path: Qwen/Qwen2.5-3B
  adapter_path: SFT_train/outputs/qwen25_3b_base_gsm8k_lora_sft_full
  tokenizer_name_or_path: SFT_train/outputs/qwen25_3b_base_gsm8k_lora_sft_full
  prompt_template: qwen_chat
  system_prompt: ""
  include_empty_system: false
```

Set `adapter_path` to empty or null to evaluate the base model without any adapter.

---

## EOS Token Compatibility

The SFT adapter is trained on the base model (not the Instruct variant) and uses Qwen chat formatting with `<|im_end|>` as the chat stop token and `<|endoftext|>` as the pad token. All RL configs register both as compatible EOS tokens:

```yaml
model:
  eos_tokens:
    - <|im_end|>
    - <|endoftext|>
```

`train_grpo.py` resolves both token IDs at startup and fails immediately if either is missing. If the tokenizer has no chat template, it injects a Qwen-style template with `"You are a helpful assistant."` as the default system prompt.

---

## Key Findings

- **Prompting alone is already meaningful.** With a standard system prompt, Qwen2.5-3B reaches 66.49% on GSM8K — it can reason out of the box.
- **SFT distillation provides a large clean gain.** Training on 7,256 answer-verified teacher traces raises accuracy to 72.78% (+6.3 pp). The improvement is entirely from better math reasoning: format failures drop from 784 to 1 (−99.9%), and math errors decrease by 65 (−15.3%).
- **RL alignment pushes well past the SFT ceiling.** All three RL algorithms surpass 80%, with GSPO reaching 82.03%.
- **Our 3B model nearly matches Qwen2.5-7B-Instruct (≈83%)** at less than half the parameters.

---

## Limitations and Future Work

- **API cost limits dataset volume.** Scaling to larger datasets or harder benchmarks (AIME, MATH) would require significant API budget.
- **Text-only distillation.** The teacher API does not expose logits, so token-level distillation methods (KL matching, DPO with teacher distributions) are not currently possible.
- Future directions include logit distillation if API access improves, scaling to harder reasoning tasks, and exploring FIPO dense advantage estimation.

---

## Deliverables

| Artifact | Location |
|---|---|
| Teacher-distilled dataset (7,256 traces) | `SFT_data_generation/outputs/runs/20260426_132022/success.jsonl` |
| SFT LoRA adapter (base model) | `SFT_train/outputs/qwen25_3b_base_gsm8k_lora_sft_full/` |
| SFT LoRA adapter (instruct model) | `SFT_train/outputs/qwen25_3b_gsm8k_lora_sft_full/` |
| GRPO best adapter | `RL_GRPO_train/outputs/qwen25_3b_base_sft_grpo_g16_trainall/final_adapter/` |
| GSPO best adapter | `RL_GRPO_train/outputs/qwen25_3b_base_sft_gspo_g16_trainall/final_adapter/` |
| DAPO best adapter | `RL_GRPO_train/outputs/qwen25_3b_base_sft_dapo_g16_trainall/final_adapter/` |
| Per-model evaluation results | `evaluation/results/` |

---

## Citation / Acknowledgements

- Dataset: [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (Cobbe et al., 2021)
- Teacher model: [Qwen3.5-397B-A17B](https://dashscope.console.aliyun.com/) via Alibaba DashScope
- Student model: [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)
- Training frameworks: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [TRL](https://github.com/huggingface/trl)
- Experiment tracking: [SwanLab](https://swanlab.cn/)

All training history of this project can be tracked through this Swanlab link: [Frontier2Local](https://swanlab.cn/@1416079864)
