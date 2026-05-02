# RL_GRPO_train

这个模块负责 TRL GRPO 训练。默认从当前 SFT LoRA adapter 继续训练，但不会修改原始 SFT checkpoint；训练输出写入固定目录 `outputs/{run.name}`。

训练阶段默认验证集是 `SFT_train/data/gsm8k_sft_val.json`，它来自 GSM8K train split 的 held-out subset，和现有 SFT 验证口径一致。官方 GSM8K test 只在最终测试时手动运行，不接入训练过程。

## 目录结构

- `configs/qwen25_3b_sft_grpo.yaml`：默认 GRPO baseline 配置。
- `train_grpo.py`：训练和 eval-only 入口。
- `outputs/{run.name}/final_adapter/`：当前 best adapter，只保留一份。
- `outputs/{run.name}/best_eval_results.json`：best checkpoint 对应的 greedy eval 结果。
- `outputs/{run.name}/resolved_config.yaml`：实际使用的配置。

`run.name` 支持从 YAML 参数自动填充，例如：

```yaml
run:
  name: qwen25_3b_base_sft_grpo_g{grpo.num_generations}_train{train_dataset.limit}
```

如果 `train_dataset.limit` 留空，目录名里的这部分会解析为 `trainall`。

## Colab 准备

在 Colab Secret 中配置：

- `HF_TOKEN`：如果模型或数据需要 Hugging Face token。
- `SWANLAB_API_KEY`：SwanLab 训练记录。

```python
from google.colab import userdata
import os

for key in ["HF_TOKEN", "SWANLAB_API_KEY"]:
    value = userdata.get(key)
    if value:
        os.environ[key] = value
```

安装依赖：

```bash
cd /content/project
pip install -e RL_common
pip install "trl>=0.25.0" swanlab accelerate
```

如果之后要开启 vLLM，再额外安装：

```bash
pip install "trl[vllm]>=0.25.0"
```

## 训练

建议先运行 `RL_dryrun_rollout`，确认 `mixed_rate` 不太低，再启动训练。

```bash
accelerate launch RL_GRPO_train/train_grpo.py \
  --config RL_GRPO_train/configs/qwen25_3b_sft_grpo.yaml
```

训练中自定义 greedy eval 默认每 50 step 跑一次，只评估 `sft_val`。如果 `sft_val/exact_match` 变好，会覆盖保存：

```text
RL_GRPO_train/outputs/qwen25_3b_base_sft_grpo_g4_trainall/final_adapter
```

## 最终测试

训练结束后，如需在官方 GSM8K test 上评估，手动运行：

```bash
python RL_GRPO_train/train_grpo.py \
  --config RL_GRPO_train/configs/qwen25_3b_sft_grpo.yaml \
  --eval-only \
  --final-test
```

注意：不要在训练过程中启用 official test。它只用于最终汇报。

## 更换模型或 adapter

所有关键模型路径都在 YAML：

```yaml
model:
  base_model_name_or_path: Qwen/Qwen2.5-3B
  adapter_path: SFT_train/outputs/qwen25_3b_base_gsm8k_lora_sft_full
  tokenizer_name_or_path: SFT_train/outputs/qwen25_3b_base_gsm8k_lora_sft_full
  prompt_template: qwen_chat
  system_prompt: ""
  include_empty_system: false
```

`include_empty_system: false` 会让 Qwen chat template 自动插入默认 system prompt：

```text
You are a helpful assistant.
```

这与 SFT 训练时的 `template: qwen` 输入格式对齐。

换成另一个 LoRA adapter：只改 `adapter_path`。

从 base 模型直接做 GRPO：把 `adapter_path` 留空，并按模型类型选择 prompt：

```yaml
prompt_template: plain
```

换成 instruct/chat 模型：通常保留：

```yaml
prompt_template: qwen_chat
```

开启 vLLM 时，只改：

```yaml
grpo:
  use_vllm: true
  vllm_mode: colocate
```
