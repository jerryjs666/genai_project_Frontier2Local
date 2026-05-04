# RL_dryrun_rollout

这个模块只做 GRPO 训练前诊断，不训练模型。它会加载一个 student model 和可选 LoRA adapter，对 GSM8K train prompts 做多次采样，检查同一题的多条 completion 是否有正确/错误混合，从而判断是否有有效 reward variance，能否在RL阶段提供有效的梯度

## 目录结构

- `configs/qwen25_3b_sft_dryrun.yaml`：实验配置，跑不同的实验就改这里。
- `run_rollout.py`：采样 rollout 主脚本，绝大多数情况不需要更改
- `outputs/{run.name}/`：固定命名输出目录
`run.name` 从 YAML 中提取实验参数自动填充，例如默认：

```yaml
run:
  name: qwen25_3b_instruct_sft_dryrun_g{generation.num_generations}_train{dataset.limit}
```

如果 `generation.num_generations: 8` 且 `dataset.limit: 500`，输出目录会是：

```text
outputs/qwen25_3b_instruct_sft_dryrun_g8_train500/
```

## Colab 运行

先在 Colab Secret 里配置需要的 token，例如 `HF_TOKEN`。由于这里用的是公开模型，也可以不配置，也能下载模型

```python
from google.colab import userdata
import os

hf_token = userdata.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
```

运行 dry-run：colab中通过run.ipynb运行

输出包括：

- `rollouts.jsonl`：每个 prompt 一行，包含多条采样结果、抽取答案、reward，这部分人读起来比较困难，仅仅是备用分析使用
- `summary.json`：`all_correct_rate`、`all_wrong_rate`、`mixed_rate`、`parse_fail_rate`、平均长度和 reward 分布等各种关键指标。
- `sampled_outputs.md`：人工可读样例。

## 更换模型或 adapter

只改 YAML 的 `model` 区域即可：

```yaml
model:
  base_model_name_or_path: Qwen/Qwen2.5-3B-Instruct
  adapter_path: SFT_train/outputs/qwen25_3b_gsm8k_lora_sft_full
  tokenizer_name_or_path: SFT_train/outputs/qwen25_3b_gsm8k_lora_sft_full
  prompt_template: qwen_chat
  system_prompt: ""
  include_empty_system: false
```

`include_empty_system: false` 会让 Qwen chat template 自动插入默认 system prompt：
```text
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
```

这与 SFT 训练时的 `template: qwen` 输入格式对齐。

如果改成非 chat 模型且不想使用 chat template，设置：

```yaml
prompt_template: plain
```
即可
