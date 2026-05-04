# RL_common

这是 GSM8K RL post-training各阶段的共享工具包，例如reward构成、评测、加载数据和模型等公共接口，单独抽象出来以供复用

## 内容

- `answers.py`：答案抽取与归一化。优先 `The answer is <number>`，其次 `\boxed{...}`，最后 fallback 到最后一个数值表达式。
- `data.py`：加载 GSM8K train/test、现有 SFT train/val ID 文件，并映射官方 gold answer，简而言之就是validation与sft时保持一致。
- `rewards.py`：answer reward、format reward、parse/length penalty、dry-run group diagnostics等各种reward组成。
- `eval.py`：通用评估脚本，可用于 validation 或 test。
- `model.py`：从 YAML 参数加载 base model、tokenizer和LoRA adapter。

## Colab 安装

```bash
cd /content/project
pip install -e RL_common
```

模型替换主要通过调用方 YAML 完成，例如：

```yaml
model:
  base_model_name_or_path: Qwen/Qwen2.5-3B-Instruct
  adapter_path: SFT_train/outputs/qwen25_3b_gsm8k_lora_sft_full
  tokenizer_name_or_path: SFT_train/outputs/qwen25_3b_gsm8k_lora_sft_full
  prompt_template: qwen_chat
  system_prompt: ""
  include_empty_system: false
```

如果不用 LoRA adapter，把 `adapter_path` 留空即可。
