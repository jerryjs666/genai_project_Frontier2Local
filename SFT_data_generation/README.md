# teacher-data-gen

This module generates teacher-model data for GSM8K SFT distillation.

## What It Does

- Reads a configurable slice from `openai/gsm8k`, split `main/train`
- Uses only the `question` field
- Calls Alibaba DashScope model `qwen3.5-397b-a17b`
- Filters generations whose final answer does not match the GSM8K official answer
- Writes `success.jsonl`, `bad.jsonl`, `run_stats.json`, and `run.log`

## Installation

```bash
pip install -e .
```

## API Key Configuration

You can either set the API key directly in the config file or provide it through an environment variable.

```bash
export DASHSCOPE_API_KEY=YOUR_KEY_HERE
```

Corresponding config:

```yaml
teacher:
  api_key_env: DASHSCOPE_API_KEY
```

The system prompt is stored separately in `prompts/teacher_system_prompt.txt`. You can change its location through `teacher.system_prompt_path`.

## Run

```bash
python -m teacher_data_gen.main --config configs/teacher_gsm8k.yaml
```

## Output Format

Successful samples are written to `success.jsonl`. Each record contains:

- `id` // manually assigned because GSM8K does not provide one
- `question`
- `response.reasoning`
- `response.answer`
- `created_at`
- `teacher_model`
- `usage`

Filtered samples or failed generations are written to `bad.jsonl`.
Answer mismatches are filtered with `filter_reason: "answer_mismatch"` and are not included in `success.jsonl`.

`run_stats.json` includes `avg_teacher_cot_tokens`, computed from successful `response.reasoning` fields.

## Notes

- By default, this module does not use the built-in thinking mode of `qwen3.5-397b-a17b`. Instead, it asks the model to return concise `reasoning` and `answer` fields in JSON format. This makes the generated data shorter, cleaner, and easier to parse. In practice, using the original thinking mode often leads to excessively long outputs, and parts of the system prompt may be repeated in the reasoning.
