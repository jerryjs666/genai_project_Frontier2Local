from __future__ import annotations

from typing import Any


def build_prompt(question: str, prompt_cfg: dict[str, Any]) -> str | list[dict[str, str]]:
    template = prompt_cfg.get("template", "qwen_chat")
    system_prompt = prompt_cfg.get("system_prompt", "")
    include_empty_system = bool(prompt_cfg.get("include_empty_system", False))
    if template == "plain":
        prefix = prompt_cfg.get("plain_prefix", "")
        suffix = prompt_cfg.get("plain_suffix", "\n")
        return f"{prefix}{question}{suffix}"
    if template == "qwen_chat":
        messages: list[dict[str, str]] = []
        if system_prompt or include_empty_system:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        return messages
    raise ValueError(f"Unsupported prompt.template: {template}")


def render_generation_prompt(tokenizer: Any, question: str, prompt_cfg: dict[str, Any]) -> str:
    prompt = build_prompt(question, prompt_cfg)
    if isinstance(prompt, str):
        return prompt
    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return str(completion or "")
