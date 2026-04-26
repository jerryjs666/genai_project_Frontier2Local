from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - exercised in real runs, not unit tests
    AsyncOpenAI = None


DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "teacher_system_prompt.txt"


@dataclass
class ProviderResult:
    reasoning: str
    content: str
    usage: dict[str, int] | None
    finish_reason: str | None
    latency_sec: float
    attempt_count: int
    error: dict[str, Any] | None


class FatalProviderError(RuntimeError):
    pass


class TeacherProvider:
    def __init__(self, config: dict[str, Any]) -> None:
        teacher_cfg = config["teacher"]
        api_key = _resolve_api_key(teacher_cfg)
        if AsyncOpenAI is None:
            raise RuntimeError("The `openai` package is required. Run `pip install -e .` first.")

        self.config = config
        self.client = AsyncOpenAI(api_key=api_key, base_url=teacher_cfg["base_url"])
        self.system_prompt = _load_system_prompt(teacher_cfg)

    async def generate(self, question: str) -> ProviderResult:
        teacher_cfg = self.config["teacher"]
        max_retries = int(teacher_cfg["max_retries"])
        backoff_base = float(teacher_cfg["backoff_base_sec"])

        last_error: dict[str, Any] | None = None
        for attempt in range(max_retries + 1):
            try:
                return await self._generate_once(question, attempt + 1)
            except FatalProviderError:
                raise
            except Exception as exc:
                last_error = _normalize_error(exc)
                if _is_fatal_error(last_error):
                    raise FatalProviderError(last_error["message"])
                if attempt >= max_retries:
                    break
                await asyncio.sleep(backoff_base * (2 ** attempt))

        return ProviderResult(
            reasoning="",
            content="",
            usage=None,
            finish_reason=None,
            latency_sec=0.0,
            attempt_count=max_retries + 1,
            error=last_error or {"kind": "unknown_error", "message": "Provider call failed"},
        )

    async def _generate_once(self, question: str, attempt_count: int) -> ProviderResult:
        teacher_cfg = self.config["teacher"]
        start = asyncio.get_running_loop().time()
        request_kwargs: dict[str, Any] = {
            "model": teacher_cfg["model"],
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "timeout": float(teacher_cfg["timeout_sec"]),
        }
        if bool(teacher_cfg.get("use_thinking", False)):
            request_kwargs["extra_body"] = {"enable_thinking": True}
        else:
            request_kwargs["response_format"] = {"type": "json_object"}

        stream = await self.client.chat.completions.create(**request_kwargs)

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        usage: dict[str, int] | None = None
        finish_reason: str | None = None

        async for chunk in stream:
            chunk_usage = _extract_usage(chunk)
            if chunk_usage is not None:
                usage = chunk_usage

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                reasoning_piece = getattr(delta, "reasoning_content", None)
                content_piece = getattr(delta, "content", None)
                if reasoning_piece:
                    reasoning_parts.append(reasoning_piece)
                if content_piece:
                    content_parts.append(content_piece)
            finish_reason = getattr(choice, "finish_reason", finish_reason)

        latency = asyncio.get_running_loop().time() - start
        return ProviderResult(
            reasoning="".join(reasoning_parts).strip(),
            content="".join(content_parts).strip(),
            usage=usage,
            finish_reason=finish_reason,
            latency_sec=latency,
            attempt_count=attempt_count,
            error=None,
        )


def _extract_usage(chunk: Any) -> dict[str, int] | None:
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def _resolve_api_key(teacher_cfg: dict[str, Any]) -> str:
    direct_key = str(teacher_cfg.get("api_key") or "").strip()
    if direct_key:
        return direct_key

    api_key_env = str(teacher_cfg.get("api_key_env") or "").strip()
    if api_key_env.startswith("sk-"):
        return api_key_env

    if api_key_env:
        api_key = os.getenv(api_key_env)
        if api_key:
            return api_key
        raise RuntimeError(
            f"Missing API key. Set environment variable `{api_key_env}` or fill `teacher.api_key` in the config."
        )

    raise RuntimeError("Missing API key. Fill `teacher.api_key` or set `teacher.api_key_env`.")


def _load_system_prompt(teacher_cfg: dict[str, Any]) -> str:
    prompt_path_value = str(teacher_cfg.get("system_prompt_path") or "").strip()
    prompt_path = Path(prompt_path_value) if prompt_path_value else DEFAULT_PROMPT_PATH
    if not prompt_path.is_absolute():
        prompt_path = Path.cwd() / prompt_path
    if not prompt_path.exists():
        raise RuntimeError(f"System prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def _normalize_error(exc: Exception) -> dict[str, Any]:
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    code = getattr(exc, "code", None)
    message = str(exc)
    kind = "api_error"
    lower = message.lower()
    if "timeout" in lower:
        kind = "timeout"
    elif "429" in lower or "rate limit" in lower:
        kind = "rate_limit"
    elif "401" in lower or "403" in lower or "unauthorized" in lower or "forbidden" in lower:
        kind = "auth_error"
    elif "quota" in lower or "insufficient" in lower or "余额" in message:
        kind = "quota_error"
    return {"kind": kind, "message": message, "http_status": status, "code": code}


def _is_fatal_error(error: dict[str, Any]) -> bool:
    if error["kind"] in {"auth_error", "quota_error"}:
        return True
    status = error.get("http_status")
    return status in {401, 403}


def build_response_text(reasoning: str, answer: str) -> str:
    return f"{reasoning.strip()}\n\nThe answer is {answer}."


def parse_response_json(content: str) -> tuple[str, str]:
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    if "reasoning" not in payload:
        raise KeyError("Missing reasoning key")
    if "answer" not in payload:
        raise KeyError("Missing answer key")
    reasoning = str(payload["reasoning"]).strip()
    answer = str(payload["answer"]).strip()
    return reasoning, answer
