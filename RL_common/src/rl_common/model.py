from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import resolve_local_path_if_exists, resolve_path


def load_tokenizer(model_cfg: dict[str, Any]):
    tokenizer_path = model_cfg.get("tokenizer_name_or_path") or model_cfg["base_model_name_or_path"]
    tokenizer_path = resolve_local_path_if_exists(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        padding_side=model_cfg.get("padding_side", "left"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_policy_model(model_cfg: dict[str, Any], *, is_trainable_adapter: bool | None = None):
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
    }
    dtype = resolve_torch_dtype(model_cfg.get("dtype", "auto"))
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    device_map = model_cfg.get("device_map")
    if device_map:
        model_kwargs["device_map"] = device_map
    if bool(model_cfg.get("load_in_4bit", False)):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model_name_or_path = resolve_local_path_if_exists(model_cfg["base_model_name_or_path"])
    model = AutoModelForCausalLM.from_pretrained(str(model_name_or_path), **model_kwargs)
    adapter_path = resolve_path(model_cfg.get("adapter_path"))
    if adapter_path is not None:
        trainable = bool(model_cfg.get("is_trainable_adapter", False)) if is_trainable_adapter is None else is_trainable_adapter
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=trainable)
    return model


def resolve_torch_dtype(dtype_name: str | None):
    if dtype_name in (None, "auto"):
        return None
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported model.dtype: {dtype_name}")
    return dtype_map[dtype_name]


def save_adapter_or_model(model: Any, output_dir: str | Path) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
