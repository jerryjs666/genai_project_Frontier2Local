from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return expand_env_vars(config)


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_path(path_value: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (Path(base_dir) if base_dir is not None else PROJECT_ROOT) / path


def resolve_local_path_if_exists(path_value: str | Path) -> str | Path:
    path = resolve_path(path_value)
    if path is not None and path.exists():
        return path
    return path_value


def format_run_name(run_name: str, config: dict[str, Any]) -> str:
    if not run_name:
        raise ValueError("run.name must be set explicitly; timestamped run names are intentionally not used")

    def replace_match(match: re.Match[str]) -> str:
        key_path = match.group(1)
        value = get_config_value(config, key_path)
        if value in (None, ""):
            return "all"
        return _sanitize_run_name_part(str(value))

    return re.sub(r"\{([A-Za-z0-9_.-]+)\}", replace_match, run_name)


def get_config_value(config: dict[str, Any], key_path: str) -> Any:
    current: Any = config
    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Unknown run.name placeholder: {{{key_path}}}")
        current = current[key]
    return current


def _sanitize_run_name_part(value: str) -> str:
    value = value.strip().replace("/", "-").replace("\\", "-").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "empty"


def make_run_dir(output_root: str | Path, run_name: str) -> Path:
    if not run_name:
        raise ValueError("run.name must be set explicitly; timestamped run names are intentionally not used")
    root = resolve_path(output_root)
    assert root is not None
    return root / run_name


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
