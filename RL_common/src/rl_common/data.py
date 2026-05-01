from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from .answers import extract_answer, extract_gsm8k_gold_answer
from .config import resolve_path
from .prompts import build_prompt


@dataclass(frozen=True)
class GSM8KExample:
    id: str
    question: str
    gold_answer: str


def make_example_id(split: str, index: int) -> str:
    return f"gsm8k_{split}_{index:05d}"


def load_examples(dataset_cfg: dict[str, Any]) -> list[GSM8KExample]:
    kind = dataset_cfg.get("kind", "gsm8k_train_subset")
    if kind == "sft_val":
        return _load_sft_id_file(dataset_cfg, "SFT_train/data/gsm8k_sft_val.json")
    if kind == "sft_train":
        return _load_sft_id_file(dataset_cfg, "SFT_train/data/gsm8k_sft_train.json")
    if kind == "gsm8k_train_subset":
        return _load_gsm8k_split("train", dataset_cfg)
    if kind == "gsm8k_test":
        return _load_gsm8k_split("test", dataset_cfg)
    raise ValueError(f"Unsupported dataset.kind: {kind}")


def build_trl_dataset(examples: list[GSM8KExample], prompt_cfg: dict[str, Any]) -> Dataset:
    rows = []
    for example in examples:
        rows.append(
            {
                "id": example.id,
                "prompt": build_prompt(example.question, prompt_cfg),
                "question": example.question,
                "gold_answer": example.gold_answer,
            }
        )
    return Dataset.from_list(rows)


def examples_to_dicts(examples: list[GSM8KExample]) -> list[dict[str, str]]:
    return [asdict(example) for example in examples]


def _load_gsm8k_split(split: str, dataset_cfg: dict[str, Any]) -> list[GSM8KExample]:
    dataset = load_dataset(
        dataset_cfg.get("name", "openai/gsm8k"),
        dataset_cfg.get("config", "main"),
        split=split,
    )
    start_index = int(dataset_cfg.get("start_index") or 0)
    limit = dataset_cfg.get("limit")
    end_index = len(dataset) if limit is None else min(start_index + int(limit), len(dataset))
    if start_index < 0 or start_index > len(dataset):
        raise ValueError(f"Invalid start_index={start_index} for GSM8K {split} size {len(dataset)}")

    examples: list[GSM8KExample] = []
    for idx in range(start_index, end_index):
        row = dataset[idx]
        examples.append(
            GSM8KExample(
                id=make_example_id(split, idx),
                question=row["question"],
                gold_answer=extract_gsm8k_gold_answer(row["answer"]),
            )
        )
    return examples


def _load_sft_id_file(dataset_cfg: dict[str, Any], default_path: str) -> list[GSM8KExample]:
    sft_path = resolve_path(dataset_cfg.get("path", default_path))
    assert sft_path is not None
    records = json.loads(Path(sft_path).read_text(encoding="utf-8"))
    limit = dataset_cfg.get("limit")
    if limit is not None:
        records = records[: int(limit)]

    gsm8k_train = load_dataset(
        dataset_cfg.get("name", "openai/gsm8k"),
        dataset_cfg.get("config", "main"),
        split="train",
    )

    examples: list[GSM8KExample] = []
    for record in records:
        record_id = record["id"]
        idx = int(record_id.rsplit("_", 1)[1])
        official_row = gsm8k_train[idx]
        gold = extract_gsm8k_gold_answer(official_row["answer"])
        question = record.get("instruction") or official_row["question"]
        fallback_gold = extract_answer(record.get("output", ""))
        examples.append(GSM8KExample(id=record_id, question=question, gold_answer=gold or fallback_gold or ""))
    return examples
