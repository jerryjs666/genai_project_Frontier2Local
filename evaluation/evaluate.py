"""
evaluate.py
-----------
Core evaluation logic for GSM8K.
Imported by run_eval_colab.ipynb — do not run directly.
"""

import json
import time
import traceback
from pathlib import Path

import torch
from tqdm.auto import tqdm

from answer_extractor import extract_answer, official_gold, is_correct


SYSTEM_PROMPT = ''


def build_messages(question: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]


def run_evaluation(
    model,
    tokenizer,
    dataset,
    device,
    batch_size: int = 64,
    max_new_tokens: int = 512,
    desc: str = "Evaluating",
):
    """
    Run greedy-decoding evaluation on a HuggingFace dataset.

    Parameters
    ----------
    model        : loaded model (base or PeftModel)
    tokenizer    : corresponding tokenizer
    dataset      : HuggingFace dataset with 'question' and 'answer' columns
    device       : torch device
    batch_size   : number of examples per generation batch
    max_new_tokens: max tokens to generate per example
    desc         : tqdm progress bar label

    Returns
    -------
    summary : dict  — accuracy + metadata
    results : list  — per-example records
    """
    results = []
    correct = 0
    errors  = 0
    start_all = time.perf_counter()

    for start_idx in tqdm(range(0, len(dataset), batch_size), desc=desc):
        batch      = dataset[start_idx : start_idx + batch_size]
        questions  = batch["question"]
        golds      = [official_gold(a) for a in batch["answer"]]
        msgs_batch = [build_messages(q) for q in questions]

        try:
            tokenized = tokenizer.apply_chat_template(
                msgs_batch,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
            )
            input_ids      = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            prompt_lengths = attention_mask.sum(dim=1).tolist()

            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            latency = time.perf_counter() - t0

            for i in range(len(questions)):
                new_tokens = output_ids[i][int(prompt_lengths[i]):]
                response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                predicted  = extract_answer(response)
                ok         = is_correct(predicted, golds[i])
                if ok:
                    correct += 1
                results.append({
                    "question":      questions[i],
                    "gold":          golds[i],
                    "predicted":     predicted,
                    "correct":       ok,
                    "response":      response,
                    "prompt_tokens": int(prompt_lengths[i]),
                    "latency_s":     round(latency / len(questions), 3),
                })

        except Exception:
            error_text = traceback.format_exc()
            errors += len(questions)
            for i in range(len(questions)):
                results.append({
                    "question":      questions[i],
                    "gold":          golds[i],
                    "predicted":     None,
                    "correct":       False,
                    "response":      error_text,
                    "prompt_tokens": 0,
                    "latency_s":     0.0,
                })

    total = len(dataset)
    summary = {
        "total":            total,
        "correct":          correct,
        "errors":           errors,
        "accuracy":         round(correct / total * 100, 2),
        "total_latency_s":  round(time.perf_counter() - start_all, 2),
        "eval_batch_size":  batch_size,
        "max_new_tokens":   max_new_tokens,
    }
    return summary, results


def save_results(summary: dict, results: list, output_path: Path):
    """Save summary + per-example results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"summary": summary, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved → {output_path}")


def print_summary(label: str, summary: dict):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Total   : {summary['total']}")
    print(f"  Correct : {summary['correct']}")
    print(f"  Errors  : {summary['errors']}")
    print(f"  Accuracy: {summary['accuracy']:.2f}%")
    print(f"  Time    : {summary['total_latency_s']:.1f}s")
    print(f"{'='*50}")


def print_wrong_examples(results: list, n: int = 5):
    wrong = [r for r in results if not r["correct"]]
    print(f"\nFirst {n} wrong examples ({len(wrong)} total):")
    for row in wrong[:n]:
        print("-" * 80)
        print("gold     :", row["gold"])
        print("predicted:", row["predicted"])
        print("response :", row["response"][:300])
