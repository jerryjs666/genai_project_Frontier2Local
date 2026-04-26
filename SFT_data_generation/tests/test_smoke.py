import asyncio
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from teacher_data_gen.main import run_pipeline
from teacher_data_gen.provider import ProviderResult


class FakeProvider:
    def __init__(self, outputs):
        self.outputs = outputs

    async def generate(self, question: str) -> ProviderResult:
        return self.outputs[question]


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class PipelineSmokeTest(unittest.TestCase):
    def test_pipeline_writes_success_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            logger = logging.getLogger("test_teacher_data_gen")
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())

            config = {
                "dataset": {"limit": 3},
                "teacher": {"model": "qwen3.5-397b-a17b", "max_concurrency": 2},
                "run": {"resume": True},
            }
            records = [
                {"id": "gsm8k_train_00000", "question": "q1"},
                {"id": "gsm8k_train_00001", "question": "q2"},
                {"id": "gsm8k_train_00002", "question": "q3"},
            ]
            provider = FakeProvider(
                {
                    "q1": ProviderResult(
                        reasoning="reason one",
                        content='{"reasoning":"reason one","answer":"1"}',
                        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                        finish_reason="stop",
                        latency_sec=1.0,
                        attempt_count=1,
                        error=None,
                    ),
                    "q2": ProviderResult(
                        reasoning="",
                        content='{"reasoning":"","answer":"2"}',
                        usage=None,
                        finish_reason="stop",
                        latency_sec=2.0,
                        attempt_count=2,
                        error=None,
                    ),
                    "q3": ProviderResult(
                        reasoning="reason three",
                        content="not json",
                        usage=None,
                        finish_reason="stop",
                        latency_sec=3.0,
                        attempt_count=1,
                        error=None,
                    ),
                }
            )

            stats = asyncio.run(run_pipeline(config, records, provider, run_dir, logger))

            success_rows = read_jsonl(run_dir / "success.jsonl")
            bad_rows = read_jsonl(run_dir / "bad.jsonl")
            stats_disk = json.loads((run_dir / "run_stats.json").read_text(encoding="utf-8"))

            self.assertEqual(len(success_rows), 1)
            self.assertNotIn("text", success_rows[0]["response"])
            self.assertEqual(success_rows[0]["response"]["reasoning"], "reason one")
            self.assertEqual(success_rows[0]["response"]["answer"], "1")
            self.assertEqual(len(bad_rows), 2)
            self.assertEqual(stats["success_count"], 1)
            self.assertEqual(stats["filtered_count"], 2)
            self.assertGreater(stats["avg_teacher_cot_tokens"], 0)
            self.assertGreaterEqual(stats_disk["retry_histogram"]["0"], 1)
            self.assertGreaterEqual(stats_disk["retry_histogram"]["1"], 1)

            provider_second = FakeProvider({})
            stats_second = asyncio.run(run_pipeline(config, records, provider_second, run_dir, logger))
            self.assertEqual(stats_second["resume_skipped_count"], 3)
            self.assertEqual(len(read_jsonl(run_dir / "success.jsonl")), 1)
            self.assertEqual(len(read_jsonl(run_dir / "bad.jsonl")), 2)

    def test_pipeline_filters_answer_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            logger = logging.getLogger("test_teacher_data_gen_mismatch")
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())

            config = {
                "dataset": {"limit": 1},
                "teacher": {"model": "qwen3.5-397b-a17b", "max_concurrency": 1},
                "run": {"resume": True},
            }
            records = [
                {
                    "id": "gsm8k_train_00000",
                    "question": "q1",
                    "ground_truth_answer": "2",
                }
            ]
            provider = FakeProvider(
                {
                    "q1": ProviderResult(
                        reasoning="reason one",
                        content='{"reasoning":"reason one","answer":"1"}',
                        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                        finish_reason="stop",
                        latency_sec=1.0,
                        attempt_count=1,
                        error=None,
                    ),
                }
            )

            stats = asyncio.run(run_pipeline(config, records, provider, run_dir, logger))

            self.assertFalse((run_dir / "success.jsonl").exists())
            bad_rows = read_jsonl(run_dir / "bad.jsonl")
            self.assertEqual(len(bad_rows), 1)
            self.assertEqual(bad_rows[0]["filter_reason"], "answer_mismatch")
            self.assertEqual(bad_rows[0]["generated_answer"], "1")
            self.assertEqual(bad_rows[0]["ground_truth_answer"], "2")
            self.assertEqual(stats["success_count"], 0)
            self.assertEqual(stats["filtered_count"], 1)


if __name__ == "__main__":
    unittest.main()
