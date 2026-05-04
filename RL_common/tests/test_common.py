import unittest

from rl_common.answers import answers_match, extract_answer, normalize_answer
from rl_common.config import format_run_name, resolve_local_path_if_exists
from rl_common.prompts import build_prompt
from rl_common.rewards import compute_group_diagnostics, score_completion


class AnswerExtractionTest(unittest.TestCase):
    def test_the_answer_is_priority(self):
        self.assertEqual(extract_answer("Earlier 12. The answer is 62."), "62")

    def test_boxed_answer(self):
        self.assertEqual(extract_answer("So the final value is \\boxed{1,234}."), "1234")

    def test_gsm8k_marker_answer(self):
        self.assertEqual(extract_answer("work\n#### 27"), "27")

    def test_fallback_last_number(self):
        self.assertEqual(extract_answer("2 + 2 = 4, then add 3 to get 7"), "7")

    def test_conclusion_sentence_fallback(self):
        self.assertEqual(extract_answer("So, it takes a total of 3 bolts to make a robe."), "3")

    def test_question_echo_is_not_answer(self):
        text = "If James is 10 and is 1 year younger than Corey, how old is Jackson?"
        self.assertIsNone(extract_answer(text))

    def test_error_statement_is_not_answer(self):
        text = "Cindy has 1.33 - 2 = -0.67 pets. However, there must be an error in the problem statement."
        self.assertIsNone(extract_answer(text))

    def test_stray_comma_is_not_answer(self):
        self.assertIsNone(extract_answer("No numeric answer here,"))

    def test_negative_decimal(self):
        self.assertEqual(extract_answer("The answer is -3.50."), "-3.5")

    def test_fraction_normalization(self):
        self.assertTrue(answers_match("1/2", "0.5"))

    def test_parse_fail(self):
        self.assertIsNone(extract_answer("No numeric answer here."))

    def test_normalize_tail_punctuation(self):
        self.assertEqual(normalize_answer("$1,200."), "1200")


class RewardTest(unittest.TestCase):
    def test_answer_reward_dominates_format_reward(self):
        correct = score_completion("Reasoning.\n\nThe answer is 42.", "42")
        wrong_format = score_completion("Reasoning.\n\nThe answer is 41.", "42")
        self.assertGreater(correct["reward"], wrong_format["reward"])
        self.assertEqual(correct["answer_reward"], 1.0)
        self.assertEqual(correct["format_reward"], 0.2)

    def test_parse_fail_penalty(self):
        score = score_completion("No answer.", "42")
        self.assertFalse(score["parsed"])
        self.assertLess(score["reward"], 0)

    def test_group_diagnostics(self):
        groups = [
            {"completions": [{"correct": True, "parsed": True, "reward": 1.1, "completion_tokens": 5, "stopped_by_eos": True}]},
            {"completions": [{"correct": False, "parsed": True, "reward": 0.1, "completion_tokens": 4, "stopped_by_eos": False}]},
            {
                "completions": [
                    {"correct": True, "parsed": True, "reward": 1.0, "completion_tokens": 6, "stopped_by_eos": True},
                    {"correct": False, "parsed": False, "reward": -0.1, "completion_tokens": 2, "stopped_by_eos": False},
                ]
            },
        ]
        summary = compute_group_diagnostics(groups)
        self.assertAlmostEqual(summary["all_correct_rate"], 1 / 3, places=6)
        self.assertAlmostEqual(summary["all_wrong_rate"], 1 / 3, places=6)
        self.assertAlmostEqual(summary["mixed_rate"], 1 / 3, places=6)
        self.assertEqual(summary["parse_fail_rate"], 0.25)
        self.assertEqual(summary["eos_stop_rate"], 0.5)


class ConfigTest(unittest.TestCase):
    def test_format_run_name_from_yaml_values(self):
        config = {"dataset": {"limit": 500}, "generation": {"num_generations": 8}}
        name = format_run_name("dryrun_g{generation.num_generations}_train{dataset.limit}", config)
        self.assertEqual(name, "dryrun_g8_train500")

    def test_format_run_name_none_as_all(self):
        config = {"train_dataset": {"limit": None}, "grpo": {"num_generations": 4}}
        name = format_run_name("grpo_g{grpo.num_generations}_train{train_dataset.limit}", config)
        self.assertEqual(name, "grpo_g4_trainall")

    def test_hf_model_id_is_not_forced_to_local_path(self):
        self.assertEqual(resolve_local_path_if_exists("Qwen/Qwen2.5-3B-Instruct"), "Qwen/Qwen2.5-3B-Instruct")


class PromptTest(unittest.TestCase):
    def test_include_empty_system_message(self):
        prompt = build_prompt(
            "What is 1+1?",
            {"template": "qwen_chat", "system_prompt": "", "include_empty_system": True},
        )
        self.assertEqual(
            prompt,
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": "What is 1+1?"},
            ],
        )

    def test_empty_system_message_is_skipped_by_default(self):
        prompt = build_prompt(
            "What is 1+1?",
            {"template": "qwen_chat", "system_prompt": "", "include_empty_system": False},
        )
        self.assertEqual(prompt, [{"role": "user", "content": "What is 1+1?"}])


if __name__ == "__main__":
    unittest.main()
