from __future__ import annotations

import unittest

from long_memory_capacity import (
    benchmark_contract,
    build_cases,
    compile_automata_state,
    render_arm,
    retrieve_raw_documents,
    retrieve_state_and_path,
)


class WordTokenizer:
    def __init__(self):
        self.values: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        result = []
        for word in text.split():
            if word not in self.values:
                self.values[word] = len(self.values) + 1
            result.append(self.values[word])
        return result


class LongMemoryCapacityCourt(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer()
        self.cases = build_cases(self.tokenizer, distances=(10, 50), seed=42)

    def test_contract_is_deterministic_and_distance_complete(self):
        first = benchmark_contract(self.cases)
        second = benchmark_contract(build_cases(WordTokenizer(), distances=(10, 50), seed=42))
        self.assertEqual(first["contract_root"], second["contract_root"])
        self.assertEqual(
            {(case.task, case.history_distance) for case in self.cases},
            {
                ("distant_recall", 10),
                ("causal_reasoning", 10),
                ("distant_recall", 50),
                ("causal_reasoning", 50),
            },
        )

    def test_arms_share_bound_but_not_hidden_memory(self):
        case = next(case for case in self.cases if case.task == "distant_recall")
        arms = {
            arm: render_arm(case, self.tokenizer, arm=arm, max_prompt_tokens=64)
            for arm in ("A", "B", "C")
        }
        self.assertTrue(all(len(value["prompt_ids"]) <= 64 for value in arms.values()))
        self.assertEqual(arms["A"]["retrieved_bytes"], 0)
        self.assertGreater(arms["B"]["retrieved_bytes"], 0)
        self.assertGreater(arms["C"]["automata_state_bytes"], 0)
        self.assertGreater(arms["C"]["transition_log_bytes"], arms["C"]["automata_state_bytes"])

    def test_state_resolves_supersession_and_causal_path(self):
        recall = next(case for case in self.cases if case.task == "distant_recall")
        causal = next(case for case in self.cases if case.task == "causal_reasoning")
        self.assertIn("STATE vault-kestrel.access-code=cobalt-29", retrieve_state_and_path(recall))
        causal_evidence = retrieve_state_and_path(causal)
        self.assertTrue(any("beacon-orin.state=glowing" in value for value in causal_evidence))
        self.assertEqual(sum(value.startswith("EDGE") for value in causal_evidence), 4)
        self.assertEqual(
            dict(compile_automata_state(recall)),
            {"vault-kestrel.access-code": "cobalt-29"},
        )

    def test_raw_retrieval_is_deterministic_and_bounded(self):
        case = next(case for case in self.cases if case.task == "causal_reasoning")
        first = retrieve_raw_documents(case, limit=2)
        self.assertEqual(first, retrieve_raw_documents(case, limit=2))
        self.assertEqual(len(first), 2)
        self.assertFalse(any("derived" in value for value in first))
        with self.assertRaisesRegex(ValueError, "positive"):
            retrieve_raw_documents(case, limit=0)

    def test_missing_causal_event_is_refused(self):
        case = next(case for case in self.cases if case.task == "causal_reasoning")
        broken = type(case)(**{**case.__dict__, "causal_path": ("missing",)})
        with self.assertRaisesRegex(ValueError, "missing events"):
            retrieve_state_and_path(broken)


if __name__ == "__main__":
    unittest.main()
