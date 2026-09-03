import json
import tempfile
import unittest
from pathlib import Path

from project_mlflow_compat import ProjectionContract, project_event, read_complete_events


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.contract = ProjectionContract(seq_len=1024, aligned_causal_targets=1_500_028_992)

    def test_training_projection_preserves_accumulated_semantics(self):
        projected = project_event(
            {
                "event": "metrics",
                "step": 2,
                "metrics": {
                    "train/loss": 7.0,
                    "train/ppl": 1096.0,
                    "train/causal_targets_seen": 65_472.0,
                    "train/causal_targets_per_second": 11_000.0,
                },
            },
            self.contract,
        )
        self.assertEqual(projected["train/accum_loss"], 7.0)
        self.assertEqual(projected["train/accum_ppl"], 1096.0)
        self.assertEqual(projected["train/raw_tokens_seen"], 65_536.0)
        self.assertEqual(projected["train/sequences_seen"], 64.0)
        self.assertEqual(projected["train/optimizer_steps_completed"], 2.0)
        self.assertNotIn("train_loss", projected)
        self.assertNotIn("train_ppl", projected)

    def test_validation_aliases(self):
        projected = project_event(
            {"event": "metrics", "step": 500, "metrics": {"val/loss": 4.0, "val/ppl": 54.6}},
            self.contract,
        )
        self.assertEqual(projected, {"val_loss": 4.0, "val_ppl": 54.6, "train/optimizer_steps_completed": 500.0})

    def test_partial_jsonl_tail_is_not_consumed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            first = json.dumps({"event": "metrics", "step": 1, "metrics": {"train/loss": 1.0}})
            path.write_bytes((first + "\n" + '{"event":').encode())
            events, offset = read_complete_events(path, 0)
            self.assertEqual(len(events), 1)
            self.assertEqual(offset, len((first + "\n").encode()))


if __name__ == "__main__":
    unittest.main()
