from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from automata_state_probe import compression_accounting, observe_hidden_sequence
from run_sutra_100m_baseline import (
    ESTIMATED_CHECKPOINT_BYTES,
    aligned_training_budget,
    count_causal_targets,
    iter_batches,
    scheduler_state,
    set_optimizer_lr,
    storage_court,
)
from sutra_100m_preflight import EXPECTED_PARAMETER_COUNT, dataset_court, model_court
from sutra_stream import SutraStreamOffset, iter_packed_sequences


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text, *, add_special_tokens=False):
        self.assert_special_tokens = add_special_tokens
        return [ord(char) for char in text]


class StateProbeCourt(unittest.TestCase):
    def test_probe_is_segmented_bounded_and_mutation_sensitive(self):
        hidden = torch.arange(2 * 128 * 768, dtype=torch.float32).reshape(2, 128, 768)
        tokens = torch.arange(2 * 128, dtype=torch.int64).reshape(2, 128)
        records = observe_hidden_sequence(hidden, tokens, segment_tokens=64)
        self.assertEqual(len(records), 2)
        self.assertEqual([record.source_end_token for record in records], [64, 128])
        self.assertTrue(all(record.register_count == 32 for record in records))
        self.assertNotEqual(records[0].transition_root, records[1].transition_root)
        accounting = compression_accounting(raw_history_bytes=1_000_000, records=records)
        self.assertGreater(accounting["state_compression_ratio"], 1)
        self.assertEqual(accounting["candidate_live_state_bytes"], records[-1].state_bytes)
        self.assertEqual(
            accounting["transition_log_bytes"], sum(record.state_bytes for record in records)
        )

    def test_nonfinite_state_is_unavailable(self):
        hidden = torch.zeros((1, 64, 768))
        hidden[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            observe_hidden_sequence(hidden, torch.zeros((1, 64), dtype=torch.int64))


class SutraPreflightCourt(unittest.TestCase):
    def test_exact_parameter_count(self):
        result = model_court()
        self.assertEqual(result["parameter_count_total"], EXPECTED_PARAMETER_COUNT)
        self.assertEqual(result["seq_len"], 1_024)

    def test_dataset_court_binds_order_and_rejects_missing_text(self):
        rows = [{"text": "alpha", "domain": "a"}, {"text": "beta", "domain": "b"}]
        first = dataset_court(rows, limit=2)
        second = dataset_court(reversed(rows), limit=2)
        self.assertNotEqual(first["ordered_row_text_root"], second["ordered_row_text_root"])
        with self.assertRaisesRegex(RuntimeError, "nonempty text"):
            dataset_court([{"not_text": "no"}], limit=1)

    def test_stream_resume_matches_uninterrupted_token_sequence(self):
        rows = [{"text": "abcdefgh"}, {"text": "ijklmnop"}, {"text": "qrstuv"}]
        tokenizer = FakeTokenizer()
        uninterrupted = list(iter_packed_sequences(rows, tokenizer, seq_len=5))
        first_tokens, first_offset = uninterrupted[0]
        resumed_rows = rows[first_offset.row_index :]
        resumed = list(
            iter_packed_sequences(
                resumed_rows,
                tokenizer,
                seq_len=5,
                start=first_offset,
            )
        )
        self.assertEqual(first_tokens.tolist(), uninterrupted[0][0].tolist())
        self.assertEqual(
            [tokens.tolist() for tokens, _ in resumed],
            [tokens.tolist() for tokens, _ in uninterrupted[1:]],
        )
        self.assertEqual(resumed[-1][1].causal_targets_emitted, 20)

    def test_stream_rejects_unreadable_resume_offset(self):
        with self.assertRaisesRegex(RuntimeError, "exceeds encoded row length"):
            list(
                iter_packed_sequences(
                    [{"text": "abc"}],
                    FakeTokenizer(),
                    seq_len=4,
                    start=SutraStreamOffset(row_index=0, token_offset=99),
                )
            )

    def test_training_batches_preserve_exact_causal_target_count(self):
        sequences = iter(
            [
                (torch.arange(8, dtype=torch.int64), SutraStreamOffset(sequences_emitted=1)),
                (torch.arange(8, dtype=torch.int64), SutraStreamOffset(sequences_emitted=2)),
            ]
        )
        batch, offset = next(iter_batches(sequences, batch_size=2))
        self.assertEqual(list(batch["input_ids"].shape), [2, 8])
        self.assertEqual(count_causal_targets(batch["labels"]), 14)
        self.assertEqual(offset.sequences_emitted, 2)

    def test_storage_court_accounts_for_periodic_and_atomic_images(self):
        with patch(
            "run_sutra_100m_baseline.shutil.disk_usage",
            return_value=type("Usage", (), {"free": ESTIMATED_CHECKPOINT_BYTES * 20})(),
        ):
            result = storage_court(
                Path("/tmp"), max_optimizer_steps=1_000, checkpoint_every=100
            )
        self.assertEqual(result["planned_periodic_checkpoints"], 10)
        self.assertEqual(result["required_free_bytes"], ESTIMATED_CHECKPOINT_BYTES * 13)

    def test_budget_aligns_to_complete_optimizer_steps(self):
        result = aligned_training_budget(
            target_causal_targets=10_000, batch_size=2, grad_accum=3
        )
        self.assertEqual(result["causal_targets_per_optimizer_step"], 6_138)
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["aligned_causal_targets"], 12_276)

    def test_scheduler_warms_then_holds_constant(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1.5e-4)
        schedule = scheduler_state(
            base_lr=1.5e-4, warmup_microbatches=2_000, grad_accum=8
        )
        self.assertEqual(schedule["warmup_optimizer_steps"], 250)
        first = set_optimizer_lr(
            optimizer,
            base_lr=1.5e-4,
            optimizer_step_number=1,
            warmup_optimizer_steps=250,
        )
        terminal = set_optimizer_lr(
            optimizer,
            base_lr=1.5e-4,
            optimizer_step_number=251,
            warmup_optimizer_steps=250,
        )
        self.assertAlmostEqual(first, 6.0e-7)
        self.assertAlmostEqual(terminal, 1.5e-4)


if __name__ == "__main__":
    unittest.main()
