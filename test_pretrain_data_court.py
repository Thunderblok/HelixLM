#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pretrain_data_court import CourtFailure, IntegerTokenizer, replay_store, run_fixture_equivalence
from helix_lm.pretrain_data import (
    PERMUTATION_DTYPE,
    PretrainPermutation,
    PretrainSampleCompiler,
)


class PretrainDataCourtTest(unittest.TestCase):
    def test_independent_fixture_equivalence_court_passes(self):
        terminal = run_fixture_equivalence()

        self.assertEqual(terminal["sample_count"], 4)
        self.assertEqual(terminal["causal_target_count"], 12)
        self.assertEqual(
            terminal["ordered_sample_ids_sha256"],
            terminal["permutation_sha256"],
        )

    def test_store_replay_reports_exact_order_and_performance(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Path(temporary) / "samples"
            manifest = PretrainSampleCompiler(
                IntegerTokenizer(),
                4,
                store,
                samples_per_shard=2,
            ).compile(["1 2 3", "4 5 6", "7 8 9", "10 11 12", "13 14 15"])
            permutation = PretrainPermutation.create(
                store / "permutations" / "epoch-0000-seed-42.u32",
                manifest.sample_count,
                42,
            )

            terminal = replay_store(
                store,
                permutation.path,
                batch_size=2,
                num_workers=0,
                maximum_samples=None,
                minimum_samples_per_second=0,
            )

            self.assertTrue(terminal["complete_replay"])
            self.assertEqual(terminal["sample_count"], manifest.sample_count)
            self.assertEqual(
                terminal["ordered_sample_ids_sha256"],
                permutation.metadata["sha256"],
            )
            self.assertGreater(terminal["samples_per_second"], 0)

    def test_performance_floor_convicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Path(temporary) / "samples"
            manifest = PretrainSampleCompiler(IntegerTokenizer(), 4, store).compile(
                ["1 2 3", "4 5 6"]
            )
            permutation = PretrainPermutation.create(
                store / "permutations" / "epoch-0000-seed-42.u32",
                manifest.sample_count,
                42,
            )

            with self.assertRaisesRegex(CourtFailure, "below the declared"):
                replay_store(
                    store,
                    permutation.path,
                    batch_size=1,
                    num_workers=0,
                    maximum_samples=None,
                    minimum_samples_per_second=1e30,
                )

    def test_complete_store_replay_convicts_duplicate_and_omitted_sample_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Path(temporary) / "samples"
            manifest = PretrainSampleCompiler(IntegerTokenizer(), 4, store).compile(
                ["1 2 3", "4 5 6", "7 8 9", "10 11 12"]
            )
            permutation = PretrainPermutation.create(
                store / "permutations" / "epoch-0000-seed-42.u32",
                manifest.sample_count,
                42,
            )
            values = np.asarray(permutation.values()).copy()
            values[-1] = values[0]
            values.astype(PERMUTATION_DTYPE, copy=False).tofile(permutation.path)
            metadata_path = permutation.path.with_suffix(permutation.path.suffix + ".json")
            metadata = json.loads(metadata_path.read_text())
            metadata["sha256"] = hashlib.sha256(permutation.path.read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata))

            with self.assertRaisesRegex(CourtFailure, "duplicate sample ID"):
                replay_store(
                    store,
                    permutation.path,
                    batch_size=2,
                    num_workers=0,
                    maximum_samples=None,
                    minimum_samples_per_second=0,
                )


if __name__ == "__main__":
    unittest.main()
