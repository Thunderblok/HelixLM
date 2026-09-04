#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from helix_lm.pretrain_data import (
    PretrainIndexedDataset,
    PretrainPermutation,
    PretrainPermutationSampler,
    PretrainSampleCompiler,
    create_pretrain_indexed_loader,
)
from helix_lm.dataset import ContinuousWindowDataset
from helix_lm.trainer import PretrainTrainer


class IntegerTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        self.last_add_special_tokens = add_special_tokens
        return [int(value) for value in text.split()]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, labels=None, attention_mask=None, cca_step=None):
        return {"loss": self.weight.square() + input_ids.float().mean() * 0}

    def count_parameters(self):
        return {"total": 1}

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "model.pt")


class PretrainDataTest(unittest.TestCase):
    def compile_fixture(self, root: Path, *, samples_per_shard=2):
        return PretrainSampleCompiler(
            IntegerTokenizer(),
            4,
            root,
            samples_per_shard=samples_per_shard,
            source={"fixture": "integer-v1"},
        ).compile(["1 2", "3 4 5", "6 7 8 9 10"])

    def test_compiler_preserves_continuous_eos_joined_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "samples"
            manifest = self.compile_fixture(root)
            dataset = PretrainIndexedDataset(root, verify=True)

            self.assertEqual(manifest.sample_count, 3)
            self.assertEqual(manifest.value["causal_target_count"], 9)
            self.assertEqual(manifest.value["dropped_tail_tokens"], 1)
            self.assertEqual(dataset[0]["input_ids"].tolist(), [1, 2, 99, 3])
            self.assertEqual(dataset[1]["input_ids"].tolist(), [4, 5, 99, 6])
            self.assertEqual(dataset[2]["input_ids"].tolist(), [7, 8, 9, 10])

    def test_persisted_permutation_is_unique_and_resumeable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "epoch-0000.u32"
            created = PretrainPermutation.create(path, 32, 42, epoch=0)
            loaded = PretrainPermutation.load(path)
            values = [int(value) for value in loaded.values()]

            self.assertEqual(created.metadata, loaded.metadata)
            self.assertEqual(sorted(values), list(range(32)))
            self.assertEqual(
                list(PretrainPermutationSampler(loaded, cursor=7)),
                values[7:],
            )

    def test_disk_loader_replays_the_exact_persisted_batch_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "samples"
            self.compile_fixture(root, samples_per_shard=1)
            dataset = PretrainIndexedDataset(root, verify=True)
            permutation = PretrainPermutation.create(
                root / "permutations" / "epoch-0000.u32",
                len(dataset),
                7,
                epoch=0,
            )
            expected_ids = [int(value) for value in permutation.values()]
            loader = create_pretrain_indexed_loader(
                dataset,
                permutation,
                batch_size=2,
                drop_last=False,
                pin_memory=False,
            )

            observed_ids = []
            observed_tokens = []
            for batch in loader:
                observed_ids.extend(batch["sample_id"].tolist())
                observed_tokens.extend(batch["input_ids"].tolist())

            self.assertEqual(observed_ids, expected_ids)
            self.assertEqual(
                observed_tokens,
                [dataset[sample_id]["input_ids"].tolist() for sample_id in expected_ids],
            )
            self.assertTrue(all(label.dtype == torch.int64 for label in [batch["labels"]]))

    def test_compiled_and_live_pretraining_paths_have_identical_samples(self):
        texts = ["1 2", "3 4 5", "6 7 8 9 10"]
        tokenizer = IntegerTokenizer()
        live = list(ContinuousWindowDataset(texts, tokenizer, 4, shuffle=False))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "samples"
            PretrainSampleCompiler(tokenizer, 4, root).compile(texts)
            compiled = PretrainIndexedDataset(root, verify=True)

            self.assertEqual(len(live), len(compiled))
            for sample_id, live_sample in enumerate(live):
                compiled_sample = compiled[sample_id]
                self.assertTrue(torch.equal(live_sample["input_ids"], compiled_sample["input_ids"]))
                self.assertTrue(torch.equal(live_sample["labels"], compiled_sample["labels"]))

    def test_manifest_verification_rejects_mutated_sample_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "samples"
            manifest = self.compile_fixture(root)
            shard_path = root / manifest.value["shards"][0]["file"]
            payload = bytearray(shard_path.read_bytes())
            payload[0] ^= 1
            shard_path.write_bytes(payload)

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                PretrainIndexedDataset(root, verify=True)

    def test_pretrain_trainer_writes_exact_data_resume_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "samples"
            self.compile_fixture(root)
            cfg = SimpleNamespace(
                seq_len=4,
                batch_size=2,
                lr=0.01,
                weight_decay=0.0,
                warmup_steps=1,
                grad_clip=1.0,
                device="cpu",
                epochs=1,
                use_titans_memory=False,
                use_cca=False,
                max_new_tokens=1,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
            )
            trainer = PretrainTrainer(
                model=TinyModel(),
                cfg=cfg,
                train_store_dir=root,
                tokenizer=IntegerTokenizer(),
                output_dir=temporary / "checkpoints",
                seed=42,
                num_workers=0,
                verbose=False,
            )

            trainer.train_epoch(1)
            trainer.save_checkpoint(1, "resume-court")
            state_path = temporary / "checkpoints" / "resume-court" / "pretrain_data_state.json"
            state = __import__("json").loads(state_path.read_text())

            self.assertEqual(state["sample_cursor"], 2)
            self.assertEqual(state["global_step"], 1)
            self.assertEqual(state["dataset_manifest_sha256"], trainer._train_dataset.manifest.manifest_sha256)
            self.assertEqual(state["permutation_sha256"], trainer._train_permutation.metadata["sha256"])


if __name__ == "__main__":
    unittest.main()
