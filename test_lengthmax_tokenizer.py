import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from helix_lm.dataset import DocumentAwareDataset
from helix_lm.tokenizer import HelixTokenizer


SPECIAL = "<|endoftext|>"


def write_artifact(path: Path, **overrides) -> Path:
    multis = [b"abc"] + [b"\xff" + i.to_bytes(2, "big") for i in range(49_999)]
    vocab = {bytes([value]).decode("latin-1"): value for value in range(256)}
    for payload in multis:
        vocab[payload.decode("latin-1")] = len(vocab)
    vocab[SPECIAL] = len(vocab)
    artifact = {
        "vocab": vocab,
        "max_token_len": 3,
        "vocab_size": 50_257,
        "algorithm": "iterative-byte-bpe-vocab-leftmost-longest-v0",
        "special_tokens": [SPECIAL],
        "training": {"selection_data": "test-only"},
    }
    artifact.update(overrides)
    path.write_text(json.dumps(artifact, ensure_ascii=True), encoding="utf-8")
    return path


class LengthMaxTokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.artifact = write_artifact(self.root / "tokenizer.json")

    def tearDown(self):
        self.tmp.cleanup()

    def tokenizer(self) -> HelixTokenizer:
        return HelixTokenizer(f"lengthmax:{self.artifact.resolve()}")

    def test_matched_gpt2_identity_and_longest_match(self):
        tokenizer = self.tokenizer()
        self.assertEqual(len(tokenizer), 50_257)
        self.assertEqual(tokenizer.pad_token_id, 50_256)
        self.assertEqual(tokenizer.eos_token_id, 50_256)
        self.assertEqual(tokenizer.bos_token_id, 50_256)
        self.assertEqual(tokenizer.unk_token_id, 50_256)
        self.assertEqual(tokenizer.encode("zabcq"), [ord("z"), 256, ord("q")])
        self.assertEqual(
            tokenizer.encode("zabcq", add_special_tokens=True),
            tokenizer.encode("zabcq", add_special_tokens=False),
        )

    def test_utf8_round_trip_and_special_token_projection(self):
        tokenizer = self.tokenizer()
        text = "Helix — Καλημέρα — こんにちは — 🧬\n"
        ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertEqual(tokenizer.decode(ids + [50_256]), text)
        self.assertEqual(
            tokenizer.decode(ids + [50_256], skip_special_tokens=False),
            text + SPECIAL,
        )

    def test_batch_and_document_chunking_contract(self):
        tokenizer = self.tokenizer()
        encoded = tokenizer(
            ["abc", "z"],
            return_tensors="pt",
            padding=True,
        )
        self.assertTrue(torch.equal(encoded["input_ids"], torch.tensor([[256], [ord("z")]])))
        self.assertTrue(torch.equal(encoded["attention_mask"], torch.ones((2, 1), dtype=torch.long)))

        dataset = DocumentAwareDataset(
            ["abc", "zabcq"],
            tokenizer,
            seq_len=4,
            stride=4,
            lazy=False,
        )
        first = dataset[0]
        self.assertEqual(first["input_ids"].tolist(), [256, 50_256, 50_256, 50_256])
        self.assertEqual(first["labels"].tolist(), [256, 50_256, -100, -100])
        self.assertEqual(first["attention_mask"].tolist(), [1, 1, 0, 0])

    def test_checkpoint_round_trip_binds_artifact_hash(self):
        tokenizer = self.tokenizer()
        checkpoint = self.root / "checkpoint"
        tokenizer.save_pretrained(checkpoint)
        restored = HelixTokenizer.from_pretrained(checkpoint)
        self.assertEqual(restored.encode("zabcq"), tokenizer.encode("zabcq"))
        self.assertEqual(restored.decode(restored.encode("🧬")), "🧬")

        artifact = checkpoint / "lengthmax-tokenizer.json"
        artifact.write_bytes(artifact.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            HelixTokenizer.from_pretrained(checkpoint)

    def test_malformed_or_ambiguous_artifacts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "absolute artifact path"):
            HelixTokenizer("lengthmax:relative.json")

        bad_algorithm = write_artifact(
            self.root / "bad-algorithm.json",
            algorithm="some-other-tokenizer-v0",
        )
        with self.assertRaisesRegex(ValueError, "Unsupported LengthMAX algorithm"):
            HelixTokenizer(f"lengthmax:{bad_algorithm.resolve()}")

        malformed = json.loads(self.artifact.read_text(encoding="utf-8"))
        malformed["vocab"]["A"] = 999
        malformed_path = self.root / "duplicate-id.json"
        malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate LengthMAX token id"):
            HelixTokenizer(f"lengthmax:{malformed_path.resolve()}")

    def test_checkpoint_config_records_real_artifact_digest(self):
        checkpoint = self.root / "checkpoint"
        self.tokenizer().save_pretrained(checkpoint)
        config = json.loads((checkpoint / "helix_tokenizer_config.json").read_text())
        observed = hashlib.sha256((checkpoint / "lengthmax-tokenizer.json").read_bytes()).hexdigest()
        self.assertEqual(config["artifact_sha256"], observed)
        self.assertEqual(config["vocab_size"], 50_257)
        self.assertEqual(config["algorithm"], "iterative-byte-bpe-vocab-leftmost-longest-v0")


if __name__ == "__main__":
    unittest.main()
