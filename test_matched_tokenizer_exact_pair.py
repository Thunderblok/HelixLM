import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import launch_matched_tokenizer_exact_pair as launch
import prepare_matched_tokenizer_exact_corpora as prepare
import run_matched_tokenizer_exact_arm as runner


class FakeTokenizer:
    eos_token_id = 50_256

    def __init__(self, offset: int):
        self.offset = offset

    def encode(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return [self.offset + value for value in text.encode("utf-8")]


def passing_terminal(arm: str) -> dict:
    return {
        "status": "PASS",
        "pilot_arm": arm,
        "steps": launch.MAX_OPTIMIZER_STEPS,
        "data_offset": {"causal_targets_seen": launch.TARGET_CAUSAL_TARGETS},
        "initial_model_state_root": "a" * 64,
        "resolved_config_root": "b" * 64,
        "mlflow_run_id": f"run-{arm}",
        "raw_bytes_seen": 123,
        "last_validation": {"exact_bits_per_byte": 1.5},
        "run_root": f"/tmp/{arm}",
    }


class MatchedPilotTest(unittest.TestCase):
    def test_window_accounting_separates_exposure_from_causal_target_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token_path = root / "shard-00000.u16"
            raw_path = root / "shard-00000.rawbytes.u32"
            prepare.np.asarray([1, 2, 3, 4], dtype="<u2").tofile(token_path)
            prepare.np.asarray([10, 20, 30, 40], dtype="<u4").tofile(raw_path)
            shard = runner.ShardRef(0, token_path, 4, 8, "unused", raw_path, "unused")
            row, offset, exposed, causal = next(
                runner.iter_u16_windows(
                    [shard], seq_len=4, seed=42, start=runner.DataOffset(), target_causal_targets=3
                )
            )
            self.assertEqual(row.tolist(), [1, 2, 3, 4])
            self.assertEqual(exposed, 100)
            self.assertEqual(causal, 90)
            self.assertEqual(offset.raw_bytes_seen, 100)
            self.assertEqual(offset.causal_raw_bytes_seen, 90)

    def test_materialization_binds_both_arms_to_identical_raw_rows(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            prepare, "iter_parquet_texts", return_value=iter(["abc", "de"])
        ), mock.patch.object(
            prepare, "token_byte_lengths", side_effect=lambda _tokenizer, ids, arm: [1] * len(ids)
        ):
            manifests = prepare.materialize_matched_split(
                paths=[Path("unused.parquet")],
                split="train",
                tokenizers={"gpt2": FakeTokenizer(0), "lengthmax": FakeTokenizer(256)},
                tokenizer_identities={"gpt2": {"name": "gpt2"}, "lengthmax": {"name": "lengthmax"}},
                output_root=Path(directory),
                target_raw_bytes=5,
                shard_tokens=512,
                sources=[],
            )
            prepare.assert_matched_raw_subject(manifests)
            self.assertEqual(
                manifests["gpt2"]["source_record_stream_sha256"],
                manifests["lengthmax"]["source_record_stream_sha256"],
            )
            self.assertEqual(manifests["gpt2"]["rows"], 2)
            self.assertEqual(manifests["lengthmax"]["rows"], 2)
            self.assertEqual(manifests["gpt2"]["raw_utf8_bytes"], 5)
            self.assertEqual(manifests["gpt2"]["attributed_raw_bytes"], 5)
            self.assertEqual(manifests["lengthmax"]["attributed_raw_bytes"], 5)
            self.assertNotEqual(
                manifests["gpt2"]["manifest_root"], manifests["lengthmax"]["manifest_root"]
            )

    def test_pair_verifier_accepts_only_matched_model_state_and_budget(self):
        terminals = {arm: passing_terminal(arm) for arm in launch.ARMS}
        result = launch.verify_pair(
            pair_id="pair",
            identity={"head": "h", "tree": "t", "dirty": ""},
            data={"summary": {"summary_root": "s"}},
            terminals=terminals,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["causal_targets_per_arm"], launch.TARGET_CAUSAL_TARGETS)
        self.assertEqual(set(result["arms"]), set(launch.ARMS))

    def test_pair_verifier_refuses_different_initial_model_state(self):
        terminals = {arm: passing_terminal(arm) for arm in launch.ARMS}
        terminals["lengthmax"]["initial_model_state_root"] = "c" * 64
        with self.assertRaisesRegex(SystemExit, "initial model state differs"):
            launch.verify_pair(
                pair_id="pair",
                identity={"head": "h", "tree": "t", "dirty": ""},
                data={"summary": {"summary_root": "s"}},
                terminals=terminals,
            )

    def test_preparation_manifests_are_parseable_json(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            prepare, "iter_parquet_texts", return_value=iter(["abcde"])
        ), mock.patch.object(
            prepare, "token_byte_lengths", side_effect=lambda _tokenizer, ids, arm: [1] * len(ids)
        ):
            prepare.materialize_matched_split(
                paths=[Path("unused.parquet")],
                split="validation",
                tokenizers={"gpt2": FakeTokenizer(0), "lengthmax": FakeTokenizer(256)},
                tokenizer_identities={"gpt2": {"name": "gpt2"}, "lengthmax": {"name": "lengthmax"}},
                output_root=Path(directory),
                target_raw_bytes=5,
                shard_tokens=512,
                sources=[],
            )
            for arm in launch.ARMS:
                path = Path(directory) / "validation" / arm / "manifest.json"
                self.assertEqual(json.loads(path.read_text())["tokenizer"], arm)


if __name__ == "__main__":
    unittest.main()
