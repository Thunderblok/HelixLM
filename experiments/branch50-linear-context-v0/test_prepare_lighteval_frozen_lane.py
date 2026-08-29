#!/usr/bin/env python3
"""Dependency-light hostile courts for the frozen Lighteval lane."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("prepare_lighteval_frozen_lane.py")
SPEC = importlib.util.spec_from_file_location("prepare_lighteval_frozen_lane", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def assert_refuses(fn, needle: str) -> None:
    try:
        fn()
    except SystemExit as exc:
        assert needle in str(exc), str(exc)
    else:
        raise AssertionError(f"court did not refuse with {needle!r}")


def write_receipt(path: Path, *, export: Path, seq_len: int = 512) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "lighteval_executed": False,
                "receipt_root": "r" * 64,
                "export": {"path": str(export.resolve())},
                "tokenizer": {
                    "pad_token_id": 50256,
                    "eos_token_id": 50256,
                    "bos_token_id": 50256,
                },
                "resolved_config": {
                    "parameter_count": 53_592_340,
                    "seq_len": seq_len,
                },
                "checkpoint": {"path": "/tmp/checkpoint.pt", "sha256": "c" * 64},
            },
            sort_keys=True,
        )
    )


def main() -> None:
    assert MODULE.canonical_root({"b": 2, "a": 1}) == MODULE.canonical_root(
        {"a": 1, "b": 2}
    )
    assert MODULE.canonical_root({"a": 1}) != MODULE.canonical_root({"a": 2})

    clean_values = {
        ("rev-parse", "HEAD"): "h" * 40,
        ("rev-parse", "HEAD^{tree}"): "t" * 40,
        ("status", "--porcelain", "--untracked-files=all"): "",
    }
    clean_contract = MODULE.resolve_source_contract(
        lambda *args: clean_values[args]
    )
    assert clean_contract == {
        "branch50_source_head": "h" * 40,
        "branch50_source_tree": "t" * 40,
    }

    dirty_values = dict(clean_values)
    dirty_values[("status", "--porcelain", "--untracked-files=all")] = " M dirty.py"
    assert_refuses(
        lambda: MODULE.resolve_source_contract(lambda *args: dirty_values[args]),
        "clean committed Branch-50 source",
    )

    yaml_text = MODULE.render_model_yaml(
        Path("/tmp/export"),
        batch_size=4,
        dtype="float32",
        device="cuda",
    )
    assert "trust_remote_code: false" in yaml_text
    assert "compile: false" in yaml_text
    assert 'dtype: "float32"' in yaml_text
    assert 'device: "cuda"' in yaml_text
    assert "model_loading_kwargs: {}" in yaml_text

    command = MODULE.shell_script(
        [
            "lighteval",
            "accelerate",
            "/tmp/model.yaml",
            "arc:easy|0",
            "--max-samples",
            "16",
        ]
    )
    assert "--max-samples 16" in command
    assert "TOKENIZERS_PARALLELISM=false" in command

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        export = root / "export"
        export.mkdir()
        (export / "config.json").write_text("{}")
        receipt = root / "receipt.json"
        write_receipt(receipt, export=export)

        accepted = MODULE.validate_preflight_receipt(receipt, export)
        assert accepted["status"] == "PASS"

        assert_refuses(
            lambda: MODULE.validate_preflight_receipt(receipt, root / "other-export"),
            "export path mismatch",
        )

        bad = root / "bad.json"
        bad.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "lighteval_executed": True,
                    "export": {"path": str(export.resolve())},
                }
            )
        )
        assert_refuses(
            lambda: MODULE.validate_preflight_receipt(bad, export),
            "must not claim Lighteval execution",
        )

        wrong_tokenizer = root / "wrong-tokenizer.json"
        write_receipt(wrong_tokenizer, export=export)
        payload = json.loads(wrong_tokenizer.read_text())
        payload["tokenizer"]["eos_token_id"] = 0
        wrong_tokenizer.write_text(json.dumps(payload))
        assert_refuses(
            lambda: MODULE.validate_preflight_receipt(wrong_tokenizer, export),
            "tokenizer identity mismatch",
        )

        wrong_seq = root / "wrong-seq.json"
        write_receipt(wrong_seq, export=export, seq_len=1024)
        assert_refuses(
            lambda: MODULE.validate_preflight_receipt(wrong_seq, export),
            "seq_len=512",
        )

    print("BRANCH50_LIGHTEVAL_FROZEN_LANE_COURTS=PASS")


if __name__ == "__main__":
    main()
