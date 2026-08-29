#!/usr/bin/env python3
"""Hostile, dependency-light courts for the Lighteval preflight helpers."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("lighteval_checkpoint_preflight.py")
SPEC = importlib.util.spec_from_file_location("lighteval_checkpoint_preflight", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    assert MODULE.canonical_root({"b": 2, "a": 1}) == MODULE.canonical_root(
        {"a": 1, "b": 2}
    )
    assert MODULE.canonical_root({"a": 1}) != MODULE.canonical_root({"a": 2})

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "b.bin").write_bytes(b"second")
        (root / "a.bin").write_bytes(b"first")
        manifest = MODULE.artifact_manifest(root)
        assert [entry["path"] for entry in manifest] == ["a.bin", "b.bin"]
        assert all(len(entry["sha256"]) == 64 for entry in manifest)

        occupied = root / "occupied"
        occupied.mkdir()
        (occupied / "evidence").write_text("held")
        try:
            MODULE.refuse_dirty_output(occupied, overwrite=False)
        except SystemExit as exc:
            assert "REFUSED" in str(exc)
        else:
            raise AssertionError("occupied output court failed to turn RED")

    print("LIGHEVAL_CHECKPOINT_PREFLIGHT_HELPER_COURTS=PASS")


if __name__ == "__main__":
    main()
