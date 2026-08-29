#!/usr/bin/env python3
"""Hostile, dependency-light courts for the Lighteval preflight helpers."""

from __future__ import annotations

import importlib.util
import stat
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

        fake_source = root / "source"
        fake_source.mkdir()
        fake_export = root / "export"
        fake_export.mkdir()
        fake_python = root / "fake-python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' '{\"model_root\":\"expected\","
            "\"tie_word_embeddings\":true,"
            "\"tied_weight_alias_observed\":true}'\n"
        )
        fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
        original_executable = MODULE.sys.executable
        MODULE.sys.executable = str(fake_python)
        try:
            reload_result = MODULE.independent_reload_court(
                export_dir=fake_export,
                source_root=fake_source,
                expected_model_root="expected",
            )
        finally:
            MODULE.sys.executable = original_executable
        assert reload_result["model_root"] == "expected"
        assert reload_result["tied_weight_alias_observed"] is True

        fake_python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' '{\"model_root\":\"wrong\","
            "\"tie_word_embeddings\":true,"
            "\"tied_weight_alias_observed\":false}'\n"
        )
        fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
        MODULE.sys.executable = str(fake_python)
        try:
            try:
                MODULE.independent_reload_court(
                    export_dir=fake_export,
                    source_root=fake_source,
                    expected_model_root="expected",
                )
            except SystemExit as exc:
                assert "model root" in str(exc)
            else:
                raise AssertionError("independent reload mismatch failed to turn RED")
        finally:
            MODULE.sys.executable = original_executable

        fake_python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' '{\"model_root\":\"expected\","
            "\"tie_word_embeddings\":true,"
            "\"tied_weight_alias_observed\":false}'\n"
        )
        fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
        MODULE.sys.executable = str(fake_python)
        try:
            try:
                MODULE.independent_reload_court(
                    export_dir=fake_export,
                    source_root=fake_source,
                    expected_model_root="expected",
                )
            except SystemExit as exc:
                assert "tied weights" in str(exc)
            else:
                raise AssertionError("untied reload failed to turn RED")
        finally:
            MODULE.sys.executable = original_executable

    print("LIGHEVAL_CHECKPOINT_PREFLIGHT_HELPER_COURTS=PASS")


if __name__ == "__main__":
    main()
