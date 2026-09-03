#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).with_name("publish_hf_checkpoint.py")
SPEC = importlib.util.spec_from_file_location("publish_hf_checkpoint", MODULE_PATH)
assert SPEC and SPEC.loader
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fixture(root: Path) -> tuple[Path, Path, Path]:
    export = root / "export"
    export.mkdir()
    (export / "config.json").write_text('{"model_type":"helix"}\n')
    (export / "model.safetensors").write_bytes(b"weights")
    (root / "trainer.pt").write_bytes(b"optimizer-and-rng-state")
    files = publisher.artifact_manifest(export)
    preflight = root / "preflight.json"
    write_json(
        preflight,
        {
            "status": "PASS",
            "publication_effect": "none",
            "receipt_root": "a" * 64,
            "resolved_config": {"parameter_count": 54_771_988},
            "checkpoint": {
                "sha256": publisher.sha256_file(root / "trainer.pt"),
                "step": 100,
                "observed_model_root": "b" * 64,
            },
            "export": {
                "files": files,
                "manifest_root": publisher.canonical_root(files),
            },
        },
    )
    config = root / "resolved-config.json"
    write_json(
        config,
        {
            "seq_len": 512,
            "n_loops": 3,
            "ffn_expansion": 3.0,
            "lr": 0.0002,
        },
    )
    return export, preflight, config


def test_model_name_is_bounded_and_legible() -> None:
    name = publisher.build_model_name(
        parameter_count=54_771_988,
        run_timestamp="260902-2125",
        seq_len=512,
        n_loops=3,
        ffn_expansion=3.0,
        learning_rate=0.0002,
        epoch=2,
        source_head="adfb95e372d3893e8bba254413699e208e972c60",
    )
    assert name == "helix-55m-260902-2125-s512-l3-f3-r0p0002-e02-gadfb95e3"
    assert len(name) <= 96


def test_tampered_export_is_refused() -> None:
    with tempfile.TemporaryDirectory() as directory:
        export, preflight, _ = fixture(Path(directory))
        (export / "model.safetensors").write_bytes(b"changed")
        try:
            publisher.verify_preflight(
                export_dir=export,
                preflight_receipt=preflight,
            )
        except SystemExit as exc:
            assert "differ" in str(exc)
        else:
            raise AssertionError("tampered export was accepted")


class FakeApi:
    def __init__(self, *, token: str):
        assert token == "test-token"
        self.files: list[str] = []

    def create_repo(self, **kwargs):
        assert kwargs["private"] is True
        return "https://huggingface.co/Thunderblok/test"

    def upload_folder(self, **kwargs):
        root = Path(kwargs["folder_path"])
        self.files = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )
        return SimpleNamespace(commit_url="https://example/commit", oid="deadbeef")

    def list_repo_files(self, **kwargs):
        return self.files


def run_main(
    root: Path, *, upload: bool, include_trainer_state: bool = False
) -> dict[str, object]:
    export, preflight, config = fixture(root)
    receipt = root / "publication.json"
    argv = [
        str(MODULE_PATH),
        "--export-dir",
        str(export),
        "--preflight-receipt",
        str(preflight),
        "--resolved-config",
        str(config),
        "--stage-dir",
        str(root / "stage"),
        "--publication-receipt",
        str(receipt),
        "--hf-namespace",
        "Thunderblok",
        "--epoch",
        "1",
        "--run-timestamp",
        "260902-2125",
        "--source-head",
        "adfb95e372d3893e8bba254413699e208e972c60",
    ]
    if upload:
        argv.append("--upload")
    if include_trainer_state:
        argv.extend(["--trainer-checkpoint", str(root / "trainer.pt")])
    with mock.patch.object(sys, "argv", argv), mock.patch.dict(
        os.environ,
        {"HF_TOKEN": "test-token"} if upload else {},
        clear=False,
    ):
        publisher.main(api_factory=FakeApi)
    return json.loads(receipt.read_text())


def test_default_stages_without_network_effect() -> None:
    with tempfile.TemporaryDirectory() as directory:
        terminal = run_main(Path(directory), upload=False)
        assert terminal["status"] == "STAGED"
        assert terminal["publication_effect"] == "none"
        assert terminal["upload_requested"] is False


def test_explicit_upload_is_private_and_read_back() -> None:
    with tempfile.TemporaryDirectory() as directory:
        terminal = run_main(
            Path(directory), upload=True, include_trainer_state=True
        )
        assert terminal["status"] == "UPLOADED"
        assert terminal["publication_effect"] == "hugging_face_model_upload"
        assert terminal["hub"]["commit_oid"] == "deadbeef"
        assert terminal["hub"]["readback_files"] == [
            "README.md",
            "checkpoint-publication.json",
            "config.json",
            "model.safetensors",
            "trainer-state.pt",
        ]
        assert terminal["trainer_state"]["included"] is True
        assert terminal["trainer_state"]["step"] == 100


if __name__ == "__main__":
    test_model_name_is_bounded_and_legible()
    test_tampered_export_is_refused()
    test_default_stages_without_network_effect()
    test_explicit_upload_is_private_and_read_back()
    print("HF_CHECKPOINT_PUBLISHER_TESTS=PASS")
