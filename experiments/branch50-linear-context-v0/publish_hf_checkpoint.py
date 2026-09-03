#!/usr/bin/env python3
"""Stage and optionally publish an admitted Helix checkpoint to Hugging Face.

The local checkpoint/export remains authoritative. Network publication is an
explicit, post-validation effect enabled only by ``--upload``. Tokens are read
from an environment variable and are never accepted on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


MODEL_NAME_LIMIT = 96
MODEL_NAME_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
TIMESTAMP_RE = re.compile(r"^\d{6}-\d{4}$")
ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_root(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"REFUSED: cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"REFUSED: expected JSON object: {path}")
    return value


def artifact_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def compact_decimal(value: float) -> str:
    text = format(value, ".8g").lower()
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def build_model_name(
    *,
    parameter_count: int,
    run_timestamp: str,
    seq_len: int,
    n_loops: int,
    ffn_expansion: float,
    learning_rate: float,
    epoch: int,
    source_head: str,
) -> str:
    if parameter_count < 1 or seq_len < 1 or n_loops < 1 or epoch < 1:
        raise SystemExit("REFUSED: model naming values must be positive")
    if ffn_expansion <= 0 or learning_rate <= 0:
        raise SystemExit("REFUSED: FFN expansion and learning rate must be positive")
    if not TIMESTAMP_RE.fullmatch(run_timestamp):
        raise SystemExit("REFUSED: run timestamp must use YYMMDD-HHMM")
    if not re.fullmatch(r"[0-9a-f]{7,64}", source_head):
        raise SystemExit("REFUSED: source head must be a hexadecimal commit id")
    size_m = max(1, round(parameter_count / 1_000_000))
    name = (
        f"helix-{size_m}m-{run_timestamp}-s{seq_len}-l{n_loops}"
        f"-f{compact_decimal(ffn_expansion)}-r{compact_decimal(learning_rate)}"
        f"-e{epoch:02d}-g{source_head[:8]}"
    )
    if len(name) > MODEL_NAME_LIMIT:
        raise SystemExit(
            f"REFUSED: generated model name exceeds {MODEL_NAME_LIMIT} characters"
        )
    if not MODEL_NAME_RE.fullmatch(name):
        raise SystemExit(f"REFUSED: generated model name is not Hub-safe: {name}")
    return name


def verify_preflight(
    *, export_dir: Path, preflight_receipt: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not export_dir.is_dir():
        raise SystemExit(f"REFUSED: export directory missing: {export_dir}")
    receipt = load_json(preflight_receipt)
    if receipt.get("status") != "PASS":
        raise SystemExit("REFUSED: checkpoint preflight did not PASS")
    if receipt.get("publication_effect") != "none":
        raise SystemExit("REFUSED: preflight receipt has an unexpected publication effect")
    declared = receipt.get("export", {}).get("files")
    if not isinstance(declared, list) or not declared:
        raise SystemExit("REFUSED: preflight receipt has no export manifest")
    observed = artifact_manifest(export_dir)
    if declared != observed:
        raise SystemExit("REFUSED: export files differ from the admitted preflight manifest")
    declared_root = receipt.get("export", {}).get("manifest_root")
    if declared_root != canonical_root(observed):
        raise SystemExit("REFUSED: preflight export manifest root mismatch")
    return receipt, observed


def prepare_stage(
    *,
    export_dir: Path,
    stage_dir: Path,
    publication: dict[str, Any],
    trainer_checkpoint: Path | None = None,
) -> list[dict[str, Any]]:
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise SystemExit(f"REFUSED: staging directory is not empty: {stage_dir}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(export_dir.rglob("*")):
        relative = source.relative_to(export_dir)
        target = stage_dir / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    (stage_dir / "checkpoint-publication.json").write_text(
        json.dumps(publication, indent=2, sort_keys=True) + "\n"
    )
    if trainer_checkpoint is not None:
        shutil.copy2(trainer_checkpoint, stage_dir / "trainer-state.pt")
    (stage_dir / "README.md").write_text(
        "---\n"
        "library_name: transformers\n"
        "pipeline_tag: text-generation\n"
        "---\n\n"
        f"# {publication['model_name']}\n\n"
        "This checkpoint was saved locally, reloaded, and verified before its "
        "optional Hugging Face publication step.\n\n"
        "## Name legend\n\n"
        "`s` sequence length, `l` Helix loops, `f` FFN expansion, `r` learning "
        "rate, `e` completed epoch, and `g` source commit prefix. In decimal "
        "fields, `p` means decimal point and `m` means a negative exponent sign.\n\n"
        "The repository record describes system-reported training evidence; it "
        "does not independently prove dataset provenance or model quality.\n"
    )
    return artifact_manifest(stage_dir)


def upload_stage(
    *,
    api_factory: Callable[..., Any],
    repo_id: str,
    stage_dir: Path,
    revision: str,
    token: str,
    private: bool,
    commit_message: str,
) -> dict[str, Any]:
    api = api_factory(token=token)
    repo = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=stage_dir,
        revision=revision,
        commit_message=commit_message,
    )
    expected_files = sorted(
        item.relative_to(stage_dir).as_posix()
        for item in stage_dir.rglob("*")
        if item.is_file()
    )
    observed_files = sorted(
        api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)
    )
    normalized_observed = [
        path for path in observed_files if path != ".gitattributes"
    ]
    if normalized_observed != expected_files:
        raise RuntimeError(
            "HUGGING_FACE_READBACK_MISMATCH: "
            f"expected={expected_files!r} observed={observed_files!r}"
        )
    return {
        "repo_url": str(repo),
        "commit_url": str(getattr(commit, "commit_url", "")),
        "commit_oid": str(getattr(commit, "oid", "")),
        "readback_files": observed_files,
    }


def write_terminal(path: Path, terminal: dict[str, Any]) -> None:
    terminal["receipt_root"] = canonical_root(
        {key: value for key, value in terminal.items() if key != "receipt_root"}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument(
        "--trainer-checkpoint",
        type=Path,
        help="optional exact-resume optimizer/RNG checkpoint to publish as trainer-state.pt",
    )
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--publication-receipt", type=Path, required=True)
    parser.add_argument("--hf-namespace", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--run-timestamp", required=True, help="YYMMDD-HHMM")
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main(*, api_factory: Callable[..., Any] | None = None) -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", args.hf_namespace):
        raise SystemExit("REFUSED: invalid Hugging Face namespace")
    if not ENV_NAME_RE.fullmatch(args.token_env):
        raise SystemExit("REFUSED: invalid token environment variable name")
    preflight, export_manifest = verify_preflight(
        export_dir=args.export_dir,
        preflight_receipt=args.preflight_receipt,
    )
    config = load_json(args.resolved_config)
    trainer_checkpoint_record: dict[str, Any] | None = None
    if args.trainer_checkpoint is not None:
        if not args.trainer_checkpoint.is_file():
            raise SystemExit(
                f"REFUSED: trainer checkpoint missing: {args.trainer_checkpoint}"
            )
        declared_checkpoint = preflight.get("checkpoint", {})
        observed_checkpoint_sha = sha256_file(args.trainer_checkpoint)
        if declared_checkpoint.get("sha256") != observed_checkpoint_sha:
            raise SystemExit(
                "REFUSED: trainer checkpoint differs from the preflight subject"
            )
        trainer_checkpoint_record = {
            "included": True,
            "source_path": str(args.trainer_checkpoint.resolve()),
            "sha256": observed_checkpoint_sha,
            "step": declared_checkpoint.get("step"),
            "model_root": declared_checkpoint.get("observed_model_root"),
        }
    parameter_count = int(preflight["resolved_config"]["parameter_count"])
    model_name = build_model_name(
        parameter_count=parameter_count,
        run_timestamp=args.run_timestamp,
        seq_len=int(config["seq_len"]),
        n_loops=int(config["n_loops"]),
        ffn_expansion=float(config["ffn_expansion"]),
        learning_rate=float(config["lr"]),
        epoch=args.epoch,
        source_head=args.source_head,
    )
    repo_id = f"{args.hf_namespace}/{model_name}"
    publication = {
        "schema": "helix.hf_checkpoint_publication.v0",
        "model_name": model_name,
        "repo_id": repo_id,
        "revision": args.revision,
        "visibility": "public" if args.public else "private",
        "epoch": args.epoch,
        "source_head": args.source_head,
        "preflight_receipt_root": preflight.get("receipt_root"),
        "export_manifest_root": canonical_root(export_manifest),
        "config": {
            "parameter_count": parameter_count,
            "seq_len": int(config["seq_len"]),
            "n_loops": int(config["n_loops"]),
            "ffn_expansion": float(config["ffn_expansion"]),
            "learning_rate": float(config["lr"]),
        },
        "trainer_state": trainer_checkpoint_record or {"included": False},
        "upload_requested": args.upload,
        "token_source": f"environment:{args.token_env}",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    staged_manifest = prepare_stage(
        export_dir=args.export_dir,
        stage_dir=args.stage_dir,
        publication=publication,
        trainer_checkpoint=args.trainer_checkpoint,
    )
    terminal: dict[str, Any] = {
        **publication,
        "stage_dir": str(args.stage_dir.resolve()),
        "staged_manifest": staged_manifest,
        "staged_manifest_root": canonical_root(staged_manifest),
        "status": "STAGED",
        "publication_effect": "none",
    }
    write_terminal(args.publication_receipt, terminal)
    if args.upload:
        token = os.environ.get(args.token_env)
        if not token:
            raise SystemExit(
                f"REFUSED: --upload requires a token in {args.token_env}"
            )
        if api_factory is None:
            from huggingface_hub import HfApi

            api_factory = HfApi
        try:
            terminal["hub"] = upload_stage(
                api_factory=api_factory,
                repo_id=repo_id,
                stage_dir=args.stage_dir,
                revision=args.revision,
                token=token,
                private=not args.public,
                commit_message=f"Publish admitted Helix checkpoint epoch {args.epoch}",
            )
        except Exception as exc:
            terminal["status"] = "UPLOAD_FAILED"
            terminal["publication_effect"] = "hugging_face_upload_attempted"
            terminal["failure"] = {
                "error_type": type(exc).__name__,
                "message": str(exc).replace(token, "[REDACTED]")[:500],
            }
            write_terminal(args.publication_receipt, terminal)
            raise
        terminal["status"] = "UPLOADED"
        terminal["publication_effect"] = "hugging_face_model_upload"
    write_terminal(args.publication_receipt, terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
