#!/usr/bin/env python3
"""Supervise the bounded Branch-50 trials into one justified corpus run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO.parent
PROGRAM = REPO / "experiments" / "branch50-linear-context-v0"
RUNNER = PROGRAM / "run_branch50_300m_ablation.py"
LIVE_COURTS = PROGRAM / "run_branch50_live_campaign_courts.py"
COMPARE = PROGRAM / "compare_branch50_ablation_terminals.py"
PROMOTE = PROGRAM / "build_branch50_promotion_manifest.py"
ARTIFACT_ROOT = EXPERIMENT_ROOT / "artifacts" / "ablation-300m-v0"
CAMPAIGN_ROOT = EXPERIMENT_ROOT / "artifacts" / "overnight-campaign-v0"
OLD_QUEUE_PID = 1_859_185
CONTROL_300M = ARTIFACT_ROOT / (
    "branch50-ablation-control-s512-b12-a7-t300000000-20260829T020213Z"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def process_matches(pid: int, needle: str) -> bool:
    cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        content = cmdline.read_bytes()
    except OSError:
        return False
    return needle in content.replace(b"\x00", b" ").decode(errors="replace")


def newest_run(ablation_id: str, target: str) -> Path:
    matches = list(
        ARTIFACT_ROOT.glob(
            f"branch50-ablation-{ablation_id}-s512-b12-a7-t{target}-*"
        )
    )
    if not matches:
        raise RuntimeError(f"no run root for {ablation_id} target={target}")
    return max(matches, key=lambda path: path.stat().st_mtime)


class Campaign:
    def __init__(self, root: Path, expected_head: str, expected_tree: str) -> None:
        self.root = root
        self.expected_head = expected_head
        self.expected_tree = expected_tree
        self.log_path = root / "campaign.log"
        self.state_path = root / "campaign-state.json"
        self.state: dict[str, Any] = {
            "schema": "helix.branch50.overnight-campaign.v0",
            "status": "RUNNING",
            "started_at": utc_now(),
            "source_head": expected_head,
            "source_tree": expected_tree,
            "stages": {},
        }
        root.mkdir(parents=True, exist_ok=False)
        self.write_state()

    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        with self.log_path.open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def write_state(self) -> None:
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.state_path)

    def stage(self, name: str, **facts: Any) -> None:
        stage = self.state["stages"].setdefault(name, {})
        stage.update({"observed_at": utc_now(), **facts})
        self.write_state()

    def run_command(self, name: str, command: list[str]) -> None:
        self.log(f"STAGE_START name={name} argv={json.dumps(command)}")
        log_path = self.root / f"{name}.log"
        with log_path.open("a") as handle:
            result = subprocess.run(
                command,
                cwd=REPO,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        self.stage(name, returncode=result.returncode, log=str(log_path))
        if result.returncode != 0:
            raise RuntimeError(f"stage failed: {name}; see {log_path}")
        self.log(f"STAGE_PASS name={name}")

    def run_candidate(
        self,
        name: str,
        ablation_id: str,
        *,
        target: int | None = None,
        extra: list[str] | None = None,
        full_corpus: bool = False,
        accepted_statuses: tuple[str, ...] = ("PASS",),
    ) -> Path:
        free_bytes = shutil.disk_usage(EXPERIMENT_ROOT).free
        if free_bytes < 100 * 1024**3:
            raise RuntimeError(
                f"disk safety court failed before {name}: free_bytes={free_bytes}"
            )
        self.stage(name, disk_free_bytes_before=free_bytes)
        before = set(ARTIFACT_ROOT.glob(f"branch50-ablation-{ablation_id}-*"))
        command = [sys.executable, str(RUNNER), "--ablation-id", ablation_id]
        if target is not None:
            command.extend(["--target-causal-targets", str(target)])
        if full_corpus:
            command.append("--full-corpus-pass")
        if extra:
            command.extend(extra)
        self.run_command(name, command)
        created = list(set(ARTIFACT_ROOT.glob(f"branch50-ablation-{ablation_id}-*")) - before)
        if len(created) != 1:
            raise RuntimeError(f"expected one new run root for {ablation_id}; got {created}")
        run_root = created[0]
        terminal = load_object(run_root / "terminal.json")
        if terminal.get("status") not in accepted_statuses:
            raise RuntimeError(
                f"candidate terminal refused: {ablation_id} status={terminal.get('status')}"
            )
        self.stage(
            name,
            returncode=0,
            run_root=str(run_root),
            terminal_status=terminal.get("status"),
            mlflow_run_id=terminal.get("mlflow_run_id"),
            checkpoint=terminal.get("checkpoint"),
            best_checkpoint=terminal.get("best_checkpoint"),
        )
        return run_root


def knob_args(manifest: dict[str, Any]) -> list[str]:
    knobs = manifest["selected_knobs"]
    return [
        "--learning-rate",
        str(knobs["learning_rate"]),
        "--warmup-microbatches",
        str(knobs["warmup_microbatches"]),
        "--scheduler-policy",
        str(knobs["scheduler_policy"]),
        "--scheduler-min-lr-ratio",
        str(knobs["scheduler_min_lr_ratio"]),
        "--checkpoint-every",
        str(knobs["checkpoint_every"]),
        "--weight-decay",
        str(knobs["weight_decay"]),
        "--grad-clip",
        str(knobs["grad_clip"]),
        "--dropout",
        str(knobs["dropout"]),
        "--attention-dropout",
        str(knobs["attention_dropout"]),
        "--ffn-expansion",
        str(knobs["ffn_expansion"]),
    ]


def require_terminal(run_root: Path, expected_status: str = "PASS") -> dict[str, Any]:
    terminal = load_object(run_root / "terminal.json")
    if terminal.get("status") != expected_status:
        raise RuntimeError(f"required terminal is not {expected_status}: {run_root}")
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-tree", required=True)
    args = parser.parse_args()
    head = git_value("rev-parse", "HEAD")
    tree = git_value("rev-parse", "HEAD^{tree}")
    dirty = git_value("status", "--porcelain", "--untracked-files=all")
    if dirty or head != args.expected_head or tree != args.expected_tree:
        raise SystemExit(
            f"REFUSED: source admission mismatch dirty={bool(dirty)} head={head} tree={tree}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign = Campaign(CAMPAIGN_ROOT / f"branch50-overnight-{stamp}", head, tree)
    try:
        while process_matches(OLD_QUEUE_PID, "branch50_ablation_queue.sh"):
            campaign.log(f"WAIT_OLD_QUEUE pid={OLD_QUEUE_PID}")
            time.sleep(300)

        lr_root = newest_run("lr1e4", "300000000")
        warmup_root = newest_run("warmup500", "300000000")
        control_terminal = require_terminal(CONTROL_300M)
        lr_terminal = require_terminal(lr_root)
        warmup_terminal = require_terminal(warmup_root)
        campaign.stage(
            "old_queue_terminal",
            control_mlflow=control_terminal["mlflow_run_id"],
            lr_root=str(lr_root),
            lr_mlflow=lr_terminal["mlflow_run_id"],
            warmup_root=str(warmup_root),
            warmup_mlflow=warmup_terminal["mlflow_run_id"],
        )

        campaign.run_command("live_campaign_courts", [sys.executable, str(LIVE_COURTS), "--execute"])

        control100 = campaign.run_candidate(
            "operational_control_100m", "control", target=100_000_000
        )
        scheduler100 = campaign.run_candidate(
            "scheduler_cosine_100m",
            "scheduler-cosine-r0p1",
            target=100_000_000,
            extra=["--scheduler-policy", "cosine_decay", "--scheduler-min-lr-ratio", "0.1"],
        )
        cadence100 = campaign.run_candidate(
            "checkpoint_cadence_100m",
            "checkpoint-every250",
            target=100_000_000,
            extra=["--checkpoint-every", "250"],
        )

        primary_packet = campaign.root / "primary-300m-comparison.json"
        operational_packet = campaign.root / "operational-100m-comparison.json"
        campaign.run_command(
            "compare_primary_300m",
            [
                sys.executable,
                str(COMPARE),
                "--run-root",
                str(CONTROL_300M),
                "--run-root",
                str(lr_root),
                "--run-root",
                str(warmup_root),
                "--output",
                str(primary_packet),
            ],
        )
        campaign.run_command(
            "compare_operational_100m",
            [
                sys.executable,
                str(COMPARE),
                "--run-root",
                str(control100),
                "--run-root",
                str(scheduler100),
                "--run-root",
                str(cadence100),
                "--output",
                str(operational_packet),
            ],
        )

        pilot_manifest = campaign.root / "combined-pilot-promotion.json"
        campaign.run_command(
            "build_combined_pilot_manifest",
            [
                sys.executable,
                str(PROMOTE),
                "--primary-comparison",
                str(primary_packet),
                "--operational-comparison",
                str(operational_packet),
                "--output",
                str(pilot_manifest),
            ],
        )
        pilot_selection = load_object(pilot_manifest)
        combined = campaign.run_candidate(
            "combined_pilot_100m",
            "promoted-combined100m",
            target=100_000_000,
            extra=[
                *knob_args(pilot_selection),
                "--promotion-manifest",
                str(pilot_manifest),
            ],
        )

        full_manifest = campaign.root / "full-corpus-promotion.json"
        campaign.run_command(
            "build_full_corpus_manifest",
            [
                sys.executable,
                str(PROMOTE),
                "--primary-comparison",
                str(primary_packet),
                "--operational-comparison",
                str(operational_packet),
                "--combined-run-root",
                str(combined),
                "--output",
                str(full_manifest),
            ],
        )
        full_selection = load_object(full_manifest)
        full = campaign.run_candidate(
            "full_corpus",
            "promoted-full-corpus",
            full_corpus=True,
            extra=[
                *knob_args(full_selection),
                "--promotion-manifest",
                str(full_manifest),
                "--diminishing-window-evals",
                "10",
                "--diminishing-min-improvement",
                "0.002",
                "--diminishing-patience-windows",
                "5",
                "--diminishing-min-optimizer-steps",
                "6990",
            ],
            accepted_statuses=("PASS", "STOPPED_DIMINISHING_RETURN"),
        )
        campaign.state["status"] = "PASS"
        campaign.state["full_run_root"] = str(full)
        campaign.state["completed_at"] = utc_now()
        campaign.write_state()
        campaign.log(f"CAMPAIGN_PASS full_run_root={full}")
    except BaseException as error:
        campaign.state["status"] = "HOLD"
        campaign.state["error_type"] = type(error).__name__
        campaign.state["error"] = str(error)
        campaign.state["held_at"] = utc_now()
        campaign.write_state()
        campaign.log(f"CAMPAIGN_HOLD error_type={type(error).__name__} error={error}")
        raise


if __name__ == "__main__":
    main()
