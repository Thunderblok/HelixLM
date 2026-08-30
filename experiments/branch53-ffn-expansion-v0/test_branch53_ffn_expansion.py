#!/usr/bin/env python3
"""Hostile contract courts for the queued Branch53 FFN expansion."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT.parent / "branch51-quality-vram-v0" / "run_branch51_quality_vram_ablation.py"
MANIFEST = ROOT / "evidence" / "ffn3p0-promotion.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("branch53_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise AssertionError("runner import unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_knobs() -> dict[str, object]:
    return {
        "learning_rate": 0.00015,
        "warmup_microbatches": 2280,
        "scheduler_policy": "linear_warmup_then_constant",
        "scheduler_min_lr_ratio": 1.0,
        "checkpoint_every": 500,
        "weight_decay": 0.05,
        "grad_clip": 1.0,
        "dropout": 0.05,
        "attention_dropout": 0.05,
        "ffn_expansion": 3.0,
        "batch_size": 8,
        "grad_accum": 8,
        "n_loops": 3,
        "activation_checkpointing": True,
    }


def main() -> None:
    runner = load_runner()
    manifest = json.loads(MANIFEST.read_text())
    changed = ["optimizer_geometry", "ffn_expansion", "activation_checkpointing"]
    runner.validate_promotion_manifest(
        manifest,
        resolved_knobs=selected_knobs(),
        changed_knobs=changed,
    )

    hostile = dict(manifest)
    hostile["selected_knobs"] = dict(manifest["selected_knobs"])
    hostile["selected_knobs"]["ffn_expansion"] = 2.5
    try:
        runner.validate_promotion_manifest(
            hostile,
            resolved_knobs=selected_knobs(),
            changed_knobs=changed,
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("FFN mutation did not turn the promotion court RED")

    print("BRANCH53_FFN_EXPANSION_COURTS=PASS")


if __name__ == "__main__":
    main()
