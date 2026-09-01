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


def court_profile_binding_refuses_an_omitted_ffn_flag(runner) -> None:
    hostile = selected_knobs()
    hostile["learning_rate"] = 0.0002
    hostile["ffn_expansion"] = 2.5

    try:
        runner.resolve_branch53_profile(
            expected_profile="ffn_expansion_3p0_v0",
            resolved_knobs=hostile,
            changed_knobs=["learning_rate"],
        )
    except SystemExit as exc:
        assert "profile/config mismatch" in str(exc)
    else:
        raise AssertionError("omitted FFN flag did not turn the profile court RED")

    assert (
        runner.resolve_branch53_profile(
            expected_profile="auto",
            resolved_knobs=hostile,
            changed_knobs=["learning_rate"],
        )
        == "learning_rate_ablation_v0"
    )
    assert (
        runner.resolve_branch53_profile(
            expected_profile="ffn_expansion_3p0_v0",
            resolved_knobs=selected_knobs(),
            changed_knobs=[
                "optimizer_geometry",
                "ffn_expansion",
                "activation_checkpointing",
            ],
        )
        == "ffn_expansion_3p0_v0"
    )


def court_ffn_profile_binds_the_observed_parameter_count(runner) -> None:
    try:
        runner.validate_branch53_parameter_count(
            profile="ffn_expansion_3p0_v0",
            parameter_count={"total": 53_592_340, "trainable": 53_592_340},
        )
    except SystemExit as exc:
        assert "parameter count mismatch" in str(exc)
    else:
        raise AssertionError("baseline-sized FFN 3.0 model did not turn the court RED")

    runner.validate_branch53_parameter_count(
        profile="ffn_expansion_3p0_v0",
        parameter_count={"total": 54_771_988, "trainable": 54_771_988},
    )


def main() -> None:
    runner = load_runner()
    court_profile_binding_refuses_an_omitted_ffn_flag(runner)
    court_ffn_profile_binds_the_observed_parameter_count(runner)
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
