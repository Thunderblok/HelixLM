#!/usr/bin/env python3
"""CPU-only courts for Branch-51 quality/VRAM experiment controls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROGRAM = Path(__file__).resolve().parent
REPO = PROGRAM.parents[1]
RUNNER = PROGRAM / "run_branch51_quality_vram_ablation.py"
BRANCH50_TERMINAL = (
    REPO
    / "experiments"
    / "branch50-linear-context-v0"
    / "evidence"
    / "full-corpus-terminal-summary.json"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("branch51_runner_under_test", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_close(left: float, right: float, *, eps: float = 1e-12) -> None:
    if abs(left - right) > eps:
        raise AssertionError(f"{left!r} != {right!r}")


def baseline_knobs() -> dict[str, object]:
    return {
        "learning_rate": 1.5e-4,
        "warmup_microbatches": 2_000,
        "scheduler_policy": "linear_warmup_then_constant",
        "scheduler_min_lr_ratio": 1.0,
        "checkpoint_every": 500,
        "weight_decay": 0.05,
        "grad_clip": 1.0,
        "dropout": 0.05,
        "attention_dropout": 0.05,
        "ffn_expansion": 2.5,
        "batch_size": 12,
        "grad_accum": 7,
        "n_loops": 3,
    }


def court_branch50_terminal_is_bound_as_control_reference() -> None:
    terminal = json.loads(BRANCH50_TERMINAL.read_text())
    assert terminal["schema"] == "helix.branch50.full-corpus-terminal-summary.v0"
    assert terminal["status"] == "PASS"
    assert terminal["configuration"]["batch_size"] == 12
    assert terminal["configuration"]["gradient_accumulation"] == 7
    assert terminal["configuration"]["ffn_expansion"] == 2.5
    assert terminal["configuration"]["n_loops"] == 3
    assert terminal["configuration"]["grad_buffer_ratio"] == 0.0
    assert terminal["outcome"]["optimizer_steps"] == 34_971
    assert_close(
        terminal["outcome"]["best_validation_perplexity"],
        47.378291172127675,
    )


def court_common_runner_receipt_is_inherited_from_branch50(runner) -> None:
    expected = (
        REPO
        / "experiments"
        / "branch50-linear-context-v0"
        / "executed"
        / "baseline-runner.sha256"
    )
    assert runner.COMMON_RECEIPT == expected
    assert expected.is_file()


def court_optimizer_geometry_is_one_factor(runner) -> None:
    baseline = baseline_knobs()
    geometry = dict(baseline)
    geometry["batch_size"] = 10
    geometry["grad_accum"] = 6
    geometry["warmup_microbatches"] = 1_710
    assert runner.changed_knobs_from(baseline, geometry) == ["optimizer_geometry"]

    geometry = dict(baseline)
    geometry["batch_size"] = 8
    geometry["grad_accum"] = 8
    geometry["warmup_microbatches"] = 2_280
    assert runner.changed_knobs_from(baseline, geometry) == ["optimizer_geometry"]


def court_mixed_geometry_and_ffn_is_not_single_factor(runner) -> None:
    baseline = baseline_knobs()
    hostile = dict(baseline)
    hostile["batch_size"] = 10
    hostile["grad_accum"] = 6
    hostile["warmup_microbatches"] = 1_710
    hostile["ffn_expansion"] = 3.0
    assert runner.changed_knobs_from(baseline, hostile) == [
        "optimizer_geometry",
        "ffn_expansion",
    ]


def court_scheduler_is_one_factor_with_min_ratio(runner) -> None:
    baseline = baseline_knobs()
    cosine = dict(baseline)
    cosine["scheduler_policy"] = "cosine_decay"
    cosine["scheduler_min_lr_ratio"] = 0.1
    assert runner.changed_knobs_from(baseline, cosine) == ["scheduler"]
    state = runner.build_scheduler_state(
        policy="cosine_decay",
        base_lr=1.0,
        warmup_microbatches=20,
        grad_accum=2,
        total_optimizer_steps=110,
        min_lr_ratio=0.1,
    )
    assert state["warmup_optimizer_steps"] == 10
    assert_close(
        runner.optimizer_lr_for_step(
            base_lr=1.0,
            optimizer_step_number=110,
            warmup_optimizer_steps=10,
            total_optimizer_steps=110,
            min_lr_ratio=0.1,
            policy="cosine_decay",
        ),
        0.1,
    )


def court_full_corpus_geometry_math_is_bound(runner) -> None:
    b12a7 = runner.corpus_pass_plan(
        {"tokens": 1_504_000_000},
        seq_len=512,
        batch_size=12,
        grad_accum=7,
    )
    b10a6 = runner.corpus_pass_plan(
        {"tokens": 1_504_000_000},
        seq_len=512,
        batch_size=10,
        grad_accum=6,
    )
    b8a8 = runner.corpus_pass_plan(
        {"tokens": 1_504_000_000},
        seq_len=512,
        batch_size=8,
        grad_accum=8,
    )
    assert b12a7["causal_targets"] == 1_501_062_500
    assert b12a7["optimizer_steps"] == 34_971
    assert b10a6["optimizer_steps"] == 48_959
    assert b8a8["optimizer_steps"] == 45_899


def court_geometry_warmup_profiles_preserve_optimizer_steps(runner) -> None:
    expected = {
        (12, 7): 2_000,
        (10, 6): 1_710,
        (8, 8): 2_280,
        (7, 13): 3_705,
        (12, 9): 2_565,
    }
    assert runner.GEOMETRY_WARMUP_MICROBATCHES == expected
    for (batch_size, grad_accum), warmup_microbatches in expected.items():
        assert warmup_microbatches // grad_accum == 285, batch_size


def court_model_and_mlflow_parameters_are_logged() -> None:
    source = RUNNER.read_text()
    required = [
        '"batch_size": args.batch_size',
        '"gradient_accumulation": args.grad_accum',
        '"effective_sequences": args.batch_size * args.grad_accum',
        '"parameter_count_total": int(params["total"])',
        '"parameter_count_trainable": int(params["trainable"])',
        '"parameter_count_delta_from_branch50": (',
        '"ffn_expansion": cfg.ffn_expansion',
        '"n_loops": cfg.n_loops',
        '"scheduler_policy": schedule["type"]',
        '"scheduler_min_lr_ratio": schedule["minimum_lr_ratio_after_warmup"]',
        '"single_factor_contract": promotion_manifest is None',
        '"thunderline_projection_schema": "thunderline.training.mission.projection.v0"',
        'run_root / "thunderline_training_projection.json"',
    ]
    for needle in required:
        if needle not in source:
            raise AssertionError(f"missing MLflow/config logging field: {needle}")


def court_promotion_manifest_binds_single_factor_decision(runner) -> None:
    selected = baseline_knobs()
    selected["scheduler_policy"] = "cosine_decay"
    selected["scheduler_min_lr_ratio"] = 0.1
    changed = ["scheduler"]
    manifest = {
        "schema": "helix.branch51.promotion-decision.v0",
        "status": "PROMOTED",
        "selected_knobs": selected,
        "changed_knobs": changed,
        "evidence_run_ids": ["control", "scheduler-cosine-r0p1"],
        "decision": "selected by fixed-validation and operational courts",
    }
    assert runner.validate_promotion_manifest(
        manifest,
        resolved_knobs=selected,
        changed_knobs=changed,
    ) == manifest

    hostile = dict(manifest)
    hostile["selected_knobs"] = {**selected, "learning_rate": 2e-4}
    try:
        runner.validate_promotion_manifest(
            hostile,
            resolved_knobs=selected,
            changed_knobs=changed,
        )
    except SystemExit as exc:
        assert "REFUSED" in str(exc), exc
    else:
        raise AssertionError("promotion knob mutation did not turn the court RED")


def main() -> None:
    runner = load_runner()
    courts = [
        court_branch50_terminal_is_bound_as_control_reference,
        court_common_runner_receipt_is_inherited_from_branch50,
        court_optimizer_geometry_is_one_factor,
        court_mixed_geometry_and_ffn_is_not_single_factor,
        court_scheduler_is_one_factor_with_min_ratio,
        court_full_corpus_geometry_math_is_bound,
        court_geometry_warmup_profiles_preserve_optimizer_steps,
        court_model_and_mlflow_parameters_are_logged,
        court_promotion_manifest_binds_single_factor_decision,
    ]
    for court in courts:
        try:
            court(runner)
        except TypeError:
            court()
        print(f"{court.__name__}=PASS")
    print("BRANCH51_QUALITY_VRAM_CONTROL_COURTS=PASS")


if __name__ == "__main__":
    main()
