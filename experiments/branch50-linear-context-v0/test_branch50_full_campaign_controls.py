#!/usr/bin/env python3
"""CPU-only courts for Branch-50 full-campaign control helpers."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path


RUNNER = Path(__file__).with_name("run_branch50_300m_ablation.py")


def load_runner():
    spec = importlib.util.spec_from_file_location("branch50_runner_under_test", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_close(left: float, right: float, *, eps: float = 1e-12) -> None:
    if abs(left - right) > eps:
        raise AssertionError(f"{left!r} != {right!r}")


def court_default_scheduler_matches_existing_warmup_constant(runner) -> None:
    state = runner.build_scheduler_state(
        policy="linear_warmup_then_constant",
        base_lr=1.5e-4,
        warmup_microbatches=2_000,
        grad_accum=7,
        total_optimizer_steps=6_990,
        min_lr_ratio=1.0,
    )
    assert state == {
        "type": "linear_warmup_then_constant",
        "base_lr": 1.5e-4,
        "warmup_microbatches": 2_000,
        "grad_accum": 7,
        "warmup_optimizer_steps": 285,
        "total_optimizer_steps": 6_990,
        "minimum_lr_ratio_after_warmup": 1.0,
    }
    assert_close(
        runner.optimizer_lr_for_step(
            base_lr=1.5e-4,
            optimizer_step_number=1,
            warmup_optimizer_steps=285,
            total_optimizer_steps=6_990,
            min_lr_ratio=1.0,
            policy="linear_warmup_then_constant",
        ),
        1.5e-4 / 285,
    )
    assert_close(
        runner.optimizer_lr_for_step(
            base_lr=1.5e-4,
            optimizer_step_number=285,
            warmup_optimizer_steps=285,
            total_optimizer_steps=6_990,
            min_lr_ratio=1.0,
            policy="linear_warmup_then_constant",
        ),
        1.5e-4,
    )
    assert_close(
        runner.optimizer_lr_for_step(
            base_lr=1.5e-4,
            optimizer_step_number=6_990,
            warmup_optimizer_steps=285,
            total_optimizer_steps=6_990,
            min_lr_ratio=1.0,
            policy="linear_warmup_then_constant",
        ),
        1.5e-4,
    )


def court_cosine_scheduler_has_real_min_ratio_effect(runner) -> None:
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
            optimizer_step_number=1,
            warmup_optimizer_steps=10,
            total_optimizer_steps=110,
            min_lr_ratio=0.1,
            policy="cosine_decay",
        ),
        0.1,
    )
    assert_close(
        runner.optimizer_lr_for_step(
            base_lr=1.0,
            optimizer_step_number=10,
            warmup_optimizer_steps=10,
            total_optimizer_steps=110,
            min_lr_ratio=0.1,
            policy="cosine_decay",
        ),
        1.0,
    )
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


def court_full_corpus_remainder_is_bound(runner) -> None:
    plan = runner.corpus_pass_plan(
        {"tokens": 1_504_000_000},
        seq_len=512,
        batch_size=12,
        grad_accum=7,
    )
    assert plan == {
        "raw_tokens": 1_504_000_000,
        "total_samples": 2_937_500,
        "causal_targets": 1_501_062_500,
        "samples_per_full_optimizer_step": 84,
        "full_optimizer_steps": 34_970,
        "remaining_samples": 20,
        "optimizer_steps": 34_971,
    }


def court_diminishing_stop_is_deterministic(runner) -> None:
    improving = [
        {"step": 100, "val_loss": 5.0},
        {"step": 200, "val_loss": 4.9},
        {"step": 300, "val_loss": 4.8},
        {"step": 400, "val_loss": 4.7},
        {"step": 500, "val_loss": 4.6},
    ]
    hold = runner.diminishing_return_decision(
        improving,
        window_evals=2,
        min_improvement=0.01,
        patience_windows=2,
        min_optimizer_steps=200,
    )
    assert not hold["should_stop"], hold

    flat = [
        {"step": 100, "val_loss": 5.0},
        {"step": 200, "val_loss": 4.9},
        {"step": 300, "val_loss": 4.899},
        {"step": 400, "val_loss": 4.8985},
        {"step": 500, "val_loss": 4.8981},
        {"step": 600, "val_loss": 4.8980},
    ]
    stop = runner.diminishing_return_decision(
        flat,
        window_evals=2,
        min_improvement=0.01,
        patience_windows=2,
        min_optimizer_steps=200,
    )
    assert stop["should_stop"], stop
    assert stop["bad_windows"] >= 2, stop


def court_diminishing_stop_is_not_promotional_pass(runner) -> None:
    stop_state = {
        "stop_reason": "diminishing_return",
        "diminishing_decision": {"enabled": True, "should_stop": True},
    }
    record = runner.terminal_status_record(
        max_optimizer_steps=0,
        stop_state=stop_state,
        mlflow_errors=[],
    )
    assert record["status"] == "STOPPED_DIMINISHING_RETURN", record
    assert record["promotion_eligible"] is False, record
    assert record["numerical_health"] == "PASS", record
    assert record["checkpoint_health"] == "PASS", record


def court_periodic_checkpoint_after_validation_source_order() -> None:
    source = RUNNER.read_text()
    validation_index = source.index("if step % args.eval_every == 0 or step == steps:")
    checkpoint_index = source.index("if step % args.checkpoint_every == 0:")
    if checkpoint_index < validation_index:
        raise AssertionError("periodic checkpoint still occurs before validation")
    if '"validation_history_len": len(validation_history)' not in source:
        raise AssertionError("checkpoint event does not report validation history binding")
    if '"stop_state": stop_state' not in source:
        raise AssertionError("checkpoint event does not report stop state binding")


@dataclass(frozen=True)
class FakeOffset:
    raw_tokens_seen: int
    causal_targets_seen: int
    samples_seen: int


class FakeCommon:
    @staticmethod
    def get_rng_state():
        return {"python": "p", "numpy": "n", "torch": "t", "cuda": "c"}


def court_checkpoint_payload_binds_latest_validation_and_stop_state(runner) -> None:
    history = [
        {"step": 2_900, "val_loss": 4.91, "val_ppl": 135.6, "val_targets": 98_112},
        {"step": 3_000, "val_loss": 4.88, "val_ppl": 131.6, "val_targets": 98_112},
    ]
    stop_state = {
        "stop_reason": "target_reached",
        "diminishing_decision": {"enabled": True, "should_stop": False},
    }
    payload = runner.checkpoint_payload(
        FakeCommon(),
        model_state={},
        model_root="model-root",
        optimizer_state={"state": {1: "adam"}},
        step=3_000,
        data_offset=FakeOffset(
            raw_tokens_seen=129_024_000,
            causal_targets_seen=128_772_000,
            samples_seen=252_000,
        ),
        scheduler={"type": "linear_warmup_then_constant"},
        manifest_roots={"train_manifest_sha256": "train"},
        ablation_contract={"ablation_id": "control"},
        best_val_loss=4.88,
        best_val_step=3_000,
        last_val_loss=4.88,
        validation_history=history,
        stop_state=stop_state,
    )
    assert payload["step"] == 3_000, payload
    assert payload["validation_history"] == history, payload
    assert payload["validation_history"][-1]["step"] == 3_000, payload
    assert payload["stop_state"] == stop_state, payload
    assert payload["best_val_step"] == 3_000, payload
    assert payload["last_val_loss"] == 4.88, payload


def court_checkpoint_identity_detects_mismatch(runner) -> None:
    base = runner.build_scheduler_state(
        policy="linear_warmup_then_constant",
        base_lr=1.5e-4,
        warmup_microbatches=2_000,
        grad_accum=7,
        total_optimizer_steps=6_990,
        min_lr_ratio=1.0,
    )
    changed = runner.build_scheduler_state(
        policy="cosine_decay",
        base_lr=1.5e-4,
        warmup_microbatches=2_000,
        grad_accum=7,
        total_optimizer_steps=6_990,
        min_lr_ratio=0.2,
    )
    if base == changed:
        raise AssertionError("scheduler mismatch would not be refused on resume")


def court_checkpoint_cadence_is_a_real_knob(runner) -> None:
    baseline = {
        "learning_rate": 1.5e-4,
        "warmup_microbatches": 2_000,
        "scheduler_policy": "linear_warmup_then_constant",
        "scheduler_min_lr_ratio": 1.0,
        "checkpoint_every": 500,
    }
    resolved = dict(baseline)
    resolved["checkpoint_every"] = 250
    assert runner.changed_knobs_from(baseline, resolved) == ["checkpoint_every"]


def court_promoted_full_run_requires_exact_manifest(runner) -> None:
    selected = {
        "learning_rate": 1e-4,
        "warmup_microbatches": 500,
        "scheduler_policy": "cosine_decay",
        "scheduler_min_lr_ratio": 0.1,
        "checkpoint_every": 250,
    }
    changed = ["learning_rate", "warmup_microbatches", "checkpoint_every", "scheduler"]
    manifest = {
        "schema": "helix.branch50.promotion-decision.v0",
        "status": "PROMOTED",
        "selected_knobs": selected,
        "changed_knobs": changed,
        "evidence_run_ids": ["control", "lr1e4", "warmup500", "cosine", "ckpt250"],
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
        court_default_scheduler_matches_existing_warmup_constant,
        court_cosine_scheduler_has_real_min_ratio_effect,
        court_full_corpus_remainder_is_bound,
        court_diminishing_stop_is_deterministic,
        court_diminishing_stop_is_not_promotional_pass,
        court_periodic_checkpoint_after_validation_source_order,
        court_checkpoint_payload_binds_latest_validation_and_stop_state,
        court_checkpoint_identity_detects_mismatch,
        court_checkpoint_cadence_is_a_real_knob,
        court_promoted_full_run_requires_exact_manifest,
    ]
    for court in courts:
        try:
            court(runner)
        except TypeError:
            court()
        print(f"{court.__name__}=PASS")
    print("BRANCH50_FULL_CAMPAIGN_CONTROL_COURTS=PASS")


if __name__ == "__main__":
    main()
