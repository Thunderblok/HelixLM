#!/usr/bin/env python3
"""Static courts for the Branch-50 overnight supervisor."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_branch50_overnight_campaign.py")
SPEC = importlib.util.spec_from_file_location("branch50_overnight", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def court_knob_args_are_complete_and_stable() -> None:
    selected = {
        "learning_rate": 1e-4,
        "warmup_microbatches": 500,
        "scheduler_policy": "cosine_decay",
        "scheduler_min_lr_ratio": 0.1,
        "checkpoint_every": 250,
        "weight_decay": 0.05,
        "grad_clip": 1.0,
        "dropout": 0.05,
        "attention_dropout": 0.05,
        "ffn_expansion": 2.5,
    }
    args = MODULE.knob_args({"selected_knobs": selected})
    expected_flags = {
        "--learning-rate",
        "--warmup-microbatches",
        "--scheduler-policy",
        "--scheduler-min-lr-ratio",
        "--checkpoint-every",
        "--weight-decay",
        "--grad-clip",
        "--dropout",
        "--attention-dropout",
        "--ffn-expansion",
    }
    assert set(args[::2]) == expected_flags
    assert len(args) == 2 * len(expected_flags)


def court_full_stage_is_after_live_and_bounded_trials() -> None:
    source = MODULE_PATH.read_text()
    ordered_fragments = [
        '"live_campaign_courts"',
        '"operational_control_100m"',
        '"scheduler_cosine_100m"',
        '"checkpoint_cadence_100m"',
        '"combined_pilot_100m"',
        '"full_corpus"',
    ]
    offsets = [source.index(fragment) for fragment in ordered_fragments]
    assert offsets == sorted(offsets), offsets
    assert 'accepted_statuses=("PASS", "STOPPED_DIMINISHING_RETURN")' in source


def main() -> None:
    courts = [
        court_knob_args_are_complete_and_stable,
        court_full_stage_is_after_live_and_bounded_trials,
    ]
    for court in courts:
        court()
        print(f"{court.__name__}=PASS")
    print("BRANCH50_OVERNIGHT_CAMPAIGN_STATIC_COURTS=PASS")


if __name__ == "__main__":
    main()
