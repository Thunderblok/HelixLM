#!/usr/bin/env python3
"""Hostile controls for exact epoch boundaries in the Branch55 runner."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


RUNNER = Path(__file__).with_name("run_branch55_multiepoch_ablation.py")
SPEC = importlib.util.spec_from_file_location("branch55_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


@dataclass(frozen=True)
class Offset:
    samples_seen: int


def rows(count: int):
    for sample in range(1, count + 1):
        yield torch.tensor([sample, sample + 1]), Offset(samples_seen=sample)


def court_epoch_boundaries_are_never_mixed() -> None:
    groups = list(
        runner.iter_accumulation_groups(
            rows(20),
            batch_size=3,
            grad_accum=2,
            allow_partial_batch=True,
            allow_partial_accumulation=True,
            samples_per_epoch=10,
        )
    )
    assert [group[-1][1].samples_seen for group in groups] == [6, 10, 16, 20]
    for group in groups:
        sample_ids = [
            int(value)
            for batch, _ in group
            for value in batch["input_ids"][:, 0].tolist()
        ]
        assert (min(sample_ids) - 1) // 10 == (max(sample_ids) - 1) // 10


def court_each_epoch_retains_every_sample() -> None:
    groups = list(
        runner.iter_accumulation_groups(
            rows(22),
            batch_size=4,
            grad_accum=2,
            allow_partial_batch=True,
            allow_partial_accumulation=True,
            samples_per_epoch=11,
        )
    )
    observed = [
        int(value)
        for group in groups
        for batch, _ in group
        for value in batch["input_ids"][:, 0].tolist()
    ]
    assert observed == list(range(1, 23))
    assert [group[-1][1].samples_seen for group in groups] == [8, 11, 19, 22]


def main() -> None:
    court_epoch_boundaries_are_never_mixed()
    court_each_epoch_retains_every_sample()
    print("BRANCH55_MULTIEPOCH_COURTS=PASS")


if __name__ == "__main__":
    main()
