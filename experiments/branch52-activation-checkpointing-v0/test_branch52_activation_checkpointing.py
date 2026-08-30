#!/usr/bin/env python3
"""Branch52 courts for real recurrent activation checkpointing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

PROGRAM = Path(__file__).resolve().parent
REPO = PROGRAM.parents[1]
sys.path.insert(0, str(REPO))

from helix_lm.config import HelixConfig
from helix_lm.hf_model import HelixForCausalLM


RUNNER = (
    REPO
    / "experiments"
    / "branch51-quality-vram-v0"
    / "run_branch51_quality_vram_ablation.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("branch52_runner_under_test", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_config() -> HelixConfig:
    return HelixConfig(
        vocab_size=64,
        d_model=32,
        n_columns=1,
        nodes_per_column=(2,),
        n_heads=4,
        n_loops=2,
        seq_len=8,
        attention_mode="full",
        use_ssm=False,
        use_titans_memory=False,
        use_rope=False,
        dropout=0.0,
        attn_dropout=0.0,
        tie_word_embeddings=True,
        grad_buffer_ratio=0.0,
        seed=42,
    )


def court_runner_treats_checkpointing_as_one_factor() -> None:
    runner = load_runner()
    baseline = {
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
        "activation_checkpointing": False,
    }
    candidate = {**baseline, "activation_checkpointing": True}
    assert runner.changed_knobs_from(baseline, candidate) == [
        "activation_checkpointing"
    ]


def court_checkpointing_recomputes_and_preserves_gradients() -> None:
    torch.manual_seed(42)
    baseline = HelixForCausalLM(tiny_config())
    candidate = HelixForCausalLM(tiny_config())
    candidate.load_state_dict(baseline.state_dict())
    candidate.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    input_ids = torch.randint(0, 64, (2, 8))
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)

    baseline.train()
    candidate.train()
    baseline_loss = baseline(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    ).loss
    candidate_loss = candidate(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    ).loss
    assert baseline_loss is not None
    assert candidate_loss is not None
    baseline_loss.backward()
    candidate_loss.backward()

    torch.testing.assert_close(candidate_loss, baseline_loss, rtol=1e-5, atol=1e-6)
    baseline_grads = dict(baseline.named_parameters())
    candidate_grads = dict(candidate.named_parameters())
    assert baseline_grads.keys() == candidate_grads.keys()
    for name, baseline_parameter in baseline_grads.items():
        candidate_parameter = candidate_grads[name]
        if baseline_parameter.grad is None or candidate_parameter.grad is None:
            assert baseline_parameter.grad is None, name
            assert candidate_parameter.grad is None, name
            continue
        torch.testing.assert_close(
            candidate_parameter.grad,
            baseline_parameter.grad,
            rtol=2e-4,
            atol=2e-5,
            msg=lambda message, name=name: f"{name}: {message}",
        )

    assert candidate.gradient_checkpointing is True
    assert candidate._gradient_checkpoint_forward_calls == 1
    assert candidate._gradient_checkpoint_function_calls == 2

    candidate.eval()
    with torch.no_grad():
        candidate(input_ids=input_ids, attention_mask=attention_mask)
    assert candidate._gradient_checkpoint_forward_calls == 1
    assert candidate._gradient_checkpoint_function_calls == 2


def court_disabled_checkpointing_does_not_claim_execution() -> None:
    model = HelixForCausalLM(tiny_config())
    model.train()
    input_ids = torch.randint(0, 64, (1, 8))
    loss = model(input_ids=input_ids, labels=input_ids).loss
    assert loss is not None
    loss.backward()
    assert model.gradient_checkpointing is False
    assert model._gradient_checkpoint_forward_calls == 0
    assert model._gradient_checkpoint_function_calls == 0


def main() -> None:
    court_runner_treats_checkpointing_as_one_factor()
    court_checkpointing_recomputes_and_preserves_gradients()
    court_disabled_checkpointing_does_not_claim_execution()
    print("BRANCH52_ACTIVATION_CHECKPOINTING_COURTS=PASS")


if __name__ == "__main__":
    main()
