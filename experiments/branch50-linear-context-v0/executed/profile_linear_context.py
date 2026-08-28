#!/usr/bin/env python3
"""Measure Branch-50 context scaling with one real optimizer step per context.

The Helix source tree is read-only evidence for this harness.  The model shape,
attention geometry, optimizer, dtype, and seed stay fixed; only ``seq_len``
changes.  FLOPs cover forward + backward.  Dynamic VRAM is measured above the
resident model and initialized optimizer state so the context-dependent slope
is not hidden by the constant parameter/optimizer footprint.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
EXPECTED_SOURCE_HEAD = "03d0698dd3365c81695d9ed8d4568d35d6044fbb"
EXPECTED_SOURCE_TREE = "745c042db9860bca4cdfa180543f8a60a769c936"
EXPECTED_PARAMETER_COUNT = 53_592_340
GPT2_SPECIAL_ID = 50_256


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE), *args], text=True
    ).strip()


def verify_source() -> dict[str, str]:
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    dirty = git("status", "--porcelain")
    if head != EXPECTED_SOURCE_HEAD or tree != EXPECTED_SOURCE_TREE or dirty:
        raise SystemExit(
            "REFUSED: Branch-50 source drifted: "
            f"head={head} tree={tree} dirty={bool(dirty)}"
        )
    return {"head": head, "tree": tree, "dirty": "false"}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_fit(xs: list[float], ys: list[float]) -> dict[str, float]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.square(y - predicted).sum())
    centered = float(np.square(y - y.mean()).sum())
    r2 = 1.0 if centered == 0.0 and residual == 0.0 else 1.0 - residual / centered
    max_relative_error = float(
        np.max(np.abs(y - predicted) / np.maximum(np.abs(y), 1.0))
    )
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "max_relative_error": max_relative_error,
    }


def build_config(seq_len: int, batch_size: int, seed: int):
    from helix_lm.config import HelixConfig

    return HelixConfig.small_v2(
        vocab_size=50_257,
        d_model=512,
        n_heads=8,
        n_loops=3,
        seq_len=seq_len,
        batch_size=batch_size,
        n_columns=3,
        nodes_per_column=(2, 3, 2),
        attention_mode="multi_scale_windowed",
        local_window=64,
        coarse_window=128,
        compressed_windows=8,
        compressed_views=8,
        consensus_type="cosine",
        corrector_type="ffn",
        use_titans_memory=False,
        use_ssm=False,
        use_cca=False,
        strict_nan_check=True,
        dtype="float32",
        amp_dtype="bfloat16",
        dropout=0.05,
        attn_dropout=0.05,
        ffn_expansion=2.5,
        lr=1.5e-4,
        warmup_steps=2_000,
        weight_decay=0.05,
        grad_clip=1.0,
        tokenizer_name="gpt2",
        pad_token_id=GPT2_SPECIAL_ID,
        eos_token_id=GPT2_SPECIAL_ID,
        bos_token_id=GPT2_SPECIAL_ID,
        tie_word_embeddings=True,
        grad_buffer_ratio=0.0,
        architectures=["HelixForCausalLM"],
        seed=seed,
    )


def finite_gradients(model: torch.nn.Module) -> tuple[bool, int, int]:
    present = 0
    nonfinite = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        present += parameter.numel()
        nonfinite += int((~torch.isfinite(parameter.grad)).sum().item())
    return present > 0 and nonfinite == 0, present, nonfinite


def execute_step(
    *, seq_len: int, batch_size: int, seed: int, warmup: bool
) -> dict[str, Any]:
    from helix_lm.hf_model import HelixForCausalLM

    seed_all(seed)
    cfg = build_config(seq_len, batch_size, seed)
    model = HelixForCausalLM(cfg).to(device="cuda", dtype=torch.float32)
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count != EXPECTED_PARAMETER_COUNT or trainable_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            "REFUSED: parameter drift: "
            f"total={parameter_count} trainable={trainable_count}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-4, weight_decay=0.05
    )
    model.train()

    def tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = torch.randint(
            0, cfg.vocab_size, (batch_size, seq_len), device="cuda"
        )
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)
        return input_ids, labels, attention_mask

    # Initialize optimizer state and CUDA kernels outside the measured step.
    if warmup:
        input_ids, labels, attention_mask = tensors()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        del input_ids, labels, attention_mask, output

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident_allocated = torch.cuda.memory_allocated()
    resident_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    input_ids, labels, attention_mask = tensors()
    start = time.perf_counter()
    with FlopCounterMode(display=False) as flop_counter:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
        loss = output.loss
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"REFUSED: nonfinite loss at seq_len={seq_len}")
        loss.backward()
    grad_ok, gradient_values, gradient_nonfinite = finite_gradients(model)
    if not grad_ok:
        raise RuntimeError(
            "REFUSED: invalid gradients: "
            f"present={gradient_values} nonfinite={gradient_nonfinite}"
        )
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm).item():
        raise RuntimeError("REFUSED: nonfinite gradient norm")
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    result = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "tokens": seq_len * batch_size,
        "loss": float(loss.detach().cpu()),
        "gradient_norm_pre_clip": float(grad_norm.detach().cpu()),
        "gradient_values": gradient_values,
        "gradient_nonfinite": gradient_nonfinite,
        "forward_backward_flops": int(flop_counter.get_total_flops()),
        "elapsed_seconds": elapsed,
        "tokens_per_second": seq_len * batch_size / elapsed,
        "resident_allocated_bytes": resident_allocated,
        "resident_reserved_bytes": resident_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "dynamic_peak_allocated_bytes": peak_allocated - resident_allocated,
        "dynamic_peak_reserved_bytes": peak_reserved - resident_reserved,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "strict_nan_check": True,
        "master_dtype": "float32",
        "amp_dtype": "bfloat16",
        "local_window": 64,
        "coarse_window": 128,
        "compressed_windows": 8,
        "compressed_views": 8,
        "n_loops": 3,
    }
    del input_ids, labels, attention_mask, output, loss, model, optimizer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "linear-context-court.json")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = verify_source()
    if not torch.cuda.is_available():
        raise SystemExit("REFUSED: CUDA unavailable")
    if torch.cuda.get_device_capability() != (12, 0):
        raise SystemExit(
            f"REFUSED: expected sm_120, got {torch.cuda.get_device_capability()}"
        )
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("REFUSED: BF16 unsupported")
    if sorted(set(args.contexts)) != args.contexts or len(args.contexts) < 3:
        raise SystemExit("REFUSED: contexts must contain at least three unique ascending values")
    if min(args.contexts) < 128 or any(context % 128 for context in args.contexts):
        raise SystemExit("REFUSED: contexts must be positive multiples of coarse_window=128")

    sys.path.insert(0, str(SOURCE))
    rows = [
        execute_step(
            seq_len=context,
            batch_size=args.batch_size,
            seed=args.seed,
            warmup=not args.no_warmup,
        )
        for context in args.contexts
    ]
    contexts = [float(row["seq_len"]) for row in rows]
    flops = [float(row["forward_backward_flops"]) for row in rows]
    memory = [float(row["dynamic_peak_allocated_bytes"]) for row in rows]
    flop_fit = linear_fit(contexts, flops)
    memory_fit = linear_fit(contexts, memory)
    admitted = (
        flop_fit["r2"] >= 0.995
        and flop_fit["max_relative_error"] <= 0.05
        and memory_fit["r2"] >= 0.98
        and memory_fit["max_relative_error"] <= 0.10
        and all(
            row["peak_reserved_bytes"]
            <= 0.85 * torch.cuda.get_device_properties(0).total_memory
            for row in rows
        )
    )
    packet = {
        "court": "HELIX_BRANCH50_LINEAR_CONTEXT_V0",
        "source": source,
        "device": {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "fixed_geometry": {
            "d_model": 512,
            "n_heads": 8,
            "n_loops": 3,
            "local_window": 64,
            "coarse_window": 128,
            "compressed_windows": 8,
            "compressed_views": 8,
            "consensus_type": "cosine",
            "corrector_type": "ffn",
            "use_ssm": False,
            "use_titans_memory": False,
            "use_cca": False,
            "batch_size": args.batch_size,
        },
        "rows": rows,
        "forward_backward_flop_fit": flop_fit,
        "dynamic_peak_memory_fit": memory_fit,
        "admission": "PASS" if admitted else "HOLD",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))
    if not admitted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
