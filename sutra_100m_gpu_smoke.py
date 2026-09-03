#!/usr/bin/env python3
"""One strict BF16 forward/backward court for the frozen Sutra baseline shape."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from automata_state_probe import compression_accounting, observe_hidden_sequence
from helix_lm.hf_model import HelixForCausalLM
from sutra_100m_preflight import EXPECTED_PARAMETER_COUNT, SEQ_LEN, build_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("UNAVAILABLE: CUDA is not available")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("UNAVAILABLE: CUDA BF16 is not supported")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    config = build_config(batch_size=1)
    config.memory_efficient_forward = True
    model = HelixForCausalLM(config).to(device)
    if model.count_parameters()["total"] != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("parameter count drift")

    input_ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN), device=device)
    attention_mask = torch.ones_like(input_ids)
    started = time.perf_counter()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("non-finite loss")
    if not isinstance(output.hidden_states, torch.Tensor):
        raise RuntimeError("hidden-state observation unavailable")
    probes = observe_hidden_sequence(output.hidden_states, input_ids, segment_tokens=64)
    output.loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    terminal = {
        "schema": "helix.sutra-100m-gpu-smoke.v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "amp_dtype": "bfloat16",
        "master_dtype": "float32",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "batch_size": 1,
        "sequence_length": SEQ_LEN,
        "causal_targets": SEQ_LEN - 1,
        "loss": float(output.loss.detach().cpu()),
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "state_probe_records": len(probes),
        "state_probe_first_root": probes[0].transition_root,
        "state_probe_accounting": compression_accounting(
            raw_history_bytes=input_ids.numel() * input_ids.element_size(), records=probes
        ),
        "state_probe_posture": "detached_observer_only_no_model_feedback",
        "training_started": False,
        "production_effect": "none",
        "terminal": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(terminal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
