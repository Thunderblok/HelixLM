#!/usr/bin/env python3
"""Observer-only state summaries for the Helix automata-capacity experiment.

The probe deliberately has no model-facing output.  It converts detached hidden
states into a bounded, canonical record that can later be compared with a real
Thunderbit/BoltDigest implementation without changing training semantics.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch


SCHEMA = "helix.state-probe.v0"
REGISTER_COUNT = 32


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class StateProbeV0:
    schema: str
    source_start_token: int
    source_end_token: int
    source_token_root: str
    hidden_width: int
    register_count: int
    registers: tuple[float, ...]
    mean_l2: float
    max_abs: float
    state_bytes: int
    transition_root: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bounded_registers(hidden: torch.Tensor) -> tuple[float, ...]:
    width = hidden.shape[-1]
    if width % REGISTER_COUNT:
        raise ValueError(
            f"hidden width {width} must be divisible by {REGISTER_COUNT} registers"
        )
    summary = hidden.mean(dim=(0, 1)).reshape(REGISTER_COUNT, width // REGISTER_COUNT)
    return tuple(float(value) for value in summary.mean(dim=1).tolist())


def observe_segment(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    source_start_token: int,
) -> StateProbeV0:
    """Create one bounded record from a [batch, tokens, hidden] observation."""
    if hidden.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("hidden and input_ids must have shapes [B,T,D] and [B,T]")
    if hidden.shape[:2] != input_ids.shape:
        raise ValueError("hidden and input_ids batch/token dimensions must match")
    if hidden.shape[1] <= 0:
        raise ValueError("empty token spans cannot produce state")

    detached = hidden.detach().to(device="cpu", dtype=torch.float32).contiguous()
    token_copy = input_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if not torch.isfinite(detached).all():
        raise ValueError("non-finite hidden state is unavailable, not a valid probe")

    registers = _bounded_registers(detached)
    source_end_token = source_start_token + int(hidden.shape[1])
    source_root = sha256_bytes(token_copy.numpy().tobytes(order="C"))
    mean_l2 = float(torch.linalg.vector_norm(detached, dim=-1).mean())
    max_abs = float(detached.abs().max())
    body = {
        "schema": SCHEMA,
        "source_start_token": source_start_token,
        "source_end_token": source_end_token,
        "source_token_root": source_root,
        "hidden_width": int(hidden.shape[-1]),
        "register_count": REGISTER_COUNT,
        "registers": registers,
        "mean_l2": mean_l2,
        "max_abs": max_abs,
    }
    encoded = canonical_json(body)
    return StateProbeV0(
        **body,
        state_bytes=len(encoded),
        transition_root=sha256_bytes(encoded),
    )


def observe_hidden_sequence(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    segment_tokens: int = 64,
    source_start_token: int = 0,
) -> list[StateProbeV0]:
    if segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")
    records: list[StateProbeV0] = []
    for start in range(0, hidden.shape[1], segment_tokens):
        stop = min(start + segment_tokens, hidden.shape[1])
        records.append(
            observe_segment(
                hidden[:, start:stop],
                input_ids[:, start:stop],
                source_start_token=source_start_token + start,
            )
        )
    return records


def compression_accounting(*, raw_history_bytes: int, records: list[StateProbeV0]) -> dict[str, float | int]:
    """Separate the current state from the append-only observation history.

    A state machine may carry only its latest state at inference time, while a
    lineage/DAG system may retain every transition.  Reporting both prevents
    the durable history from disappearing inside a flattering ratio.
    """
    live_state_bytes = records[-1].state_bytes if records else 0
    transition_log_bytes = sum(record.state_bytes for record in records)
    ratio = raw_history_bytes / live_state_bytes if live_state_bytes else math.inf
    return {
        "raw_history_bytes": raw_history_bytes,
        "candidate_live_state_bytes": live_state_bytes,
        "transition_log_bytes": transition_log_bytes,
        "state_compression_ratio": ratio,
    }
