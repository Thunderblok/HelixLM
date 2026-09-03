#!/usr/bin/env python3
"""Project truthful compatibility metrics from a local MLflow JSONL spool.

The training process remains authoritative for model state and its original
metrics.  This companion reads only completed JSONL records and adds aliases or
derived counters whose semantics are exactly recoverable from the run contract.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ProjectionContract:
    seq_len: int
    aligned_causal_targets: int

    def __post_init__(self) -> None:
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if self.aligned_causal_targets <= 0:
            raise ValueError("aligned_causal_targets must be positive")


def project_event(event: dict[str, Any], contract: ProjectionContract) -> dict[str, float]:
    """Return only aliases and counters whose values are exactly recoverable."""
    if event.get("event") != "metrics":
        return {}
    metrics = event.get("metrics")
    if not isinstance(metrics, dict):
        return {}

    projected: dict[str, float] = {}
    if "train/loss" in metrics:
        projected["train/accum_loss"] = float(metrics["train/loss"])
    if "train/ppl" in metrics:
        projected["train/accum_ppl"] = float(metrics["train/ppl"])
    if "val/loss" in metrics:
        projected["val_loss"] = float(metrics["val/loss"])
    if "val/ppl" in metrics:
        projected["val_ppl"] = float(metrics["val/ppl"])

    causal = metrics.get("train/causal_targets_seen")
    causal_rate = metrics.get("train/causal_targets_per_second")
    if causal is not None:
        causal_value = float(causal)
        sequences = causal_value / (contract.seq_len - 1)
        raw_tokens = sequences * contract.seq_len
        projected.update(
            {
                "train/raw_tokens_seen": raw_tokens,
                "train/sequences_seen": sequences,
                "train/progress_fraction": causal_value / contract.aligned_causal_targets,
            }
        )
    if causal_rate is not None:
        projected["train/raw_tokens_per_second"] = (
            float(causal_rate) * contract.seq_len / (contract.seq_len - 1)
        )

    step = event.get("step")
    if isinstance(step, int) and step >= 0:
        projected["train/optimizer_steps_completed"] = float(step)

    return {key: value for key, value in projected.items() if math.isfinite(value)}


def read_complete_events(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Read complete newline-terminated records, retaining a partial tail."""
    events: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        handle.seek(offset)
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                return events, handle.tell()
            if not line.endswith(b"\n"):
                return events, line_start
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL record at byte {line_start}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"non-object JSONL record at byte {line_start}")
            events.append(event)


def metric_entities(events: Iterable[dict[str, Any]], contract: ProjectionContract):
    from mlflow.entities import Metric

    result = []
    for event in events:
        step = event.get("step")
        if not isinstance(step, int):
            continue
        timestamp = int(float(event.get("ts", time.time())) * 1000)
        for key, value in project_event(event, contract).items():
            result.append(Metric(key=key, value=value, timestamp=timestamp, step=step))
    return result


def write_state(path: Path, *, offset: int) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"offset": offset}, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_offset(path: Path) -> int:
    if not path.exists():
        return 0
    value = json.loads(path.read_text(encoding="utf-8"))
    return int(value.get("offset", 0))


def project_once(*, client: Any, run_id: str, spool: Path, state: Path, contract: ProjectionContract) -> int:
    offset = load_offset(state)
    events, next_offset = read_complete_events(spool, offset)
    entities = metric_entities(events, contract)
    for start in range(0, len(entities), 1000):
        client.log_batch(run_id, metrics=entities[start : start + 1000])
    if next_offset != offset:
        write_state(state, offset=next_offset)
    return len(entities)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--aligned-causal-targets", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    contract = ProjectionContract(args.seq_len, args.aligned_causal_targets)
    client.set_tag(args.run_id, "metric_compat_projection", "helix_branch49_v0")
    client.set_tag(args.run_id, "metric_compat_source", "local_jsonl_spool")
    client.set_tag(args.run_id, "metric_compat_train_loss", "accumulated_only")
    client.set_tag(args.run_id, "metric_compat_immediate_loss", "unavailable_not_reconstructed")

    while True:
        count = project_once(
            client=client,
            run_id=args.run_id,
            spool=args.spool,
            state=args.state,
            contract=contract,
        )
        print(json.dumps({"projected_metrics": count, "state": str(args.state)}), flush=True)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
