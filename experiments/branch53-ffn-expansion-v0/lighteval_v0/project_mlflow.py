#!/usr/bin/env python3
"""Project a completed paired Lighteval packet into MLflow.

The local packet remains authoritative. This script refuses to run until the
paired comparison and packet manifest already exist, then mirrors identities,
metrics, and artifacts to an MLflow tracking server through the REST API.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compare_results import compare
from contract import canonical_json, contract_root, load_contract, sha256_bytes


ROOT = Path(__file__).resolve().parent
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "helix-lighteval-paired-v0"
ARTIFACT_FILES = (
    "paired_comparison.json",
    "MANIFEST.sha256",
    "logs/comparison.log",
)


class ProjectionError(RuntimeError):
    """The local packet could not be projected to MLflow."""


@dataclass(frozen=True)
class Projection:
    tracking_uri: str
    experiment_id: str
    parent_run_id: str
    checkpoint_run_ids: dict[str, str]
    artifact_count: int


class MlflowRestClient:
    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri.rstrip("/")

    def get_experiment_by_name(self, name: str) -> str | None:
        query = urllib.parse.urlencode({"experiment_name": name})
        try:
            response = self._json_request("GET", f"/api/2.0/mlflow/experiments/get-by-name?{query}")
        except ProjectionError as exc:
            if "RESOURCE_DOES_NOT_EXIST" in str(exc) or "404" in str(exc):
                return None
            raise
        experiment = response.get("experiment")
        if not isinstance(experiment, dict):
            return None
        experiment_id = experiment.get("experiment_id")
        return str(experiment_id) if experiment_id is not None else None

    def create_experiment(self, name: str) -> str:
        response = self._json_request("POST", "/api/2.0/mlflow/experiments/create", {"name": name})
        experiment_id = response.get("experiment_id")
        if experiment_id is None:
            raise ProjectionError("MLflow create_experiment response omitted experiment_id")
        return str(experiment_id)

    def create_run(self, experiment_id: str, run_name: str, tags: dict[str, str]) -> str:
        tag_rows = [{"key": "mlflow.runName", "value": run_name}]
        tag_rows.extend({"key": key, "value": value} for key, value in sorted(tags.items()))
        response = self._json_request(
            "POST",
            "/api/2.0/mlflow/runs/create",
            {
                "experiment_id": experiment_id,
                "start_time": _millis(),
                "tags": tag_rows,
            },
        )
        info = response.get("run", {}).get("info", {})
        run_id = info.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ProjectionError("MLflow create_run response omitted run_id")
        return run_id

    def log_batch(
        self,
        run_id: str,
        *,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str],
    ) -> None:
        timestamp = _millis()
        self._json_request(
            "POST",
            "/api/2.0/mlflow/runs/log-batch",
            {
                "run_id": run_id,
                "params": [
                    {"key": key, "value": _param_value(value)}
                    for key, value in sorted(params.items())
                ],
                "metrics": [
                    {"key": key, "value": float(value), "timestamp": timestamp, "step": 0}
                    for key, value in sorted(metrics.items())
                ],
                "tags": [
                    {"key": key, "value": value}
                    for key, value in sorted(tags.items())
                ],
            },
        )

    def upload_artifact(self, run_id: str, local_path: Path, artifact_path: str) -> None:
        encoded_path = "/".join(urllib.parse.quote(part) for part in artifact_path.split("/"))
        endpoint = f"/api/2.0/mlflow-artifacts/artifacts/{run_id}/artifacts/{encoded_path}"
        self._raw_request("PUT", endpoint, local_path.read_bytes(), "application/octet-stream")

    def finish_run(self, run_id: str, status: str = "FINISHED") -> None:
        self._json_request(
            "POST",
            "/api/2.0/mlflow/runs/update",
            {"run_id": run_id, "status": status, "end_time": _millis()},
        )

    def _json_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if body is None else canonical_json(body)
        response = self._raw_request(method, path, data, "application/json")
        if not response:
            return {}
        try:
            decoded = json.loads(response.decode())
        except json.JSONDecodeError as exc:
            raise ProjectionError(f"MLflow returned non-JSON response for {path}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ProjectionError(f"MLflow returned non-object JSON response for {path}")
        return decoded

    def _raw_request(
        self,
        method: str,
        path: str,
        data: bytes | None,
        content_type: str,
    ) -> bytes:
        request = urllib.request.Request(
            f"{self.tracking_uri}{path}",
            data=data,
            method=method,
            headers={"Content-Type": content_type},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise ProjectionError(f"MLflow HTTP {exc.code} for {method} {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProjectionError(f"MLflow unavailable for {method} {path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    return parser.parse_args()


def project_pair(
    pair_root: Path,
    *,
    tracking_uri: str,
    experiment_name: str,
    client: MlflowRestClient | None = None,
) -> Projection:
    pair_root = pair_root.resolve()
    comparison_path = pair_root / "paired_comparison.json"
    manifest_path = pair_root / "MANIFEST.sha256"
    receipt_path = pair_root / "projections" / "mlflow_projection_receipt.json"
    if not comparison_path.is_file():
        raise ProjectionError(f"paired comparison unavailable: {comparison_path}")
    if not manifest_path.is_file():
        raise ProjectionError(f"packet manifest unavailable: {manifest_path}")
    if receipt_path.exists():
        raise ProjectionError(f"projection receipt already exists: {receipt_path}")

    contract = load_contract(ROOT / "evaluation_contract.json")
    expected_root = contract_root(contract)
    try:
        comparison = json.loads(comparison_path.read_text())
    except json.JSONDecodeError as exc:
        raise ProjectionError(f"paired comparison is malformed JSON: {exc}") from exc
    recomputed = compare(pair_root)
    if comparison != recomputed:
        raise ProjectionError("paired comparison drifted from current scoring contract")
    if comparison.get("status") != "complete" or comparison.get("contract_root") != expected_root:
        raise ProjectionError("paired comparison is not complete under the frozen contract")

    client = client or MlflowRestClient(tracking_uri)
    experiment_id = client.get_experiment_by_name(experiment_name) or client.create_experiment(
        experiment_name
    )
    parent_run_id = client.create_run(
        experiment_id,
        run_name=f"paired-{expected_root[:8]}",
        tags={
            "run_kind": "lighteval_paired_comparison",
            "projection": "post_run_mlflow_rest_v0",
            "local_packet_authoritative": "true",
            "production_effect": "none",
            "contract_root": expected_root,
        },
    )

    child_run_ids: dict[str, str] = {}
    artifact_count = 0
    try:
        client.log_batch(
            parent_run_id,
            params=_parent_params(contract, pair_root),
            metrics=_parent_metrics(comparison),
            tags={"promotion_decision": str(comparison["promotion_decision"])},
        )

        for checkpoint, checkpoint_comparison in zip(
            contract["checkpoints"], comparison["checkpoints"], strict=True
        ):
            run_id = client.create_run(
                experiment_id,
                run_name=checkpoint["id"],
                tags={
                    "run_kind": "lighteval_checkpoint_result",
                    "mlflow.parentRunId": parent_run_id,
                    "checkpoint_id": checkpoint["id"],
                    "contract_root": expected_root,
                    "production_effect": "none",
                },
            )
            child_run_ids[checkpoint["id"]] = run_id
            client.log_batch(
                run_id,
                params=_checkpoint_params(checkpoint, checkpoint_comparison),
                metrics=_checkpoint_metrics(checkpoint_comparison),
                tags={"paired_parent_run_id": parent_run_id},
            )

        for path, artifact_path in _artifact_paths(pair_root, contract):
            client.upload_artifact(parent_run_id, path, artifact_path)
            artifact_count += 1
        for run_id in child_run_ids.values():
            client.finish_run(run_id)
        client.finish_run(parent_run_id)
    except Exception:
        for run_id in child_run_ids.values():
            _finish_best_effort(client, run_id, "FAILED")
        _finish_best_effort(client, parent_run_id, "FAILED")
        raise

    projection = Projection(
        tracking_uri=tracking_uri,
        experiment_id=experiment_id,
        parent_run_id=parent_run_id,
        checkpoint_run_ids=child_run_ids,
        artifact_count=artifact_count,
    )
    _write_receipt(pair_root, projection, expected_root)
    return projection


def _parent_params(contract: dict[str, Any], pair_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "helix.lighteval.mlflow-projection.v0",
        "contract_root": contract_root(contract),
        "pair_root": str(pair_root),
        "evaluator": contract["evaluator"]["name"],
        "evaluator_version": contract["evaluator"]["version"],
        "python": contract["evaluator"]["python"],
        "model_adapter_kind": contract["model_adapter"]["kind"],
        "model_adapter_registration": contract["model_adapter"]["registration_module"],
        "scoring_aggregate": contract["scoring"]["aggregate"],
        "winner_rule": contract["scoring"]["winner_rule"],
        "task_count": len(contract["tasks"]),
        "checkpoint_count": len(contract["checkpoints"]),
        "tasks": ",".join(task["name"] for task in contract["tasks"]),
        "checkpoint_ids": ",".join(checkpoint["id"] for checkpoint in contract["checkpoints"]),
    }


def _parent_metrics(comparison: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "macro_delta_candidate_minus_control": float(
            comparison["macro_delta_candidate_minus_control"]
        )
    }
    for name, delta in comparison["task_deltas_candidate_minus_control"].items():
        metrics[f"delta/{name}"] = float(delta)
    return metrics


def _checkpoint_params(
    checkpoint: dict[str, Any],
    checkpoint_comparison: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_id": checkpoint["id"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "model_root_sha256": checkpoint["model_root_sha256"],
        "export_manifest_root_sha256": checkpoint["export_manifest_root_sha256"],
        "execution_receipt_sha256": checkpoint_comparison["execution_receipt_sha256"],
        "results_sha256": checkpoint_comparison["results_sha256"],
        "ffn_expansion": checkpoint["ffn_expansion"],
        "parameter_count": checkpoint["parameter_count"],
    }


def _checkpoint_metrics(checkpoint_comparison: dict[str, Any]) -> dict[str, float]:
    metrics = {"macro_mean": float(checkpoint_comparison["macro_mean"])}
    for name, score in checkpoint_comparison["scores"].items():
        metrics[f"score/{name}"] = float(score)
    return metrics


def _artifact_paths(pair_root: Path, contract: dict[str, Any]) -> list[tuple[Path, str]]:
    paths: list[tuple[Path, str]] = []
    for relative in ARTIFACT_FILES:
        path = pair_root / relative
        if path.is_file():
            paths.append((path, relative))
    for checkpoint in contract["checkpoints"]:
        checkpoint_id = checkpoint["id"]
        for filename in ("execution_receipt.json", "results.json"):
            path = pair_root / checkpoint_id / filename
            if not path.is_file():
                raise ProjectionError(f"checkpoint artifact unavailable: {path}")
            paths.append((path, f"{checkpoint_id}/{filename}"))
        log_path = pair_root / "logs" / f"{checkpoint_id}.log"
        if log_path.is_file():
            paths.append((log_path, f"logs/{checkpoint_id}.log"))
    return paths


def _write_receipt(pair_root: Path, projection: Projection, expected_root: str) -> None:
    receipt_dir = pair_root / "projections"
    receipt_dir.mkdir(exist_ok=True)
    receipt_path = receipt_dir / "mlflow_projection_receipt.json"
    if receipt_path.exists():
        raise ProjectionError(f"projection receipt already exists: {receipt_path}")
    payload = {
        "schema_version": "helix.lighteval.mlflow-projection-receipt.v0",
        "status": "complete",
        "contract_root": expected_root,
        "tracking_uri": projection.tracking_uri,
        "experiment_id": projection.experiment_id,
        "parent_run_id": projection.parent_run_id,
        "checkpoint_run_ids": projection.checkpoint_run_ids,
        "artifact_count": projection.artifact_count,
        "production_effect": "none",
    }
    receipt_path.write_bytes(canonical_json(payload))


def _finish_best_effort(client: MlflowRestClient, run_id: str, status: str) -> None:
    try:
        client.finish_run(run_id, status)
    except Exception:
        pass


def _param_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _millis() -> int:
    return int(time.time() * 1000)


def main() -> None:
    args = parse_args()
    try:
        projection = project_pair(
            args.pair_root,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
        )
    except ProjectionError as exc:
        raise SystemExit(f"LIGHTEVAL_MLFLOW_PROJECTION=UNAVAILABLE\nREASON={exc}") from exc
    receipt_path = args.pair_root / "projections" / "mlflow_projection_receipt.json"
    print(f"LIGHTEVAL_MLFLOW_PARENT_RUN_ID={projection.parent_run_id}")
    print(f"LIGHTEVAL_MLFLOW_ARTIFACTS={projection.artifact_count}")
    print(f"LIGHTEVAL_MLFLOW_PROJECTION_RECEIPT_SHA256={sha256_bytes(receipt_path.read_bytes())}")
    print("LIGHTEVAL_MLFLOW_PROJECTION=PASS")


if __name__ == "__main__":
    main()
