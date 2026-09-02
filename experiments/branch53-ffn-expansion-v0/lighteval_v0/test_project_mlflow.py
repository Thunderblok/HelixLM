#!/usr/bin/env python3
"""Courts for post-run MLflow projection.

These tests never contact a live MLflow server. They validate the local packet
preconditions and the REST-shaped payloads through a recording fake client.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import project_mlflow
from contract import canonical_json
from test_compare_results import write_pair


class RecordingMlflowClient:
    def __init__(self) -> None:
        self.created_experiments: list[str] = []
        self.created_runs: list[tuple[str, str, dict[str, str]]] = []
        self.logged_batches: list[tuple[str, dict[str, Any], dict[str, float], dict[str, str]]] = []
        self.uploaded_artifacts: list[tuple[str, str]] = []
        self.finished_runs: list[tuple[str, str]] = []

    def get_experiment_by_name(self, name: str) -> str | None:
        return None

    def create_experiment(self, name: str) -> str:
        self.created_experiments.append(name)
        return "experiment-1"

    def create_run(self, experiment_id: str, run_name: str, tags: dict[str, str]) -> str:
        run_id = f"run-{len(self.created_runs) + 1}"
        self.created_runs.append((experiment_id, run_name, tags))
        return run_id

    def log_batch(
        self,
        run_id: str,
        *,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str],
    ) -> None:
        self.logged_batches.append((run_id, params, metrics, tags))

    def upload_artifact(self, run_id: str, local_path: Path, artifact_path: str) -> None:
        assert local_path.is_file()
        self.uploaded_artifacts.append((run_id, artifact_path))

    def finish_run(self, run_id: str, status: str = "FINISHED") -> None:
        self.finished_runs.append((run_id, status))


def prepare_complete_pair(pair_root: Path) -> None:
    write_pair(pair_root)
    comparison = project_mlflow.compare(pair_root)
    (pair_root / "paired_comparison.json").write_bytes(canonical_json(comparison))
    (pair_root / "logs").mkdir(exist_ok=True)
    (pair_root / "logs" / "comparison.log").write_text("comparison pass\n")
    for checkpoint in comparison["checkpoints"]:
        checkpoint_id = checkpoint["id"]
        (pair_root / "logs" / f"{checkpoint_id}.log").write_text(f"{checkpoint_id} pass\n")
    manifest_rows = []
    for path in sorted(item for item in pair_root.rglob("*") if item.is_file()):
        relative = path.relative_to(pair_root)
        manifest_rows.append(f"{project_mlflow.sha256_bytes(path.read_bytes())}  {relative}\n")
    (pair_root / "MANIFEST.sha256").write_text("".join(manifest_rows))


def court_projects_complete_packet_after_local_comparison_exists() -> None:
    with tempfile.TemporaryDirectory() as directory:
        pair_root = Path(directory)
        prepare_complete_pair(pair_root)
        client = RecordingMlflowClient()

        projection = project_mlflow.project_pair(
            pair_root,
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="helix-lighteval-paired-v0",
            client=client,
        )

        assert projection.experiment_id == "experiment-1"
        assert projection.parent_run_id == "run-1"
        assert len(projection.checkpoint_run_ids) == 2
        assert len(client.created_runs) == 3
        assert all(run_id == "run-1" for run_id, _artifact in client.uploaded_artifacts)
        assert "paired_comparison.json" in {artifact for _run_id, artifact in client.uploaded_artifacts}
        assert "MANIFEST.sha256" in {artifact for _run_id, artifact in client.uploaded_artifacts}

        parent_batch = client.logged_batches[0]
        assert parent_batch[0] == "run-1"
        assert parent_batch[1]["evaluator_version"] == "0.13.0"
        assert "macro_delta_candidate_minus_control" in parent_batch[2]
        assert "delta/helix_piqa" in parent_batch[2]

        child_batches = client.logged_batches[1:]
        assert {batch[1]["checkpoint_id"] for batch in child_batches} == set(
            projection.checkpoint_run_ids
        )
        assert all("macro_mean" in batch[2] for batch in child_batches)
        assert all(("run-1", "FINISHED") != item for item in client.finished_runs[:-1])

        receipt = json.loads(
            (pair_root / "projections" / "mlflow_projection_receipt.json").read_text()
        )
        assert receipt["status"] == "complete"
        assert receipt["parent_run_id"] == "run-1"
        assert receipt["production_effect"] == "none"


def court_refuses_before_http_when_packet_is_incomplete() -> None:
    with tempfile.TemporaryDirectory() as directory:
        pair_root = Path(directory)
        write_pair(pair_root)
        client = RecordingMlflowClient()

        try:
            project_mlflow.project_pair(
                pair_root,
                tracking_uri="http://127.0.0.1:5000",
                experiment_name="helix-lighteval-paired-v0",
                client=client,
            )
        except project_mlflow.ProjectionError as exc:
            assert "paired comparison unavailable" in str(exc)
        else:
            raise AssertionError("incomplete local packet reached MLflow")

        assert client.created_runs == []
        assert client.logged_batches == []
        assert client.uploaded_artifacts == []


def court_refuses_stale_comparison_payload() -> None:
    with tempfile.TemporaryDirectory() as directory:
        pair_root = Path(directory)
        prepare_complete_pair(pair_root)
        comparison_path = pair_root / "paired_comparison.json"
        comparison = json.loads(comparison_path.read_text())
        comparison["macro_delta_candidate_minus_control"] = 999.0
        comparison_path.write_text(json.dumps(comparison))
        client = RecordingMlflowClient()

        try:
            project_mlflow.project_pair(
                pair_root,
                tracking_uri="http://127.0.0.1:5000",
                experiment_name="helix-lighteval-paired-v0",
                client=client,
            )
        except project_mlflow.ProjectionError as exc:
            assert "paired comparison drifted" in str(exc)
        else:
            raise AssertionError("stale comparison projected to MLflow")

        assert client.created_runs == []


def court_refuses_duplicate_projection_before_http() -> None:
    with tempfile.TemporaryDirectory() as directory:
        pair_root = Path(directory)
        prepare_complete_pair(pair_root)
        receipt_dir = pair_root / "projections"
        receipt_dir.mkdir()
        (receipt_dir / "mlflow_projection_receipt.json").write_text("{}\n")
        client = RecordingMlflowClient()

        try:
            project_mlflow.project_pair(
                pair_root,
                tracking_uri="http://127.0.0.1:5000",
                experiment_name="helix-lighteval-paired-v0",
                client=client,
            )
        except project_mlflow.ProjectionError as exc:
            assert "projection receipt already exists" in str(exc)
        else:
            raise AssertionError("duplicate projection reached MLflow")

        assert client.created_runs == []


def main() -> None:
    court_projects_complete_packet_after_local_comparison_exists()
    court_refuses_before_http_when_packet_is_incomplete()
    court_refuses_stale_comparison_payload()
    court_refuses_duplicate_projection_before_http()
    print("LIGHTEVAL_MLFLOW_PROJECTION_COURTS=PASS")


if __name__ == "__main__":
    main()
