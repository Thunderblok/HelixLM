#!/usr/bin/env python3
"""Best-effort real-time MLflow projection for the Helix experiment harness.

The JSONL spool is the local custody record.  MLflow is a live projection: a
tracking outage is recorded and reported, but never aborts training.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable


class RealtimeMLflowLogger:
    """Emit metrics as training progresses, with a local append-only spool."""

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment: str,
        run_name: str,
        spool_path: Path,
        params: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.run_name = run_name
        self.spool_path = Path(spool_path)
        self.spool_path.parent.mkdir(parents=True, exist_ok=True)
        self.params = params or {}
        self.tags = tags or {}
        self.mlflow: Any = None
        self.run: Any = None
        self.run_id: str | None = None
        self.mlflow_errors: list[str] = []
        self.started_at = time.time()
        self._append({"event": "logger_initialized", "ts": self.started_at})

    def _append(self, event: dict[str, Any]) -> None:
        with self.spool_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")

    def _safe_mlflow(self, operation: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - depends on remote service
            message = f"{operation}: {type(exc).__name__}: {exc}"
            self.mlflow_errors.append(message)
            self._append({"event": "mlflow_error", "operation": operation, "error": message, "ts": time.time()})
            return None

    def start(self) -> str | None:
        """Start the remote run before the first training step."""
        self._append({"event": "run_start_requested", "ts": time.time(), "tracking_uri": self.tracking_uri})
        try:
            import mlflow
        except Exception as exc:
            self.mlflow_errors.append(f"import: {type(exc).__name__}: {exc}")
            self._append({"event": "mlflow_unavailable", "error": self.mlflow_errors[-1], "ts": time.time()})
            return None
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(self.tracking_uri)
        self.mlflow.set_experiment(self.experiment)
        self.run = self._safe_mlflow("start_run", lambda: self.mlflow.start_run(run_name=self.run_name))
        if self.run is None:
            return None
        self.run_id = self.run.info.run_id
        self._safe_mlflow("log_params", lambda: self.mlflow.log_params({k: str(v) for k, v in self.params.items()}))
        self._safe_mlflow("set_tags", lambda: [self.mlflow.set_tag(k, str(v)) for k, v in self.tags.items()])
        self._append({"event": "run_started", "run_id": self.run_id, "ts": time.time()})
        return self.run_id

    def log_metrics(self, metrics: dict[str, float], *, step: int, phase: str = "train") -> None:
        clean = {str(k): float(v) for k, v in metrics.items() if v is not None}
        event = {"event": "metrics", "phase": phase, "step": int(step), "metrics": clean, "ts": time.time()}
        self._append(event)
        if self.mlflow is not None and self.run_id is not None and clean:
            self._safe_mlflow("log_metrics", lambda: self.mlflow.log_metrics(clean, step=int(step)))

    def log_epoch(self, epoch: int, train_metrics: dict[str, Any], val_metrics: dict[str, Any] | None = None, *, global_step: int = 0) -> None:
        self.log_metrics({f"train/{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))}, step=global_step, phase="train_epoch")
        if val_metrics:
            self.log_metrics({f"val/{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}, step=global_step, phase="validation")
        self._append({"event": "epoch", "epoch": int(epoch), "global_step": int(global_step), "ts": time.time()})

    def finish(self, *, status: str = "FINISHED") -> None:
        self._append({"event": "run_finish_requested", "status": status, "ts": time.time(), "mlflow_errors": self.mlflow_errors})
        if self.mlflow is not None and self.run_id is not None:
            self._safe_mlflow("end_run", lambda: self.mlflow.end_run(status=status))
        self._append({"event": "run_finished", "status": status, "ts": time.time(), "mlflow_errors": self.mlflow_errors})


def install_epoch_hooks(trainer: Any, logger: RealtimeMLflowLogger) -> None:
    """Add live epoch/validation projection without modifying Helix source."""
    original_train_epoch = trainer.train_epoch
    original_evaluate = trainer.evaluate
    trainer._realtime_mlflow_last_validation = None

    def train_epoch(epoch: int) -> dict[str, Any]:
        result = original_train_epoch(epoch)
        trainer._realtime_mlflow_last_train = result
        logger.log_epoch(epoch, result, trainer._realtime_mlflow_last_validation, global_step=int(getattr(trainer, "global_step", epoch)))
        return result

    def evaluate() -> dict[str, Any]:
        result = original_evaluate()
        trainer._realtime_mlflow_last_validation = result
        metrics = {
            f"val/{k}": v
            for k, v in result.items()
            if isinstance(v, (int, float))
        }
        logger.log_metrics(
            metrics,
            step=int(getattr(trainer, "global_step", 0)),
            phase="validation",
        )
        return result

    trainer.train_epoch = train_epoch
    trainer.evaluate = evaluate
