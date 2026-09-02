"""Exact-subset loader for Hugging Face converted-parquet revisions.

Datasets 4 no longer executes the original dataset scripts. The corresponding
``refs/convert/parquet`` revisions expose all source configurations through a
single synthetic ``default`` config, which would silently combine ARC Easy
with ARC Challenge and similar sibling subsets. This adapter selects the
contract-bound parquet shards directly instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from contract import ContractError


def projection_data_files(task_contract: dict[str, Any]) -> dict[str, list[str]]:
    repo = task_contract["hf_repo"]
    revision = task_contract["hf_revision"]
    subset = task_contract["hf_subset"]
    projection_files = task_contract["projection_files"]
    prefix = f"hf://datasets/{repo}@{revision}/{subset}"
    return {
        split: [f"{prefix}/{filename}" for filename in filenames]
        for split, filenames in projection_files.items()
    }


def install_exact_parquet_loader(contract: dict[str, Any]) -> None:
    from datasets import load_dataset
    from lighteval.tasks.lighteval_task import LightevalTask

    tasks = {task["name"]: task for task in contract["tasks"]}

    def download(task: Any):
        task_name = task.name.split("|", 1)[0]
        task_contract = tasks.get(task_name)
        if task_contract is None:
            raise ContractError(f"unadmitted dataset projection task: {task_name}")
        dataset = load_dataset(
            path="parquet",
            data_files=projection_data_files(task_contract),
        )
        if task.dataset_filter is not None:
            dataset = dataset.filter(task.dataset_filter)
        return dataset

    LightevalTask.download_dataset_worker = staticmethod(download)
