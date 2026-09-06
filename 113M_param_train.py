#!/usr/bin/env python3
"""Production continuous-pretraining launcher for one 16 GB GPU.

The file is intentionally readable from top to bottom: defaults, resolved run
contract, validation, data, model, tracking, training, local save, and optional
publication. Every setting that changes the executable training subject is a
named default below and has a matching ``HELIX_*`` environment override.

``PretrainTrainer`` owns continuous EOS-joined causal pretraining. ``Trainer``
remains the separate document-aware fine-tuning path. Experimental NCA, Rule-30,
adapter, and transfer-state treatments do not belong in this production entry
point; they must remain explicit experiment layers around the same public APIs.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from datasets import load_dataset

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, PretrainTrainer
from helix_lm.experiment_tracking import ExperimentTracker


SUTRA_DATASET = "codelion/sutra-10B"
SUTRA_REVISION = "415549cff1a92b69df8b88c6108faa6097457068"

# ---------------------------------------------------------------------------
# Public single-GPU defaults
# ---------------------------------------------------------------------------
# These defaults target a 16 GB CUDA GPU. Larger hardware should override the
# individual values so the emitted run contract records the actual subject.
PROFILE_NAME = "single-gpu-16gb"

# Data and tokenizer. Production runs pin a dataset revision so resume and
# comparisons cannot silently move when a remote dataset changes.
DATASET = SUTRA_DATASET
DATASET_REVISION = SUTRA_REVISION
DATASET_SPLIT = "train"
TEXT_COLUMN = "text"
TOKENIZER_NAME = "gpt2"

# Model and graph topology.
D_MODEL = 768
N_HEADS = 12
N_COLUMNS = 3
NODES_PER_COLUMN = (3, 3, 3)
N_LOOPS = 4
SEQUENCE_LENGTH = 1024
FFN_EXPANSION = 3.0
DROPOUT = 0.1
ATTENTION_DROPOUT = 0.05
LATERAL_P = 0.8
VERTICAL_P = 0.9
VERTICAL_DEPTH = 2

# Attention and optional architecture components.
ATTENTION_MODE = "multi_scale_windowed"
LOCAL_WINDOW = 64
COARSE_WINDOW = 128
COMPRESSED_WINDOWS = 8
COMPRESSED_VIEWS = 8
CONSENSUS_TYPE = "cosine"
CORRECTOR_TYPE = "ffn"
USE_CCA = False
USE_SSM = False
USE_TITANS_MEMORY = False
TIE_WORD_EMBEDDINGS = True
STRICT_NAN_CHECK = True

# Optimizer and numerical policy. This launcher intentionally uses one fixed
# learning rate. A staged schedule is a separate, explicitly versioned recipe.
BATCH_SIZE = 3
GRAD_ACCUM = 28
EPOCHS = 1
LEARNING_RATE = 2e-4
WARMUP_MICROBATCHES = 500
MAX_OPTIMIZER_STEPS = 3_082
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
GRAD_BUFFER_RATIO = 0.0
DTYPE = "float32"
AMP_DTYPE = "bfloat16"
USE_AMP = True

# Validation, checkpointing, and operator defaults.
VALIDATION_SAMPLES = 252
VALIDATION_BATCHES = 8
CHECKPOINT_EVERY_STEPS = 250
CHECKPOINT_SLOTS = 2
EVAL_EVERY_STEPS = 250
NUM_WORKERS = 4
SEED = 42

# MLflow is required by default for production runs. The append-only local JSONL
# spool remains the durable record if projection fails after admission.
MLFLOW_URI = "https://mlflow.thunderline.net"
MLFLOW_EXPERIMENT = "helix-pretraining"
REQUIRE_MLFLOW = True


@dataclass(frozen=True)
class RunSettings:
    """Fully resolved, validated contract for one executable training run."""

    profile_name: str
    dataset: str
    dataset_revision: str
    dataset_split: str
    text_column: str
    tokenizer_name: str
    train_store_dir: Optional[Path]
    compile_store_dir: Optional[Path]
    resume_training_state: Optional[Path]
    output_root: Path
    d_model: int
    n_heads: int
    n_columns: int
    nodes_per_column: tuple[int, ...]
    n_loops: int
    seq_len: int
    ffn_expansion: float
    dropout: float
    attn_dropout: float
    lateral_p: float
    vertical_p: float
    vertical_depth: int
    attention_mode: str
    local_window: int
    coarse_window: int
    compressed_windows: int
    compressed_views: int
    consensus_type: str
    corrector_type: str
    use_cca: bool
    use_ssm: bool
    use_titans_memory: bool
    tie_word_embeddings: bool
    strict_nan_check: bool
    batch_size: int
    grad_accum: int
    epochs: int
    learning_rate: float
    warmup_microbatches: int
    weight_decay: float
    grad_clip: float
    grad_buffer_ratio: float
    dtype: str
    amp_dtype: str
    use_amp: bool
    validation_samples: int
    validation_batches: int
    checkpoint_every_steps: int
    checkpoint_slots: int
    eval_every_steps: int
    max_optimizer_steps: Optional[int]
    num_workers: int
    seed: int
    mlflow_uri: str
    mlflow_experiment: str
    require_mlflow: bool
    push_to_hub: bool
    hf_username: str

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum


def _optional_path(value: Optional[str]) -> Optional[Path]:
    return Path(value).expanduser() if value and value.strip() else None


def _env_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _env_nodes(environ: Mapping[str, str]) -> tuple[int, ...]:
    raw = environ.get(
        "HELIX_NODES_PER_COLUMN", ",".join(str(value) for value in NODES_PER_COLUMN)
    )
    try:
        return tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError("HELIX_NODES_PER_COLUMN must be comma-separated integers") from exc


def resolve_settings(environ: Mapping[str, str] = os.environ) -> RunSettings:
    max_steps_value = int(
        environ.get("HELIX_MAX_OPTIMIZER_STEPS", str(MAX_OPTIMIZER_STEPS))
    )
    timestamp = datetime.now(timezone.utc).strftime("%y%m%d-%H%M")
    train_store_dir = _optional_path(environ.get("HELIX_PRETRAIN_STORE_DIR"))
    compile_store_dir = _optional_path(
        environ.get(
            "HELIX_PRETRAIN_COMPILE_DIR",
            "" if train_store_dir else "./pretrain_store",
        )
    )
    settings = RunSettings(
        profile_name=PROFILE_NAME,
        dataset=environ.get("HELIX_DATASET", DATASET),
        dataset_revision=environ.get("HELIX_DATASET_REVISION", DATASET_REVISION),
        dataset_split=environ.get("HELIX_DATASET_SPLIT", DATASET_SPLIT),
        text_column=environ.get("HELIX_TEXT_COLUMN", TEXT_COLUMN),
        tokenizer_name=environ.get("HELIX_TOKENIZER", TOKENIZER_NAME),
        train_store_dir=train_store_dir,
        compile_store_dir=compile_store_dir,
        resume_training_state=_optional_path(environ.get("HELIX_RESUME_TRAINING_STATE")),
        output_root=Path(
            environ.get(
                "HELIX_OUTPUT_DIR",
                f"production_runs/hlx-{PROFILE_NAME}-{timestamp}",
            )
        ).expanduser(),
        d_model=int(environ.get("HELIX_D_MODEL", str(D_MODEL))),
        n_heads=int(environ.get("HELIX_N_HEADS", str(N_HEADS))),
        n_columns=int(environ.get("HELIX_N_COLUMNS", str(N_COLUMNS))),
        nodes_per_column=_env_nodes(environ),
        n_loops=int(environ.get("HELIX_N_LOOPS", str(N_LOOPS))),
        seq_len=int(environ.get("HELIX_SEQUENCE_LENGTH", str(SEQUENCE_LENGTH))),
        ffn_expansion=float(environ.get("HELIX_FFN_EXPANSION", str(FFN_EXPANSION))),
        dropout=float(environ.get("HELIX_DROPOUT", str(DROPOUT))),
        attn_dropout=float(
            environ.get("HELIX_ATTENTION_DROPOUT", str(ATTENTION_DROPOUT))
        ),
        lateral_p=float(environ.get("HELIX_LATERAL_P", str(LATERAL_P))),
        vertical_p=float(environ.get("HELIX_VERTICAL_P", str(VERTICAL_P))),
        vertical_depth=int(environ.get("HELIX_VERTICAL_DEPTH", str(VERTICAL_DEPTH))),
        attention_mode=environ.get("HELIX_ATTENTION_MODE", ATTENTION_MODE),
        local_window=int(environ.get("HELIX_LOCAL_WINDOW", str(LOCAL_WINDOW))),
        coarse_window=int(environ.get("HELIX_COARSE_WINDOW", str(COARSE_WINDOW))),
        compressed_windows=int(
            environ.get("HELIX_COMPRESSED_WINDOWS", str(COMPRESSED_WINDOWS))
        ),
        compressed_views=int(
            environ.get("HELIX_COMPRESSED_VIEWS", str(COMPRESSED_VIEWS))
        ),
        consensus_type=environ.get("HELIX_CONSENSUS_TYPE", CONSENSUS_TYPE),
        corrector_type=environ.get("HELIX_CORRECTOR_TYPE", CORRECTOR_TYPE),
        use_cca=_env_bool(environ, "HELIX_USE_CCA", USE_CCA),
        use_ssm=_env_bool(environ, "HELIX_USE_SSM", USE_SSM),
        use_titans_memory=_env_bool(
            environ, "HELIX_USE_TITANS_MEMORY", USE_TITANS_MEMORY
        ),
        tie_word_embeddings=_env_bool(
            environ, "HELIX_TIE_WORD_EMBEDDINGS", TIE_WORD_EMBEDDINGS
        ),
        strict_nan_check=_env_bool(
            environ, "HELIX_STRICT_NAN_CHECK", STRICT_NAN_CHECK
        ),
        batch_size=int(environ.get("HELIX_BATCH_SIZE", str(BATCH_SIZE))),
        grad_accum=int(environ.get("HELIX_GRAD_ACCUM", str(GRAD_ACCUM))),
        epochs=int(environ.get("HELIX_EPOCHS", str(EPOCHS))),
        learning_rate=float(environ.get("HELIX_LEARNING_RATE", str(LEARNING_RATE))),
        warmup_microbatches=int(
            environ.get("HELIX_WARMUP_MICROBATCHES", str(WARMUP_MICROBATCHES))
        ),
        weight_decay=float(environ.get("HELIX_WEIGHT_DECAY", str(WEIGHT_DECAY))),
        grad_clip=float(environ.get("HELIX_GRAD_CLIP", str(GRAD_CLIP))),
        grad_buffer_ratio=float(
            environ.get("HELIX_GRAD_BUFFER_RATIO", str(GRAD_BUFFER_RATIO))
        ),
        dtype=environ.get("HELIX_DTYPE", DTYPE),
        amp_dtype=environ.get("HELIX_AMP_DTYPE", AMP_DTYPE),
        use_amp=_env_bool(environ, "HELIX_USE_AMP", USE_AMP),
        validation_samples=int(
            environ.get("HELIX_VALIDATION_SAMPLES", str(VALIDATION_SAMPLES))
        ),
        validation_batches=int(
            environ.get("HELIX_VALIDATION_BATCHES", str(VALIDATION_BATCHES))
        ),
        checkpoint_every_steps=int(
            environ.get("HELIX_CHECKPOINT_EVERY", str(CHECKPOINT_EVERY_STEPS))
        ),
        checkpoint_slots=int(
            environ.get("HELIX_CHECKPOINT_SLOTS", str(CHECKPOINT_SLOTS))
        ),
        eval_every_steps=int(
            environ.get("HELIX_EVAL_EVERY", str(EVAL_EVERY_STEPS))
        ),
        max_optimizer_steps=max_steps_value or None,
        num_workers=int(environ.get("HELIX_NUM_WORKERS", str(NUM_WORKERS))),
        seed=int(environ.get("HELIX_SEED", str(SEED))),
        mlflow_uri=environ.get("HELIX_MLFLOW_URI", MLFLOW_URI),
        mlflow_experiment=environ.get("HELIX_MLFLOW_EXPERIMENT", MLFLOW_EXPERIMENT),
        require_mlflow=_env_bool(environ, "HELIX_REQUIRE_MLFLOW", REQUIRE_MLFLOW),
        push_to_hub=_env_bool(environ, "HELIX_PUSH_TO_HUB", False),
        hf_username=environ.get("HF_USER", ""),
    )
    validate_settings(settings, environ)
    return settings


def validate_settings(
    settings: RunSettings, environ: Mapping[str, str] = os.environ
) -> None:
    if settings.d_model <= 0 or settings.n_heads <= 0:
        raise ValueError("d_model and n_heads must be positive")
    if settings.d_model % settings.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    if len(settings.nodes_per_column) != settings.n_columns:
        raise ValueError("nodes_per_column must contain one value per column")
    if any(value <= 0 for value in settings.nodes_per_column):
        raise ValueError("nodes_per_column values must be positive")
    if not 0.0 <= settings.lateral_p <= 1.0:
        raise ValueError("lateral_p must be between zero and one")
    if not 0.0 <= settings.vertical_p <= 1.0:
        raise ValueError("vertical_p must be between zero and one")
    if not 0.0 <= settings.dropout < 1.0:
        raise ValueError("dropout must be between zero inclusive and one exclusive")
    if not 0.0 <= settings.attn_dropout < 1.0:
        raise ValueError(
            "attention dropout must be between zero inclusive and one exclusive"
        )
    if settings.max_optimizer_steps is not None and settings.max_optimizer_steps < 0:
        raise ValueError("max optimizer steps must be zero or positive")
    if settings.epochs <= 0 or settings.learning_rate <= 0:
        raise ValueError("epochs and learning rate must be positive")
    positive = {
        "d_model": settings.d_model,
        "n_heads": settings.n_heads,
        "columns": settings.n_columns,
        "loops": settings.n_loops,
        "sequence length": settings.seq_len,
        "FFN expansion": settings.ffn_expansion,
        "batch size": settings.batch_size,
        "gradient accumulation": settings.grad_accum,
        "vertical depth": settings.vertical_depth,
        "local window": settings.local_window,
        "coarse window": settings.coarse_window,
        "compressed windows": settings.compressed_windows,
        "compressed views": settings.compressed_views,
        "warmup microbatches": settings.warmup_microbatches,
        "validation samples": settings.validation_samples,
        "validation batches": settings.validation_batches,
        "checkpoint interval": settings.checkpoint_every_steps,
        "checkpoint slots": settings.checkpoint_slots,
        "evaluation interval": settings.eval_every_steps,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError("All count and interval settings must be positive")
    if settings.resume_training_state and not settings.train_store_dir:
        raise ValueError("Exact resume requires HELIX_PRETRAIN_STORE_DIR")
    if settings.train_store_dir and settings.compile_store_dir:
        raise ValueError("Existing and new pretraining stores are mutually exclusive")
    if not settings.dataset_revision:
        raise ValueError("HELIX_DATASET_REVISION must be pinned")
    if settings.num_workers < 0:
        raise ValueError("num_workers must be zero or positive")
    if settings.require_mlflow and (
        not settings.mlflow_uri.strip() or not settings.mlflow_experiment.strip()
    ):
        raise ValueError("Required MLflow projection needs a URI and experiment")
    if settings.push_to_hub and not settings.hf_username:
        raise ValueError("HELIX_PUSH_TO_HUB=1 requires HF_USER")
    if settings.push_to_hub and not environ.get("HF_TOKEN"):
        raise ValueError("HELIX_PUSH_TO_HUB=1 requires HF_TOKEN")


def source_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()

    return {
        "source_head": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_branch": git("branch", "--show-current") or "DETACHED",
        "source_dirty": bool(git("status", "--porcelain")),
    }


def gpu_utilization_percent() -> Optional[float]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits", "--id=0"],
            check=True, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=5,
        )
        return float(completed.stdout.strip().splitlines()[0])
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return None


def build_config(settings: RunSettings, tokenizer: HelixTokenizer) -> HelixConfig:
    cfg = HelixConfig.small_v2(
        vocab_size=len(tokenizer),
        tokenizer_name=settings.tokenizer_name,
        d_model=settings.d_model,
        n_columns=settings.n_columns,
        nodes_per_column=settings.nodes_per_column,
        n_heads=settings.n_heads,
        n_loops=settings.n_loops,
        seq_len=settings.seq_len,
        dropout=settings.dropout,
        attn_dropout=settings.attn_dropout,
        ffn_expansion=settings.ffn_expansion,
        weight_decay=settings.weight_decay,
        grad_clip=settings.grad_clip,
        grad_buffer_ratio=settings.grad_buffer_ratio,
        batch_size=settings.batch_size,
        lr=settings.learning_rate,
        warmup_steps=settings.warmup_microbatches,
        epochs=settings.epochs,
        use_cca=settings.use_cca,
        use_ssm=settings.use_ssm,
        use_titans_memory=settings.use_titans_memory,
        seed=settings.seed,
        device="auto",
        dtype=settings.dtype,
        amp_dtype=settings.amp_dtype,
        lateral_p=settings.lateral_p,
        vertical_p=settings.vertical_p,
        vertical_depth=settings.vertical_depth,
        attention_mode=settings.attention_mode,
        local_window=settings.local_window,
        coarse_window=settings.coarse_window,
        compressed_windows=settings.compressed_windows,
        compressed_views=settings.compressed_views,
        consensus_type=settings.consensus_type,
        corrector_type=settings.corrector_type,
        tie_word_embeddings=settings.tie_word_embeddings,
        strict_nan_check=settings.strict_nan_check,
    )
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.bos_token_id = tokenizer.bos_token_id
    return cfg


def model_name(settings: RunSettings, timestamp: str) -> str:
    nodes = "".join(str(value) for value in settings.nodes_per_column)
    ffn = f"{settings.ffn_expansion:.1f}".replace(".", "")
    value = (
        f"hlx-{timestamp}-d{settings.d_model}-c{settings.n_columns}-n{nodes}-"
        f"l{settings.n_loops}-f{ffn}-s{settings.seq_len}-e{settings.epochs}"
    )
    if len(value) > 96:
        raise ValueError("Generated Hugging Face model name exceeds 96 characters")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def training_texts(settings: RunSettings):
    if settings.train_store_dir:
        return None
    dataset = load_dataset(
        settings.dataset, streaming=True, revision=settings.dataset_revision
    )
    return dataset[settings.dataset_split][settings.text_column]


def main() -> None:
    # 1. Resolve and validate the complete run contract before GPU, data, or
    # tracking side effects. HELIX_PRINT_CONTRACT is the cheap operator preflight.
    settings = resolve_settings()
    if os.environ.get("HELIX_PRINT_CONTRACT") == "1":
        print(json.dumps(asdict(settings), default=str, indent=2))
        return

    # 2. The supported 16 GB recipe depends on native BF16. Refuse rather than
    # silently changing precision, batch shape, or numerical behavior.
    if not torch.cuda.is_available():
        raise RuntimeError("UNAVAILABLE: a CUDA GPU is required")
    if (
        settings.use_amp
        and settings.amp_dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("UNAVAILABLE: the selected BF16 AMP policy is unsupported")

    # 3. Bind source identity and deterministic RNG state before model creation.
    timestamp = datetime.now(timezone.utc).strftime("%y%m%d-%H%M")
    settings.output_root.mkdir(parents=True, exist_ok=True)
    subject = source_identity()
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    torch.cuda.reset_peak_memory_stats()

    # 4. Construct the tokenizer, declared model configuration, and model
    # directly from HelixLM's public APIs. The measured graph is recorded below.
    tokenizer = HelixTokenizer(settings.tokenizer_name)
    cfg = build_config(settings, tokenizer)
    model = HelixForCausalLM(cfg)
    counts = model.count_parameters()
    graph_info = model.model.recurrent.graph.get_graph_info()
    run_name = model_name(settings, timestamp)
    pretrain_source = {
        "dataset": settings.dataset, "revision": settings.dataset_revision,
        "split": settings.dataset_split, "text_column": settings.text_column,
        "tokenizer": settings.tokenizer_name,
    }
    params = {
        **subject,
        **pretrain_source,
        "profile": settings.profile_name,
        "vocab_size": len(tokenizer),
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_columns": cfg.n_columns,
        "nodes_per_column": ",".join(map(str, settings.nodes_per_column)),
        "nodes_per_column_graph_effective": False,
        "n_loops": cfg.n_loops,
        "ffn_expansion": cfg.ffn_expansion,
        "dropout": cfg.dropout,
        "attention_dropout": cfg.attn_dropout,
        "sequence_length": cfg.seq_len,
        "attention_mode": cfg.attention_mode,
        "local_window": cfg.local_window,
        "coarse_window": cfg.coarse_window,
        "compressed_windows": cfg.compressed_windows,
        "compressed_views": cfg.compressed_views,
        "consensus_type": cfg.consensus_type,
        "corrector_type": cfg.corrector_type,
        "use_cca": cfg.use_cca,
        "use_ssm": cfg.use_ssm,
        "use_titans_memory": cfg.use_titans_memory,
        "lateral_p": cfg.lateral_p,
        "vertical_p": cfg.vertical_p,
        "vertical_depth": cfg.vertical_depth,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "batch_size": settings.batch_size,
        "grad_accum": settings.grad_accum,
        "effective_batch": settings.effective_batch,
        "learning_rate": settings.learning_rate,
        "warmup_microbatches": settings.warmup_microbatches,
        "weight_decay": cfg.weight_decay,
        "grad_clip": cfg.grad_clip,
        "grad_buffer_ratio": cfg.grad_buffer_ratio,
        "epochs": settings.epochs,
        "max_optimizer_steps": settings.max_optimizer_steps,
        "validation_samples": settings.validation_samples,
        "validation_batches": settings.validation_batches,
        "checkpoint_every_steps": settings.checkpoint_every_steps,
        "checkpoint_slots": settings.checkpoint_slots,
        "eval_every_steps": settings.eval_every_steps,
        "num_workers": settings.num_workers,
        "seed": settings.seed,
        "dtype": settings.dtype,
        "use_amp": settings.use_amp,
        "amp_dtype": settings.amp_dtype,
        "strict_nan_check": settings.strict_nan_check,
        "parameter_count": counts["total"],
        "graph_nodes": graph_info["n_nodes"],
        "graph_edges": graph_info["n_edges"],
        "gpu_name": torch.cuda.get_device_name(0),
    }
    # 5. Initialize the local experiment record before the trainer. MLflow is a
    # projection; the JSONL spool is retained if a later projection call fails.
    tracker = ExperimentTracker(
        tracking_uri=settings.mlflow_uri,
        experiment=settings.mlflow_experiment,
        run_name=run_name,
        spool_path=settings.output_root / "mlflow-events.jsonl",
        params=params,
        tags={
            "branch": subject["source_branch"],
            "source_head": subject["source_head"],
            "profile": settings.profile_name,
            "data_contract": "eos_joined_nonoverlap_persisted_permutation_v1",
        },
        require_remote=settings.require_mlflow,
    )
    data_stats: dict[str, float] = {}
    trainer: Optional[PretrainTrainer] = None

    def on_step(metrics: dict[str, float]) -> None:
        cursor = metrics["sample_cursor"]
        sample_count = max(data_stats.get("sample_count", 0.0), 1.0)
        raw_bytes = data_stats.get("raw_utf8_bytes", 0.0) * cursor / sample_count
        tracker.log_metrics(
            {
                "train/loss": metrics["loss"],
                "train/ppl": metrics["perplexity"],
                "train/lr": metrics["lr"],
                "train/global_step": metrics["global_step"],
                "train/grad_norm": metrics["grad_norm"],
                "train/causal_targets_total": metrics["causal_targets_total"],
                "train/causal_targets_per_second_session": metrics[
                    "causal_targets_per_second_session"
                ],
                "train/causal_targets_per_second_step": metrics[
                    "causal_targets_per_second_step"
                ],
                "train/step_seconds": metrics["step_seconds"],
                "train/skipped_batches": metrics["skipped_batches"],
                "data/sample_cursor": cursor,
                "data/sample_store_bytes_consumed": cursor * cfg.seq_len * 2,
                "data/raw_utf8_bytes_exposure_estimated": raw_bytes,
                "system/vram_allocated_bytes": metrics["vram_allocated_bytes"],
                "system/vram_reserved_bytes": metrics["vram_reserved_bytes"],
                "system/peak_vram_bytes": metrics["peak_vram_bytes"],
                "system/gpu_utilization_percent": gpu_utilization_percent(),
            },
            step=int(metrics["global_step"]),
            phase="train",
        )

    def on_validation(metrics: dict[str, float]) -> None:
        assert trainer is not None
        tracker.log_metrics(
            {
                "val/loss": metrics["loss"],
                "val/ppl": metrics["perplexity"],
                "val/causal_targets": metrics["causal_targets"],
                "val/sample_count": metrics["sample_count"],
            },
            step=trainer.global_step,
            phase="validation",
        )

    terminal_status = "FAILED"
    terminal: Optional[dict[str, Any]] = None
    started = time.time()
    try:
        # 6. PretrainTrainer compiles a text iterable into the indexed store or
        # verifies and reuses the declared existing store. Resume requires reuse.
        trainer = PretrainTrainer(
            model=model,
            cfg=cfg,
            train_texts=training_texts(settings),
            train_store_dir=settings.train_store_dir,
            pretrain_store_dir=settings.compile_store_dir,
            pretrain_source=pretrain_source,
            resume_training_state=settings.resume_training_state,
            validation_sample_count=settings.validation_samples,
            tokenizer=tokenizer,
            output_dir=settings.output_root / "checkpoints",
            grad_accum_steps=settings.grad_accum,
            use_amp=settings.use_amp,
            amp_dtype=settings.amp_dtype,
            verbose=True,
            num_workers=settings.num_workers,
            total_optimizer_steps=settings.max_optimizer_steps,
            max_optimizer_steps=settings.max_optimizer_steps,
            min_lr_ratio=1.0,
            checkpoint_every_steps=settings.checkpoint_every_steps,
            checkpoint_slots=settings.checkpoint_slots,
            step_callback=on_step,
            eval_every_steps=settings.eval_every_steps,
            validation_batches=settings.validation_batches,
            evaluation_callback=on_validation,
        )
        manifest = trainer._train_dataset.manifest
        data_stats.update(
            sample_count=float(manifest.sample_count),
            raw_utf8_bytes=float(manifest.value["raw_utf8_bytes"]),
        )
        contract = {
            "schema": "helix.production-pretrain-run.v1",
            **params,
            "run_name": run_name,
            "output_root": str(settings.output_root.resolve()),
            "sample_store": str(manifest.root.resolve()),
            "sample_manifest_sha256": manifest.manifest_sha256,
            "permutation_sha256": trainer._train_permutation.metadata["sha256"],
            "validation_policy": "tail_of_epoch_zero_persisted_permutation_v0",
            "validation_sample_ids_sha256": trainer._validation_sample_ids_sha256,
        }
        write_json(settings.output_root / "run_contract.json", contract)
        tracker.params.update(
            {
                "sample_manifest_sha256": manifest.manifest_sha256,
                "permutation_sha256": trainer._train_permutation.metadata["sha256"],
                "validation_sample_ids_sha256": trainer._validation_sample_ids_sha256,
            }
        )
        mlflow_run_id = tracker.start()
        contract["mlflow_run_id"] = mlflow_run_id
        write_json(settings.output_root / "run_contract.json", contract)

        # 7. Train the declared subject. Checkpoint rotation and validation are
        # owned by PretrainTrainer and are driven by the resolved intervals.
        history = trainer.train(num_epochs=settings.epochs, eval_every=1)

        # 8. Save locally before any optional network publication. Local custody
        # is never made conditional on Hugging Face availability.
        final_dir = settings.output_root / "final-model"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        hub_repo = ""
        if settings.push_to_hub:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HELIX_PUSH_TO_HUB=1 requires HF_TOKEN")
            hub_repo = f"{settings.hf_username}/{run_name}"
            model.push_to_hub(hub_repo, token=token)
            tokenizer.push_to_hub(hub_repo, token=token)

        train_loss = (history.get("train_loss") or [float("nan")])[-1]
        val_loss = (history.get("val_loss") or [float("nan")])[-1]
        terminal = {
            "status": "PASS",
            "mlflow_run_id": mlflow_run_id,
            "mlflow_errors": tracker.errors,
            "global_step": trainer.global_step,
            "sample_cursor": trainer._train_cursor,
            "final_train_loss": train_loss,
            "final_train_ppl": math.exp(min(train_loss, 20)),
            "final_val_loss": val_loss,
            "final_val_ppl": math.exp(min(val_loss, 20)),
            "elapsed_seconds": time.time() - started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "local_model": str(final_dir.resolve()),
            "hub_repo": hub_repo,
        }
        tracker.log_metrics(
            {
                "final/train_loss": terminal["final_train_loss"],
                "final/train_ppl": terminal["final_train_ppl"],
                "final/val_loss": terminal["final_val_loss"],
                "final/val_ppl": terminal["final_val_ppl"],
                "final/elapsed_seconds": terminal["elapsed_seconds"],
                "final/peak_vram_bytes": terminal["peak_vram_bytes"],
            },
            step=trainer.global_step,
            phase="terminal",
        )
        terminal_status = "FINISHED"
    finally:
        projected_status = tracker.finish(terminal_status)
        if terminal is not None:
            terminal["mlflow_errors"] = list(tracker.errors)
            terminal["status"] = (
                "PASS" if projected_status == "FINISHED"
                else "PASS_WITH_MLFLOW_ERRORS"
            )
            write_json(settings.output_root / "run_terminal.json", terminal)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; resume from the latest local training-state checkpoint", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
