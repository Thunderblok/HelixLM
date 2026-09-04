#!/usr/bin/env python3
"""
HelixLM 100M (approx) — 3B token pretraining, production run.
Single training job with 3 epochs, each at a distinct learning rate.
Multi-scale windowed attention.
Uses PretrainTrainer for continuous token windows.
"""

import math
import json
import sys
import os
import time
import logging
import traceback
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_file as load_safetensors

from helix_lm import (
    HelixTokenizer,
    HelixConfig,
    HelixForCausalLM,
    PretrainTrainer,          # <-- changed import
)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# 3B token dataset with train/val splits — STREAMING MODE
PUSH_TO_HUB = os.getenv("HELIX_PUSH_TO_HUB", "0") == "1"
DATASET = os.getenv("HELIX_DATASET", "david-thrower/helixlm87M-3Btoken-pretrain-dataset-v1")
DATASET_REVISION = os.getenv("HELIX_DATASET_REVISION")
PRETRAIN_STORE_DIR = os.getenv("HELIX_PRETRAIN_STORE_DIR")
HF_USERNAME = "david-thrower"
HF_TOKEN = os.getenv("HF_TOKEN")

if PUSH_TO_HUB:
    if not HF_TOKEN:
        raise ValueError("Env var HF_TOKEN must be set to push the model to HF Hub.")

# Set to None to use full dataset
NUM_SAMPLES = None

# Reproducibility
SEED = 42

# Attention mechanism node selection
ATTENTION_MODE = "multi_scale_windowed"

# Basic model structural configuration
D_MODEL = 768
N_COLUMNS = 3
NODES_PER_COLUMN = (3, 3, 3)  # WAS (2, 3, 2)
N_HEADS = D_MODEL // 64          # 12 heads for d_model=768
FFN_EXPANSION = 3                # Per PanGu-π + 0.3 for architectural reasons
SEQ_LEN = 1024                   # Target context length
N_LOOPS = 4                      # Production: 4 recurrent loops
DROPOUT = 0.1
ATTENTION_DROPOUT = 0.05
GRAD_BUFFER_RATIO = 0.0          # Standard weight tying (no gradient buffer)

# Multi-scale linear attention args
LOCAL_WINDOW = 64
COARSE_WINDOW = 128
COMPRESSED_WINDOWS = 8
COMPRESSED_VIEWS = 8
CONSENSUS_TYPE = "cosine"
CORRECTOR_TYPE = "ffn"

# Topology — dense
LATERAL_P = 0.8
VERTICAL_P = 0.9
VERTICAL_DEPTH = 2

# --- Training ---
BATCH_SIZE = 12
GRAD_ACCUM = 7                     # Effective batch size = 12*7 = 84
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

EPOCHS = 3
EPOCH_1_LR = 0.0002
LR_STAGES = [EPOCH_1_LR * (1/3) ** i for i in range(EPOCHS)]
WARMUP_STAGES = [500, 100, 50]

# AMP
USE_AMP = True
AMP_DTYPE = "bfloat16"
D_TYPE = "float32"

# Flags
USE_CCA = False
USE_SSM = False
USE_TITANS = False

# Hub
PUSH_RETRY_ATTEMPTS = 3
PUSH_RETRY_DELAY = 90

# Streaming dataset settings
STREAMING = True
PREPROCESS_BATCH_SIZE = 1000       # not used by PretrainTrainer, but kept for logging
CLEANUP_SHARDS = True              # not used by PretrainTrainer
NUM_WORKERS = int(os.getenv("HELIX_NUM_WORKERS", "4" if PRETRAIN_STORE_DIR else "0"))

# Tokenizer
TOKENIZER_NAME = "gpt2"

# ═══════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
OUTPUT_DIR = Path("production_run_HelixLM-100M_1024_nloops4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / f"production_train_{RUN_TS}.log"
RESULTS_JSON = OUTPUT_DIR / f"production_results_{RUN_TS}.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Dynamic repo naming ──────────────────────────────────────────────────
REPO_BASE = (
    f"HelixLM-100M-{RUN_TS}-d{D_MODEL}-col{N_COLUMNS}-nl{N_LOOPS}"
    f"-ffn{int(FFN_EXPANSION)}-s{SEQ_LEN}-3BT"
)
REPO_BASE_ALT = f"HelixLM-100M-{RUN_TS}-prod"


def make_repo_name(epoch: int, use_alt: bool = False) -> str:
    base = REPO_BASE_ALT if use_alt else REPO_BASE
    return f"{HF_USERNAME}/{base}-ep{epoch}"


# ═══════════════════════════════════════════════════════════════════════════
# HUB PUSH HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def push_to_hub_safe(model, tokenizer, repo_name):
    for attempt in range(1, PUSH_RETRY_ATTEMPTS + 1):
        try:
            logger.info("📤 Push: %s (attempt %d/%d)", repo_name, attempt, PUSH_RETRY_ATTEMPTS)
            model.push_to_hub(repo_name)
            tokenizer.push_to_hub(repo_name)
            logger.info("✅ Pushed: https://huggingface.co/%s", repo_name)
            return True
        except Exception as e:
            logger.warning("⚠️  Push attempt %d failed: %s", attempt, e)
            if attempt < PUSH_RETRY_ATTEMPTS:
                time.sleep(PUSH_RETRY_DELAY)
    return False


def push_checkpoint(model, tokenizer, stage_num, local_dir):
    repo_name = make_repo_name(stage_num, use_alt=False)
    if push_to_hub_safe(model, tokenizer, repo_name):
        return repo_name
    repo_name_alt = make_repo_name(stage_num, use_alt=True)
    logger.warning("⚠️  Primary name failed, fallback: %s", repo_name_alt)
    if push_to_hub_safe(model, tokenizer, repo_name_alt):
        return repo_name_alt
    logger.error("❌ ALL pushes failed. Local: %s", local_dir)
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # Log configuration summary
    logger.info("=" * 70)
    logger.info("HelixLM 100M Pretraining — d%d, cols=%d, seq=%d, n_loops=%d",
                D_MODEL, N_COLUMNS, SEQ_LEN, N_LOOPS)
    logger.info("=" * 70)
    logger.info("Run:        %s", RUN_TS)
    logger.info("Dataset:    %s", DATASET)
    logger.info("Revision:   %s", DATASET_REVISION or "provider default (not release-admissible)")
    logger.info("Config:     d=%d cols=%d heads=%d ffn=%.1f seq=%d loops=%d",
                D_MODEL, N_COLUMNS, N_HEADS, FFN_EXPANSION, SEQ_LEN, N_LOOPS)
    logger.info("Topology:   lateral=%.1f vertical=%.1f depth=%d",
                LATERAL_P, VERTICAL_P, VERTICAL_DEPTH)
    logger.info("Regularize: dropout=%.2f attn_dropout=%.2f wd=%.2f grad_buffer=%.2f",
                DROPOUT, ATTENTION_DROPOUT, WEIGHT_DECAY, GRAD_BUFFER_RATIO)
    logger.info("Batch:      %d x %d  grad_accum=%d (effective %d)",
                BATCH_SIZE, SEQ_LEN, GRAD_ACCUM, BATCH_SIZE * GRAD_ACCUM)
    logger.info("LR stages:  %s", LR_STAGES)
    logger.info("AMP:        %s (dtype=%s)", USE_AMP, AMP_DTYPE)
    logger.info("Data mode:  continuous windows (no document boundaries, no padding)")

    # ── Seed ────────────────────────────────────────────────────────────
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ── GPU check ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device:     %s", device)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU:        %s (%.1f GB VRAM)", gpu_name, gpu_mem_gb)
        if gpu_mem_gb < 40:
            logger.warning("⚠️  GPU < 40 GB VRAM: may need A100 (80GB) for this configuration.")

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer = HelixTokenizer(TOKENIZER_NAME)
    vocab_size = len(tokenizer)
    logger.info("Vocab:      %d", vocab_size)

    # ── Dataset (STREAMING MODE) ───────────────────────────────────────
    logger.info("Loading dataset (streaming=%s): %s", STREAMING, DATASET)
    try:
        load_kwargs = {"streaming": STREAMING}
        if DATASET_REVISION:
            load_kwargs["revision"] = DATASET_REVISION
        hf_ds = load_dataset(DATASET, **load_kwargs)
    except Exception as e:
        logger.error("❌ Failed to load dataset: %s", e)
        sys.exit(1)

    # Extract text columns as iterables (do not materialize)
    if PRETRAIN_STORE_DIR:
        train_iterable = None
        logger.info("Train store: %s", PRETRAIN_STORE_DIR)
    elif NUM_SAMPLES:
        train_iterable = hf_ds['train'].take(NUM_SAMPLES)['text']
    else:
        train_iterable = hf_ds['train']['text']
    val_iterable = hf_ds['validation']['text']

    logger.info("Train:      %s", "indexed sample store" if PRETRAIN_STORE_DIR else "IterableColumn (streaming)")
    logger.info("Val:        IterableColumn (streaming)")
    logger.info("Note:       Using PretrainTrainer with continuous token windows")

    # ── Training loop ────────────────────────────────────────────────────
    all_results = []
    prev_ckpt_dir = None

    for stage_idx, (lr, warmup) in enumerate(zip(LR_STAGES, WARMUP_STAGES)):
        stage_num = stage_idx + 1
        logger.info("\n" + "=" * 70)
        logger.info("EPOCH %d/%d | LR=%.1e | Warmup=%d", stage_num, EPOCHS, lr, warmup)
        logger.info("=" * 70)

        # ── Build config ─────────────────────────────────────────────
        cfg = HelixConfig.small_v2(
            vocab_size=vocab_size,
            tokenizer_name=TOKENIZER_NAME,
            d_model=D_MODEL,
            n_columns=N_COLUMNS,
            nodes_per_column=NODES_PER_COLUMN,
            n_heads=N_HEADS,
            n_loops=N_LOOPS,
            seq_len=SEQ_LEN,
            dropout=DROPOUT,
            attn_dropout=ATTENTION_DROPOUT,
            ffn_expansion=FFN_EXPANSION,
            weight_decay=WEIGHT_DECAY,
            grad_clip=GRAD_CLIP,
            grad_buffer_ratio=GRAD_BUFFER_RATIO,
            batch_size=BATCH_SIZE,
            lr=lr,
            warmup_steps=warmup,
            epochs=1,  # we train one epoch per stage
            use_cca=USE_CCA,
            use_ssm=USE_SSM,
            use_titans_memory=USE_TITANS,
            seed=SEED,
            device="auto",
            dtype=D_TYPE,
            amp_dtype=AMP_DTYPE,
            lateral_p=LATERAL_P,
            vertical_p=VERTICAL_P,
            vertical_depth=VERTICAL_DEPTH,
            attention_mode=ATTENTION_MODE,
            local_window=LOCAL_WINDOW,
            coarse_window=COARSE_WINDOW,
            compressed_windows=COMPRESSED_WINDOWS,
            compressed_views=COMPRESSED_VIEWS,
            consensus_type=CONSENSUS_TYPE,
            corrector_type=CORRECTOR_TYPE,
            tie_word_embeddings=True,
        )

        # Set token IDs
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.bos_token_id = tokenizer.bos_token_id

        logger.info("Topology check: lateral=%.1f vertical=%.1f depth=%d",
                    cfg.lateral_p, cfg.vertical_p, cfg.vertical_depth)

        # ── Create model (only once per stage) ────────────────────────
        model = HelixForCausalLM(cfg)
        graph_info = model.model.recurrent.graph.get_graph_info()
        logger.info("Graph: %d nodes, %d edges", graph_info["n_nodes"], graph_info["n_edges"])

        # Load previous checkpoint if available
        if prev_ckpt_dir is not None:
            st_path = os.path.join(prev_ckpt_dir, "model.safetensors")
            if os.path.exists(st_path):
                sd = load_safetensors(st_path)
            else:
                sd = torch.load(os.path.join(prev_ckpt_dir, "pytorch_model.bin"),
                                map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                logger.info("Load state_dict — missing: %d keys", len(missing))
            if unexpected:
                logger.info("Load state_dict — unexpected: %d keys", len(unexpected))

        params = model.count_parameters()
        logger.info("Parameters: %s total, %s trainable",
                    f"{params['total']:,}", f"{params['trainable']:,}")

        # ── Create PretrainTrainer ────────────────────────────────────
        stage_output_dir = str(OUTPUT_DIR / f"epoch{stage_num}")

        # PretrainTrainer uses continuous windows; these args are not used:
        #   min_tail_len, preprocess_batch_size, cleanup_shards
        trainer = PretrainTrainer(
            model=model,
            cfg=cfg,
            train_texts=train_iterable,
            train_store_dir=PRETRAIN_STORE_DIR,
            train_permutation_epoch=stage_idx,
            val_texts=val_iterable,            # IterableColumn — streaming mode
            tokenizer=tokenizer,
            output_dir=stage_output_dir,
            grad_accum_steps=GRAD_ACCUM,
            use_amp=USE_AMP,
            amp_dtype=AMP_DTYPE,
            verbose=True,
            num_workers=NUM_WORKERS,           # passed to DataLoader
        )

        # Constant LR: cosine with min_lr_ratio=1.0 = flat after warmup
        trainer._scheduler_min_lr = 1.0

        # ── Train ─────────────────────────────────────────────────────
        t0 = time.time()
        history = trainer.train(num_epochs=1)
        elapsed = time.time() - t0

        # ── Metrics ──────────────────────────────────────────────────
        train_loss = history.get("train_loss", [float("nan")])[-1]
        val_loss = history.get("val_loss", [float("nan")])[-1]
        train_ppl = math.exp(min(train_loss, 20)) if not math.isnan(train_loss) else float("nan")
        val_ppl = math.exp(min(val_loss, 20)) if not math.isnan(val_loss) else float("nan")

        logger.info("Epoch %d complete — Train PPL: %.2f | Val PPL: %.2f | Time: %.0fs (%.2f h)",
                    stage_num, train_ppl, val_ppl, elapsed, elapsed / 3600)

        # ── Save ─────────────────────────────────────────────────────
        canonical_ckpt = str(OUTPUT_DIR / f"ckpt_epoch{stage_num}")
        model.save_pretrained(canonical_ckpt)
        tokenizer.save_pretrained(canonical_ckpt)
        prev_ckpt_dir = canonical_ckpt

        # ── Push ─────────────────────────────────────────────────────
        if PUSH_TO_HUB:
            hub_repo = push_checkpoint(model, tokenizer, stage_num, canonical_ckpt)
        else:
            hub_repo = ""
            logger.info("Hub publication disabled; local checkpoint retained at %s", canonical_ckpt)

        # ── Track ────────────────────────────────────────────────────
        stage_result = {
            "epoch": stage_num,
            "lr": lr,
            "warmup_steps": warmup,
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "time_s": elapsed,
            "time_h": elapsed / 3600,
            "hub_repo": hub_repo,
            "local_ckpt": canonical_ckpt,
        }
        all_results.append(stage_result)

        with open(RESULTS_JSON, "w") as f:
            json.dump(all_results, f, indent=2)

        # Cleanup
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    total_time_h = sum(r["time_s"] for r in all_results) / 3600
    best_epoch = min(all_results, key=lambda r: r["val_ppl"])

    logger.info("\n" + "=" * 70)
    logger.info("🎉 TRAINING COMPLETE")
    logger.info("=" * 70)
    for r in all_results:
        logger.info("  Epoch %d | LR=%.0e | Val PPL=%7.2f | Time=%5.2f h | %s",
                    r["epoch"], r["lr"], r["val_ppl"], r["time_h"],
                    r["hub_repo"] or "LOCAL ONLY")
    logger.info("─" * 70)
    logger.info("Total time:     %.2f hours", total_time_h)
    logger.info("Best Val PPL:   %.2f (Epoch %d)", best_epoch["val_ppl"], best_epoch["epoch"])
    logger.info("Best repo:      %s", best_epoch["hub_repo"] or f"LOCAL: {best_epoch['local_ckpt']}")

    final = {
        "run_ts": RUN_TS,
        "config": {
            "d_model": D_MODEL, "n_columns": N_COLUMNS, "nodes_per_column": NODES_PER_COLUMN,
            "n_heads": N_HEADS, "ffn_expansion": FFN_EXPANSION,
            "seq_len": SEQ_LEN, "n_loops": N_LOOPS, "dropout": DROPOUT,
            "attn_dropout": ATTENTION_DROPOUT,
            "weight_decay": WEIGHT_DECAY, "grad_buffer_ratio": GRAD_BUFFER_RATIO,
            "lateral_p": LATERAL_P, "vertical_p": VERTICAL_P, "vertical_depth": VERTICAL_DEPTH,
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "lr_stages": LR_STAGES, "warmup_stages": WARMUP_STAGES,
            "streaming": STREAMING, "preprocess_batch_size": PREPROCESS_BATCH_SIZE,
            "dataset": DATASET, "dataset_revision": DATASET_REVISION,
            "pretrain_store_dir": PRETRAIN_STORE_DIR,
            "use_amp": USE_AMP, "amp_dtype": AMP_DTYPE,
            "data_mode": "continuous_windows",
        },
        "total_time_h": total_time_h,
        "best_val_ppl": best_epoch["val_ppl"],
        "best_epoch": best_epoch["epoch"],
        "epochs": all_results,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(final, f, indent=2)

    best_repo = best_epoch["hub_repo"]
    if best_repo:
        logger.info("\n📌 Load your best model:")
        logger.info('   model = AutoModelForCausalLM.from_pretrained("%s")', best_repo)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted. Checkpoints in: %s", OUTPUT_DIR)
        sys.exit(130)
    except Exception:
        logger.error("❌ Fatal error:\n%s", traceback.format_exc())
        sys.exit(1)
