#!/usr/bin/env python3
"""
HelixLM 1.5B Ablations trainer — using PRE-SHARDED pre-tokenized data.

This is MUCH faster than streaming because:
1. Tokenization is done ONCE offline in parallel
2. Training just reads pre-tokenized tensors from disk (fast IO)
3. Exact batch counts are known (from the saved dataset)

Preprocessing (run once):
    python preprocess_1.5B_shards.py --output_dir ./preprocessed_1.5B

Then run this script for training.
"""
import math
import json
import sys
import os
import time
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

from helix_lm.config import HelixConfig
from helix_lm.hf_model import HelixForCausalLM
from helix_lm.trainer import Trainer
from helix_lm.dataset import create_helix_prechunked_loader

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Path to preprocessed data (run preprocess_1.5B_shards.py first)
PREPROCESSED_DATA_DIR = "./preprocessed_1.5B"

SEED = 42
HF_USERNAME = "david-thrower"

# Architecture - scaled for ~1.5B params
D_MODEL = 512
N_HEADS = D_MODEL // 64
FFN_EXPANSION = 2.5
SEQ_LEN = 512
N_LOOPS = 3
DROPOUT = 0.15
GRAD_BUFFER_RATIO = 0.0

# Attention - LINEAR ONLY
ATTENTION_MODE = "linear"
HYBRID_FULL_ATTENTION_INTERVAL = 0

# Topology — dense
LATERAL_P = 0.8
VERTICAL_P = 0.9
VERTICAL_DEPTH = 2

# Training
BATCH_SIZE = 16
GRAD_ACCUM = 4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
LR_STAGES = [1e-3, 5e-4, 1.5e-4]
WARMUP_STAGES = [200, 20, 20]

# KITA disabled
USE_KITA = False

# AMP
USE_AMP = True
AMP_DTYPE = "bfloat16"

# Flags
USE_CCA = False
USE_SSM = False
USE_TITANS = False

# Hub
PUSH_RETRY_ATTEMPTS = 3
PUSH_RETRY_DELAY = 30

# ═══════════════════════════════════════════════════════════════════════════
# SAFEGUARD
# ═══════════════════════════════════════════════════════════════════════════
HF_TOKEN = os.getenv("HF_TOKEN")

# ═══════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
OUTPUT_DIR = Path("/home/ubuntu/streaming-data-tests/HelixLM/outputs/1_5B_ablation_run")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "1_5_B_ablations-0002.log"
RESULTS_JSON = OUTPUT_DIR / f"1_5B_ablation_results_{RUN_TS}.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# REPO NAMING
# ═══════════════════════════════════════════════════════════════════════════
_DS = "1500MT"
REPO_BASE = f"HelixLM-{RUN_TS}-d{D_MODEL}-h{N_HEADS}-nl{N_LOOPS}-ffn{int(FFN_EXPANSION)}-s{SEQ_LEN}-{_DS}"


def make_repo_name(epoch: int) -> str:
    return f"{HF_USERNAME}/{REPO_BASE}-ep{epoch}"


# ═══════════════════════════════════════════════════════════════════════════
# HUB PUSH HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def push_to_hub_safe(model, tokenizer, repo_name):
    for attempt in range(1, PUSH_RETRY_ATTEMPTS + 1):
        try:
            logger.info("📤 Push: %s (attempt %d/%d)", repo_name, attempt, PUSH_RETRY_ATTEMPTS)
            model.push_to_hub(repo_name, token=HF_TOKEN)
            tokenizer.push_to_hub(repo_name, token=HF_TOKEN)
            logger.info("✅ Pushed: https://huggingface.co/%s", repo_name)
            return True
        except Exception as e:
            logger.warning("⚠️  Push attempt %d failed: %s", attempt, e)
            if attempt < PUSH_RETRY_ATTEMPTS:
                time.sleep(PUSH_RETRY_DELAY)
    return False


def push_checkpoint(model, tokenizer, stage_num, local_dir):
    repo_name = make_repo_name(stage_num)
    if push_to_hub_safe(model, tokenizer, repo_name):
        return repo_name
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("HelixLM 1.5B Ablations (PRE-SHARDED) — d512, linear attention, n_loops=%d", N_LOOPS)
    logger.info("=" * 70)
    logger.info("Run:        %s", RUN_TS)
    logger.info("Dataset:    %s", PREPROCESSED_DATA_DIR)
    logger.info("Config:     d=%d heads=%d ffn=%.1f seq=%d loops=%d",
                D_MODEL, N_HEADS, FFN_EXPANSION, SEQ_LEN, N_LOOPS)
    logger.info("Attention:  mode=%s", ATTENTION_MODE)
    logger.info("Topology:   lateral=%.1f vertical=%.1f depth=%d",
                LATERAL_P, VERTICAL_P, VERTICAL_DEPTH)
    logger.info("Regularize: dropout=%.2f wd=%.2f grad_buffer=%.2f",
                DROPOUT, WEIGHT_DECAY, GRAD_BUFFER_RATIO)
    logger.info("Batch:      %d x %d  grad_accum=%d (effective %d)",
                BATCH_SIZE, SEQ_LEN, GRAD_ACCUM, BATCH_SIZE * GRAD_ACCUM)
    logger.info("LR:         %.0e constant  |  KITA: %s",
                LR_STAGES[0], "ON" if USE_KITA else "OFF")
    logger.info("AMP:        %s", AMP_DTYPE)

    # Check preprocessed data exists
    train_dir = Path(PREPROCESSED_DATA_DIR) / "train"
    val_dir = Path(PREPROCESSED_DATA_DIR) / "val"
    
    if not train_dir.exists():
        logger.error("❌ Preprocessed data not found: %s", train_dir)
        logger.error("   Run: python preprocess_1.5B_shards.py --output_dir %s", PREPROCESSED_DATA_DIR)
        sys.exit(1)

    # ── Seed ────────────────────────────────────────────────────────────
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

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    logger.info("Vocab:      %d", vocab_size)

    # ── Create data loaders (FAST - reads pre-tokenized data from disk) ──
    logger.info("Loading pre-tokenized data from: %s", PREPROCESSED_DATA_DIR)
    
    start_time = time.time()
    train_loader = create_helix_prechunked_loader(
        str(train_dir),
        tokenizer,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2,  # Can use workers since data is on disk
    )
    load_time = time.time() - start_time
    logger.info("Train data loaded in %.1fs — %d batches", load_time, len(train_loader))
    
    val_loader = None
    if val_dir.exists():
        val_loader = create_helix_prechunked_loader(
            str(val_dir),
            tokenizer,
            seq_len=SEQ_LEN,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=2,
        )
        logger.info("Val data loaded — %d batches", len(val_loader))

    # ── Shared config ───────────────────────────────────────────────────
    cfg_kwargs = dict(
        vocab_size=vocab_size,
        tokenizer_name="gpt2",
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_loops=N_LOOPS,
        seq_len=SEQ_LEN,
        dropout=DROPOUT,
        ffn_expansion=FFN_EXPANSION,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        grad_buffer_ratio=GRAD_BUFFER_RATIO,
        batch_size=BATCH_SIZE,
        use_cca=USE_CCA,
        use_ssm=USE_SSM,
        use_titans_memory=USE_TITANS,
        seed=SEED,
        device="auto",
        amp_dtype=AMP_DTYPE,
        lateral_p=LATERAL_P,
        vertical_p=VERTICAL_P,
        vertical_depth=VERTICAL_DEPTH,
        attention_mode=ATTENTION_MODE,
        hybrid_full_attention_interval=HYBRID_FULL_ATTENTION_INTERVAL,
    )

    # ── Training ────────────────────────────────────────────────────────
    all_results = []
    prev_ckpt_dir = None

    for stage_idx in range(3):
        stage_num = stage_idx + 1
        lr = LR_STAGES[stage_idx]
        warmup = WARMUP_STAGES[stage_idx]

        logger.info("\n" + "=" * 70)
        logger.info("STAGE %d/3 | LR=%.1e | Warmup=%d", stage_num, lr, warmup)
        logger.info("=" * 70)

        # ── Build config ────────────────────────────────────────────
        cfg = HelixConfig(
            lr=lr, epochs=1, warmup_steps=warmup,
            **cfg_kwargs
        )
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.bos_token_id = tokenizer.bos_token_id

        # ── Create model ─────────────────────────────────────────────
        model = HelixForCausalLM(cfg)
        graph_info = model.model.recurrent.graph.get_graph_info()
        logger.info("Graph: %d nodes, %d edges", graph_info["n_nodes"], graph_info["n_edges"])

        params = model.count_parameters()
        logger.info("Parameters: %s total, %s trainable",
                    f"{params['total']:,}", f"{params['trainable']:,}")

        # ── Create Trainer ───────────────────────────────────────────
        stage_output_dir = str(OUTPUT_DIR / f"stage{stage_num}")
        trainer = Trainer(
            model=model,
            cfg=cfg,
            tokenizer=tokenizer,
            output_dir=stage_output_dir,
            grad_accum_steps=GRAD_ACCUM,
            use_amp=USE_AMP,
            amp_dtype=AMP_DTYPE,
            min_tail_len=SEQ_LEN // 4,
            verbose=True,
        )
        
        # Inject pre-sharded loaders
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader
        # Mark as non-iterable (map-style) since we have exact counts
        trainer._is_iterable = False
        trainer._cached_dataset_length = len(train_loader)

        # Constant LR: cosine with min_lr_ratio=1.0 = flat after warmup
        trainer._scheduler_min_lr = 1.0

        # ── Train ────────────────────────────────────────────────────
        t0 = time.time()
        history = trainer.train(num_epochs=1)
        elapsed = time.time() - t0

        # ── Metrics ──────────────────────────────────────────────────
        train_loss = history.get("train_loss", [float("nan")])[-1]
        val_loss = history.get("val_loss", [float("nan")])[-1]
        train_ppl = math.exp(min(train_loss, 20)) if not math.isnan(train_loss) else float("nan")
        val_ppl = math.exp(min(val_loss, 20)) if not math.isnan(val_loss) else float("nan")

        logger.info("Stage %d complete — Train PPL: %.2f | Val PPL: %.2f | Time: %.0fs (%.2f h)",
                    stage_num, train_ppl, val_ppl, elapsed, elapsed / 3600)

        # ── Save ─────────────────────────────────────────────────────
        canonical_ckpt = str(OUTPUT_DIR / f"ckpt_stage{stage_num}")
        model.save_pretrained(canonical_ckpt)
        tokenizer.save_pretrained(canonical_ckpt)
        prev_ckpt_dir = canonical_ckpt

        # ── Push ─────────────────────────────────────────────────────
        hub_repo = push_checkpoint(model, tokenizer, stage_num, canonical_ckpt)

        # ── Track ────────────────────────────────────────────────────
        stage_result = {
            "stage": stage_num,
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

        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    total_time_h = sum(r["time_s"] for r in all_results) / 3600
    best_stage = min(all_results, key=lambda r: r["val_ppl"])

    logger.info("\n" + "=" * 70)
    logger.info("🎉 TRAINING COMPLETE")
    logger.info("=" * 70)
    for r in all_results:
        logger.info("  Stage %d | LR=%.0e | Val PPL=%7.2f | Time=%5.2f h | %s",
                    r["stage"], r["lr"], r["val_ppl"], r["time_h"],
                    r["hub_repo"] or "LOCAL ONLY")
    logger.info("─" * 70)
    logger.info("Total time:     %.2f hours", total_time_h)
    logger.info("Best Val PPL:   %.2f (Stage %d)", best_stage["val_ppl"], best_stage["stage"])
    logger.info("Best repo:      %s", best_stage["hub_repo"] or f"LOCAL: {best_stage['local_ckpt']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted. Checkpoints in: %s", OUTPUT_DIR)
        sys.exit(130)
    except Exception:
        logger.error("❌ Fatal error:\n%s", traceback.format_exc())
        sys.exit(1)
