"""
HelixLM 87M/3B Token Production trainer — d1024, n_columns=2, ffn=2.7, seq_len=1024, n_loops=5
Constant LR per stage, dense topology. Streaming dataset for 3B tokens.

Config (n_loops=5 FEASIBILITY variant):
  d_model=1024, n_columns=2, n_heads=16, ffn_expansion=2.7, seq_len=1024, n_loops=5
  dropout=0.1, weight_decay=0.05, grad_buffer_ratio=0.0
  batch_size=32, grad_accum=2 (effective 64)
  lateral_p=0.8, vertical_p=0.9, vertical_depth=2
  use_cca=False, use_ssm=False, use_titans_memory=False

FEASIBILITY NOTES for n_loops=5:
  + Increases recurrent depth by 25% (5 vs 4 loops)
  + May improve long-range dependencies at 1024 context
  + More computation per forward pass (slower training)
  ~ Watch for: memory usage, convergence speed, training stability
  ~ Compare PPL vs n_loops=4 baseline
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
from datasets import load_dataset
from safetensors.torch import load_file as load_safetensors

from helix_lm.config import HelixConfig
from helix_lm.hf_model import HelixForCausalLM
from helix_lm.trainer import Trainer

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# 3B token dataset with train/val splits — STREAMING MODE
DATASET = "david-thrower/helixlm87M-3Btoken-pretrain-dataset-v1"

SEED = 42
HF_USERNAME = "david-thrower"

# Architecture — Optimized configuration with n_loops=5 FEASIBILITY TEST
D_MODEL = 1024                              # High dim > many columns
N_COLUMNS = 2                               # Simple graph, faster training
NODES_PER_COLUMN = (3, 3)                   # Balanced 2-column graph
N_HEADS = D_MODEL // 64                     # 16 (1024/64 per head)
FFN_EXPANSION = 2.7                         # Per PanGu-π
SEQ_LEN = 1024                              # Target context length
N_LOOPS = 4                                 # FEASIBILITY: Increased from 4
DROPOUT = 0.1                               # Slight reduction at scale
GRAD_BUFFER_RATIO = 0.0

# Topology — dense (validated at 50M)
LATERAL_P = 0.8
VERTICAL_P = 0.9
VERTICAL_DEPTH = 2

# Training
BATCH_SIZE = 16
GRAD_ACCUM = 4                             # effective batch = 64
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
LR_STAGES = [2e-3, 1e-3, 3e-4]
WARMUP_STAGES = [100, 10, 10]

# KITA spike scheduler — disabled, uncomment if rabbit holes appear
USE_KITA = False
SPIKE_HEIGHT = 6.0
SPIKE_WIDTH = 100
SPIKE_INTERVAL_PCT = 0.02

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

# Streaming dataset settings
STREAMING = True                            # Load dataset as streaming (IterableDataset)
PREPROCESS_BATCH_SIZE = 1000                # Batch size for streaming preprocessing
CLEANUP_SHARDS = True                       # Clean up temporary shards after training

# ═══════════════════════════════════════════════════════════════════════════
# SAFEGUARD
# ═══════════════════════════════════════════════════════════════════════════
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("❌ HF_TOKEN environment variable is empty or not set!", file=sys.stderr)
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
OUTPUT_DIR = Path("production_run_87M_1024_nloops5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / f"production_train_87M_1024_nl5_{RUN_TS}.log"
RESULTS_JSON = OUTPUT_DIR / f"production_results_87M_1024_nl5_{RUN_TS}.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Dynamic repo naming ──────────────────────────────────────────────────
REPO_BASE = (
    f"HelixLM-87M-nl5-{RUN_TS}-d{D_MODEL}-col{N_COLUMNS}-h{N_HEADS}-nl{N_LOOPS}"
    f"-ffn{int(FFN_EXPANSION)}-s{SEQ_LEN}-3BT"
)
REPO_BASE_ALT = f"HelixLM-87M-nl5-{RUN_TS}-prod"


def make_repo_name(epoch: int, use_alt: bool = False) -> str:
    base = REPO_BASE_ALT if use_alt else REPO_BASE
    return f"{HF_USERNAME}/{base}-ep{epoch}"


# ── KITA scheduler (disabled by default) ─────────────────────────────────
if USE_KITA:
    from torch.optim.lr_scheduler import LambdaLR

    def _make_spike_schedule(warmup_steps, total_steps,
                             spike_interval_steps, spike_width, spike_height):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            cycle_step = (current_step - warmup_steps) % spike_interval_steps
            if cycle_step < spike_width:
                progress = cycle_step / spike_width
                if progress < 0.5:
                    return 1.0 + (spike_height - 1.0) * (progress * 2.0)
                else:
                    return 1.0 + (spike_height - 1.0) * ((1.0 - progress) * 2.0)
            return 1.0
        return lr_lambda

    def _create_spike_scheduler(optimizer, warmup_steps, total_steps,
                                spike_interval_pct, spike_width, spike_height):
        interval = max(1, int(total_steps * spike_interval_pct))
        interval = max(interval, spike_width + 1)
        return LambdaLR(optimizer, _make_spike_schedule(
            warmup_steps, total_steps, interval, spike_width, spike_height))


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
    # Calculate effective stride (Trainer will auto-apply: 512 for seq_len>512)
    effective_stride = 512 if SEQ_LEN > 512 else SEQ_LEN
    overlap_pct = (1 - effective_stride/SEQ_LEN) * 100
    
    logger.info("=" * 70)
    logger.info("HelixLM 87M Training (n_loops=5 FEASIBILITY) — d%d, cols=%d, seq=%d", D_MODEL, N_COLUMNS, SEQ_LEN)
    logger.info("⚠️  FEASIBILITY VARIANT: n_loops=5 (25%% more recurrent depth)")
    logger.info("=" * 70)
    logger.info("Run:        %s", RUN_TS)
    logger.info("Dataset:    %s", DATASET)
    logger.info("Config:     d=%d cols=%d heads=%d ffn=%.1f seq=%d loops=%d",
                D_MODEL, N_COLUMNS, N_HEADS, FFN_EXPANSION, SEQ_LEN, N_LOOPS)
    logger.info("Topology:   lateral=%.1f vertical=%.1f depth=%d",
                LATERAL_P, VERTICAL_P, VERTICAL_DEPTH)
    logger.info("Chunking:   stride=%d (%.0f%% overlap) [auto for seq_len=%d]",
                effective_stride, overlap_pct, SEQ_LEN)
    logger.info("Regularize: dropout=%.2f wd=%.2f grad_buffer=%.2f",
                DROPOUT, WEIGHT_DECAY, GRAD_BUFFER_RATIO)
    logger.info("Batch:      %d x %d  grad_accum=%d (effective %d)",
                BATCH_SIZE, SEQ_LEN, GRAD_ACCUM, BATCH_SIZE * GRAD_ACCUM)
    logger.info("LR:         %.0e constant  |  KITA: %s",
                LR_STAGES[0], "ON" if USE_KITA else "OFF")
    logger.info("AMP:        %s", AMP_DTYPE)
    logger.info("Streaming:  %s (preprocess_batch=%d)", STREAMING, PREPROCESS_BATCH_SIZE)
    
    logger.info("\n📊 FEASIBILITY COMPARISON (n_loops=5 vs n_loops=4):")
    logger.info("   Pros:   +25%% recurrent depth, better long-range deps")
    logger.info("   Cons:   Slower training (more compute/forward pass)")
    logger.info("   Watch:  Memory usage, convergence, stability")

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
        if gpu_mem_gb < 40:
            logger.warning("⚠️  GPU < 40 GB VRAM — d1024 with seq_len=1024 needs A100 (80GB).")
        if gpu_mem_gb < 80:
            logger.warning("⚠️  n_loops=5 increases memory usage; A100 (80GB) strongly recommended.")

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    logger.info("Vocab:      %d", vocab_size)

    # ── Dataset (STREAMING MODE) ────────────────────────────────────────
    logger.info("Loading dataset (streaming=True): %s", DATASET)
    try:
        # Load as streaming dataset — returns IterableDataset
        hf_ds = load_dataset(DATASET, streaming=STREAMING)
    except Exception as e:
        logger.error("❌ Failed to load dataset: %s", e)
        sys.exit(1)

    # Extract the 'text' column as IterableColumn — DO NOT MATERIALIZE
    # The Trainer will handle streaming preprocessing via create_unified_data_loader
    # which uses _handle_streaming_iterable to shard/process on-the-fly
    train_iterable = hf_ds['train']['text']  # datasets.iterable_dataset.IterableColumn
    val_iterable = hf_ds['validation']['text']  # datasets.iterable_dataset.IterableColumn
    
    logger.info("Train:      IterableColumn (streaming)")
    logger.info("Val:        IterableColumn (streaming)")
    logger.info("Note:       Dataset > memory; using sharded preprocessing")

    # ── Shared config kwargs ────────────────────────────────────────────
    cfg_kwargs = dict(
        vocab_size=vocab_size,
        tokenizer_name="gpt2",
        d_model=D_MODEL,
        n_columns=N_COLUMNS,
        nodes_per_column=NODES_PER_COLUMN,
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
    )

    # ── Training ────────────────────────────────────────────────────────
    all_results = []
    prev_ckpt_dir = None

    for stage_idx in range(1):          # 1 epoch = Chinchilla-optimal for 3B tokens
        stage_num = stage_idx + 1
        lr = LR_STAGES[stage_idx]
        warmup = WARMUP_STAGES[stage_idx]

        logger.info("\n" + "=" * 70)
        kita_str = f"KITA {SPIKE_HEIGHT:.0f}x every {SPIKE_INTERVAL_PCT*100:.0f}%" if USE_KITA else "constant"
        logger.info("STAGE %d/1 | LR=%.1e | Warmup=%d | %s",
                    stage_num, lr, warmup, kita_str)
        logger.info("=" * 70)

        # ── Build config ────────────────────────────────────────────
        cfg = HelixConfig(**cfg_kwargs, lr=lr, epochs=1, warmup_steps=warmup)
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.bos_token_id = tokenizer.bos_token_id

        logger.info("Topology check: lateral=%.1f vertical=%.1f depth=%d",
                    cfg.lateral_p, cfg.vertical_p, cfg.vertical_depth)

        # ── Create model ─────────────────────────────────────────────
        model = HelixForCausalLM(cfg)
        graph_info = model.model.recurrent.graph.get_graph_info()
        logger.info("Graph: %d nodes, %d edges", graph_info["n_nodes"], graph_info["n_edges"])

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

        # ── Create Trainer ───────────────────────────────────────────
        # Pass IterableColumn directly — Trainer will use streaming path
        stage_output_dir = str(OUTPUT_DIR / f"stage{stage_num}")
        trainer = Trainer(
            model=model,
            cfg=cfg,
            train_texts=train_iterable,        # IterableColumn — streaming mode
            val_texts=val_iterable,            # IterableColumn — streaming mode
            tokenizer=tokenizer,
            output_dir=stage_output_dir,
            grad_accum_steps=GRAD_ACCUM,
            use_amp=USE_AMP,
            amp_dtype=AMP_DTYPE,
            min_tail_len=SEQ_LEN // 4,
            verbose=True,
            # Streaming-specific options
            preprocess_batch_size=PREPROCESS_BATCH_SIZE,
            cleanup_shards=CLEANUP_SHARDS,
        )

        # Constant LR: cosine with min_lr_ratio=1.0 = flat after warmup
        trainer._scheduler_min_lr = 1.0

        # KITA override (if enabled)
        if USE_KITA:
            # For streaming, estimate steps based on token count
            # 3B tokens / (64 batch * 1024 seq_len) ≈ ~45k steps
            estimated_steps = 3_000_000_000 // (BATCH_SIZE * GRAD_ACCUM * SEQ_LEN)
            logger.info("KITA estimated steps (3B tokens): ~%d", estimated_steps)
            trainer.scheduler = _create_spike_scheduler(
                optimizer=trainer.optimizer,
                warmup_steps=warmup,
                total_steps=estimated_steps,
                spike_interval_pct=SPIKE_INTERVAL_PCT,
                spike_width=SPIKE_WIDTH,
                spike_height=SPIKE_HEIGHT,
            )
            logger.info("KITA scheduler injected: %.0fx spikes, width=%d, interval=%.0f%%",
                        SPIKE_HEIGHT, SPIKE_WIDTH, SPIKE_INTERVAL_PCT * 100)

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
    logger.info("🎉 TRAINING COMPLETE (n_loops=5 FEASIBILITY)")
    logger.info("=" * 70)
    for r in all_results:
        logger.info("  Stage %d | LR=%.0e | Val PPL=%7.2f | Time=%5.2f h | %s",
                    r["stage"], r["lr"], r["val_ppl"], r["time_h"],
                    r["hub_repo"] or "LOCAL ONLY")
    logger.info("─" * 70)
    logger.info("Total time:     %.2f hours", total_time_h)
    logger.info("Best Val PPL:   %.2f (Stage %d)", best_stage["val_ppl"], best_stage["stage"])
    logger.info("Best repo:      %s", best_stage["hub_repo"] or f"LOCAL: {best_stage['local_ckpt']}")
    
    logger.info("\n📊 FEASIBILITY SUMMARY:")
    logger.info("   Compare against n_loops=4 baseline:")
    logger.info("   - Training time increase: check elapsed time")
    logger.info("   - PPL improvement: %.2f vs baseline", best_stage["val_ppl"])
    logger.info("   - Memory overhead: check peak GPU usage")
    logger.info("   - Convergence: compare loss curves")

    final = {
        "run_ts": RUN_TS,
        "config": {
            "d_model": D_MODEL, "n_columns": N_COLUMNS, "nodes_per_column": NODES_PER_COLUMN,
            "n_heads": N_HEADS, "ffn_expansion": FFN_EXPANSION,
            "seq_len": SEQ_LEN, "n_loops": N_LOOPS, "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY, "grad_buffer_ratio": GRAD_BUFFER_RATIO,
            "lateral_p": LATERAL_P, "vertical_p": VERTICAL_P, "vertical_depth": VERTICAL_DEPTH,
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "lr": LR_STAGES[0], "kita": USE_KITA,
            "streaming": STREAMING, "preprocess_batch_size": PREPROCESS_BATCH_SIZE,
        },
        "variant": "n_loops_5_feasibility",
        "feasibility_notes": [
            "25% more recurrent depth vs baseline (n_loops=4)",
            "Expected: better long-range dependencies",
            "Tradeoff: slower training (more compute/forward)",
            "Watch: memory, convergence, stability"
        ],
        "total_time_h": total_time_h,
        "best_val_ppl": best_stage["val_ppl"],
        "best_stage": best_stage["stage"],
        "stages": all_results,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(final, f, indent=2)

    best_repo = best_stage["hub_repo"]
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
