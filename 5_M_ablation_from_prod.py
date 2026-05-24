"""
HelixLM 400M-Token Production Training — d384, n_heads=6, ffn=2.0, seq_len=128
Uses the official HelixLM Trainer API with a custom spike-LR schedule.

Spike schedule: constant LR at each stage's baseline, with brief 8× spikes
every ~1% of the epoch to escape local minima. No decay below baseline.

Production config:
  d_model=384, n_heads=6, ffn_expansion=2.0, seq_len=128, n_loops=2
  dropout=0.15, weight_decay=0.05, grad_buffer_ratio=0.0
  batch_size=32, grad_accum=2 (effective 64)
  use_cca=False, use_ssm=False, use_titans_memory=False
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
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from datasets import load_dataset
from safetensors.torch import load_file as load_safetensors

from helix_lm.config import HelixConfig
from helix_lm.hf_model import HelixForCausalLM
from helix_lm.trainer import Trainer

# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
DATASET = "david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427" # replace with "david-thrower/HelixLM-tiny-400.0Mt-730000pt-57143it-20260430" and paste over prod script
SEED = 42
HF_USERNAME = "david-thrower"

# Architecture — scaled up
D_MODEL = 384
N_HEADS = 6                     # d_model // 64
FFN_EXPANSION = 2.0
SEQ_LEN = 128
N_LOOPS = 2
DROPOUT = 0.15                  # ↑ from 0.1 — more noise against rank collapse
GRAD_BUFFER_RATIO = 0.0         # ↓ from 1/e — isolate buffer role in modality collapse

# Training — adjusted
BATCH_SIZE = 32
GRAD_ACCUM = 2
WEIGHT_DECAY = 0.05             # ↓ from 0.1 — alpha=5.38 said over-regularized
GRAD_CLIP = 1.0
LR_STAGES = [2e-3, 1e-3, 3e-4]
WARMUP_STAGES = [100, 10, 10]

# Spike LR schedule ("KITA" to nudge the optimizer out of crystallization rabbit holes)...
SPIKE_HEIGHT = 6.0              # LR multiplier during spike (5–20× range)
SPIKE_WIDTH = 100               # batches per spike (few dozen to hundred)
SPIKE_INTERVAL_PCT = 0.02       # spike every 2% of epoch

# AMP
USE_AMP = True
AMP_DTYPE = "bfloat16"

# Flags — held constant
USE_CCA = False
USE_SSM = False
USE_TITANS = False

# Hub push
PUSH_RETRY_ATTEMPTS = 3
PUSH_RETRY_DELAY = 30

# ═══════════════════════════════════════════════════════════════════════════
# SAFEGUARD 1: HF_TOKEN
# ═══════════════════════════════════════════════════════════════════════════
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("❌ HF_TOKEN environment variable is empty or not set!", file=sys.stderr)
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
OUTPUT_DIR = Path("/app/HelixLM/production_run")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / f"production_train_{RUN_TS}.log"
RESULTS_JSON = OUTPUT_DIR / f"production_results_{RUN_TS}.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# REPO NAMING
# ═══════════════════════════════════════════════════════════════════════════
REPO_BASE = f"HelixLM-{RUN_TS}-d384-h6-nl2-ffn2-s128-23-7MP-400MT"
REPO_BASE_ALT = f"HelixLM-{RUN_TS}-prod"


def make_repo_name(epoch: int, use_alt: bool = False) -> str:
    base = REPO_BASE_ALT if use_alt else REPO_BASE
    return f"{HF_USERNAME}/{base}-ep{epoch}"


# ═══════════════════════════════════════════════════════════════════════════
# SPIKE LR SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════

# def make_spike_schedule(warmup_steps: int, total_steps: int,
#                         spike_interval_steps: int, spike_width: int,
#                         spike_height: float):
#     """
#     Returns an lr_lambda function for a spike schedule.

#     - Warmup: linear 0 → 1.0 over warmup_steps
#     - Baseline: constant 1.0 (multiplied by base_lr)
#     - Spikes: triangular spikes reaching spike_height × base_lr,
#       lasting spike_width steps, every spike_interval_steps
#     - Terminal: returns to 1.0 at epoch end (no decay below baseline)
#     """
#     def lr_lambda(current_step: int) -> float:
#         if current_step < warmup_steps:
#             return float(current_step) / float(max(1, warmup_steps))

#         # Position within the current spike cycle
#         cycle_step = (current_step - warmup_steps) % spike_interval_steps

#         if cycle_step < spike_width:
#             # Inside a spike: smooth triangular shape
#             progress = cycle_step / spike_width           # 0.0 → 1.0
#             if progress < 0.5:
#                 # Ramp up: 1.0 → spike_height
#                 factor = 1.0 + (spike_height - 1.0) * (progress * 2.0)
#             else:
#                 # Ramp down: spike_height → 1.0
#                 factor = 1.0 + (spike_height - 1.0) * ((1.0 - progress) * 2.0)
#             return factor

#         # Baseline: constant 1.0
#         return 1.0

#     return lr_lambda


# def create_spike_scheduler(optimizer, warmup_steps: int, total_steps: int,
#                            spike_interval_pct: float, spike_width: int,
#                            spike_height: float) -> LambdaLR:
#     """Create a LambdaLR with the spike schedule."""
#     spike_interval_steps = max(1, int(total_steps * spike_interval_pct))
#     # Safety: ensure interval > width so spikes don't overlap
#     spike_interval_steps = max(spike_interval_steps, spike_width + 1)

#     lr_fn = make_spike_schedule(
#         warmup_steps=warmup_steps,
#         total_steps=total_steps,
#         spike_interval_steps=spike_interval_steps,
#         spike_width=spike_width,
#         spike_height=spike_height,
#     )
#     return LambdaLR(optimizer, lr_fn)


# ═══════════════════════════════════════════════════════════════════════════
# HUB PUSH HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def push_to_hub_safe(model, tokenizer, repo_name: str) -> bool:
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


def push_checkpoint(model, tokenizer, stage_num: int, local_dir: str) -> str:
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
    logger.info("=" * 70)
    logger.info("HelixLM 400M Token Production — d384 Spike-LR")
    logger.info("=" * 70)
    logger.info("Run:        %s", RUN_TS)
    logger.info("Dataset:    %s", DATASET)
    logger.info("Config:     d=%d heads=%d ffn=%.1f seq=%d loops=%d",
                D_MODEL, N_HEADS, FFN_EXPANSION, SEQ_LEN, N_LOOPS)
    logger.info("Regularize: dropout=%.2f wd=%.2f grad_buffer=%.2f",
                DROPOUT, WEIGHT_DECAY, GRAD_BUFFER_RATIO)
    logger.info("Batch:      %d × %d  grad_accum=%d (effective %d)",
                BATCH_SIZE, SEQ_LEN, GRAD_ACCUM, BATCH_SIZE * GRAD_ACCUM)
    logger.info("LR stages:  %.0e → %.0e → %.0e  + spikes %.0f× every %.0f%%",
                *LR_STAGES, SPIKE_HEIGHT, SPIKE_INTERVAL_PCT * 100)
    logger.info("AMP:        %s", AMP_DTYPE)

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
        if gpu_mem_gb < 20:
            logger.error("❌ GPU < 20 GB VRAM — need L40S (48 GB).")
            sys.exit(1)

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    logger.info("Vocab:      %d", vocab_size)

    # ── Dataset ─────────────────────────────────────────────────────────
    logger.info("Loading dataset: %s", DATASET)
    try:
        hf_ds = load_dataset(DATASET)
    except Exception as e:
        logger.error("❌ Failed to load dataset: %s", e)
        sys.exit(1)

    train_texts = [ex["text"] for ex in hf_ds["pretrain_train"]]
    val_texts = [ex["text"] for ex in hf_ds["pretrain_val"]]
    logger.info("Train docs: %s", f"{len(train_texts):,}")
    logger.info("Val docs:   %s", f"{len(val_texts):,}")

    # ── Shared config kwargs ────────────────────────────────────────────
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
        lateral_p=0.8, 
        vertical_depth=3
    )

    # ── Three-stage training ────────────────────────────────────────────
    all_results = []
    prev_ckpt_dir = None

    for stage_idx in range(1):
        stage_num = stage_idx + 1
        lr = LR_STAGES[stage_idx]
        warmup = WARMUP_STAGES[stage_idx]

        logger.info("\n" + "=" * 70)
        logger.info("STAGE %d/3 | LR=%.1e | Warmup=%d | Spikes %.0f× every %.0f%%",
                    stage_num, lr, warmup, SPIKE_HEIGHT, SPIKE_INTERVAL_PCT * 100)
        logger.info("=" * 70)

        # ── Build config ────────────────────────────────────────────
        cfg = HelixConfig.small_v2(lr=lr, epochs=1, warmup_steps=warmup, **cfg_kwargs)
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.bos_token_id = tokenizer.bos_token_id

        # ── Create model ─────────────────────────────────────────────
        model = HelixForCausalLM(cfg)

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
        stage_output_dir = str(OUTPUT_DIR / f"stage{stage_num}")
        trainer = Trainer(
            model=model,
            cfg=cfg,
            train_texts=train_texts,
            val_texts=val_texts,
            tokenizer=tokenizer,
            output_dir=stage_output_dir,
            grad_accum_steps=GRAD_ACCUM,
            use_amp=USE_AMP,
            amp_dtype=AMP_DTYPE,
            min_tail_len=SEQ_LEN // 4,   # matches ablation default (32)
            verbose=True,
        )
        trainer._scheduler_min_lr = 1.0

        # ── Inject spike LR scheduler BEFORE training ────────────────
        # Trainer creates its scheduler lazily in train_epoch() only if
        # self.scheduler is None. Pre-setting it here avoids any code
        # modification to the Trainer class.
        # steps_per_epoch = math.ceil(len(trainer.train_loader) / GRAD_ACCUM)
        # total_optim_steps = steps_per_epoch * 1  # cfg.epochs = 1

        # trainer.scheduler = create_spike_scheduler(
        #     optimizer=trainer.optimizer,
        #     warmup_steps=warmup,
        #     total_steps=total_optim_steps,
        #     spike_interval_pct=SPIKE_INTERVAL_PCT,
        #     spike_width=SPIKE_WIDTH,
        #     spike_height=SPIKE_HEIGHT,
        # )

        # ── Train one epoch ──────────────────────────────────────────
        t0 = time.time()
        history = trainer.train(num_epochs=1)
        elapsed = time.time() - t0

        # ── Extract metrics ──────────────────────────────────────────
        train_loss = history.get("train_loss", [float("nan")])[-1]
        val_loss = history.get("val_loss", [float("nan")])[-1]
        train_ppl = math.exp(min(train_loss, 20)) if not math.isnan(train_loss) else float("nan")
        val_ppl = math.exp(min(val_loss, 20)) if not math.isnan(val_loss) else float("nan")

        logger.info("Stage %d complete — Train PPL: %.2f | Val PPL: %.2f | Time: %.0fs (%.2f h)",
                    stage_num, train_ppl, val_ppl, elapsed, elapsed / 3600)

        # ── Save canonical checkpoint ────────────────────────────────
        canonical_ckpt = str(OUTPUT_DIR / f"ckpt_stage{stage_num}")
        model.save_pretrained(canonical_ckpt)
        tokenizer.save_pretrained(canonical_ckpt)
        prev_ckpt_dir = canonical_ckpt

        raise ValueError("Test ablation, don't push")
        # ── Push to HF Hub ───────────────────────────────────────────
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

        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    total_time_h = sum(r["time_s"] for r in all_results) / 3600
    best_stage = min(all_results, key=lambda r: r["val_ppl"])

    logger.info("\n" + "=" * 70)
    logger.info("🎉 PRODUCTION TRAINING COMPLETE")
    logger.info("=" * 70)
    for r in all_results:
        logger.info("  Stage %d | LR=%.0e | Val PPL=%7.2f | Time=%5.2f h | %s",
                    r["stage"], r["lr"], r["val_ppl"], r["time_h"],
                    r["hub_repo"] or "LOCAL ONLY")
    logger.info("─" * 70)
    logger.info("Total time:     %.2f hours", total_time_h)
    logger.info("Best Val PPL:   %.2f (Stage %d)", best_stage["val_ppl"], best_stage["stage"])
    logger.info("Best repo:      %s", best_stage["hub_repo"] or f"LOCAL: {best_stage['local_ckpt']}")

    final = {
        "run_ts": RUN_TS,
        "config": {
            "d_model": D_MODEL, "n_heads": N_HEADS, "ffn_expansion": FFN_EXPANSION,
            "seq_len": SEQ_LEN, "n_loops": N_LOOPS, "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY, "grad_buffer_ratio": GRAD_BUFFER_RATIO,
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "lr_stages": LR_STAGES,
            "spike_height": SPIKE_HEIGHT, "spike_width": SPIKE_WIDTH,
            "spike_interval_pct": SPIKE_INTERVAL_PCT,
        },
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
