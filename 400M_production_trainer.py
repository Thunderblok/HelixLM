"""
HelixLM 400M-Token Production Training — d256, n_heads=4, ffn=2.0, seq_len=128
Uses the official HelixLM Trainer API. Three stages (2e-3 → 1e-3 → 3e-4),
one epoch each. Pushes model + tokenizer to HF Hub after each stage.

Production config (vetted by 5M-token factorial ablation):
  d_model=256, n_heads=4, ffn_expansion=2.0, seq_len=128, n_loops=2
  dropout=0.1, grad_buffer_ratio=1/e, batch_size=32
  weight_decay=0.1, grad_clip=1.0
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
from transformers import AutoTokenizer
from datasets import load_dataset
from safetensors.torch import load_file as load_safetensors

from helix_lm.config import HelixConfig
from helix_lm.hf_model import HelixForCausalLM
from helix_lm.trainer import Trainer

# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
DATASET = "david-thrower/HelixLM-tiny-400.0Mt-730000pt-57143it-20260430"
SEED = 42
HF_USERNAME = "david-thrower"

# Architecture
D_MODEL = 256
N_HEADS = 4
FFN_EXPANSION = 2.0
SEQ_LEN = 128
N_LOOPS = 2
DROPOUT = 0.1
GRAD_BUFFER_RATIO = 1.0 / math.e

# Training
BATCH_SIZE = 32
GRAD_ACCUM = 1
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
LR_STAGES = [2e-3, 1e-3, 3e-4]
WARMUP_STAGES = [50, 10, 10]

# AMP
USE_AMP = True
AMP_DTYPE = "bfloat16"

# Flags
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
    print("   Set it with: export HF_TOKEN=hf_...", file=sys.stderr)
    print("   Or pass --secrets HF_TOKEN in hf jobs command.", file=sys.stderr)
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
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# REPO NAMING
# ═══════════════════════════════════════════════════════════════════════════
REPO_BASE = f"HelixLM-{RUN_TS}-d256-h4-nl2-ffn2-s128-14-8MP-400MT"
REPO_BASE_ALT = f"HelixLM-{RUN_TS}-prod"


def make_repo_name(epoch: int, use_alt: bool = False) -> str:
    """Build HF Hub repo name for a given epoch checkpoint."""
    base = REPO_BASE_ALT if use_alt else REPO_BASE
    return f"{HF_USERNAME}/{base}-ep{epoch}"


# ═══════════════════════════════════════════════════════════════════════════
# HUB PUSH HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def push_to_hub_safe(model, tokenizer, repo_name: str) -> bool:
    """Push model + tokenizer to HF Hub with retries. No commit message."""
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
                logger.info("   Retrying in %ds...", PUSH_RETRY_DELAY)
                time.sleep(PUSH_RETRY_DELAY)
    return False


def push_checkpoint(model, tokenizer, stage_num: int, local_dir: Path) -> str:
    """Push to hub with fallback repo name. Local save is handled by Trainer."""
    # Primary name
    repo_name = make_repo_name(stage_num, use_alt=False)
    if push_to_hub_safe(model, tokenizer, repo_name):
        return repo_name

    # Fallback shorter name
    repo_name_alt = make_repo_name(stage_num, use_alt=True)
    logger.warning("⚠️  Primary name failed, trying fallback: %s", repo_name_alt)
    if push_to_hub_safe(model, tokenizer, repo_name_alt):
        return repo_name_alt

    # Both failed — local checkpoint is your insurance
    logger.error("❌ ALL hub pushes failed! Local checkpoint: %s", local_dir)
    logger.error("   Manually upload:  hf upload %s %s", repo_name, local_dir)
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("HelixLM 400M Token Production Training  (Trainer API)")
    logger.info("=" * 70)
    logger.info("Run:        %s", RUN_TS)
    logger.info("Dataset:    %s", DATASET)
    logger.info("Config:     d=%d heads=%d ffn=%.1f seq=%d loops=%d",
                D_MODEL, N_HEADS, FFN_EXPANSION, SEQ_LEN, N_LOOPS)
    logger.info("Batch:      %d × %d  grad_accum=%d", BATCH_SIZE, SEQ_LEN, GRAD_ACCUM)
    logger.info("LR stages:  %.0e → %.0e → %.0e", *LR_STAGES)
    logger.info("AMP:        %s", AMP_DTYPE)
    logger.info("Trainer:    cosine schedule w/ warmup per stage")

    # ── Seed ────────────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ── SAFEGUARD 2: Check GPU ──────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device:     %s", device)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU:        %s (%.1f GB VRAM)", gpu_name, gpu_mem_gb)
        logger.info("BF16:       %s", torch.cuda.is_bf16_supported())

        if gpu_mem_gb < 20:
            logger.error("❌ GPU has <20 GB VRAM. This run needs L40S (48 GB) or similar.")
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
    )

    # ── Three-stage training ────────────────────────────────────────────
    all_results = []
    prev_ckpt_dir = None  # path to previous stage's saved model directory

    for stage_idx in range(3):
        stage_num = stage_idx + 1
        lr = LR_STAGES[stage_idx]
        warmup = WARMUP_STAGES[stage_idx]

        logger.info("\n" + "=" * 70)
        logger.info("STAGE %d/3 | LR=%.1e | Warmup=%d steps | Epochs=1",
                    stage_num, lr, warmup)
        logger.info("=" * 70)

        # ── Build config for this stage ─────────────────────────────
        cfg = HelixConfig.small_v2(lr=lr, epochs=1, warmup_steps=warmup, **cfg_kwargs)
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.bos_token_id = tokenizer.bos_token_id

        # ── Create model (fresh HelixGraph, same seed → same topology) ─
        model = HelixForCausalLM(cfg)

        # Load weights from previous stage if not first
        if prev_ckpt_dir is not None:
            st_path = os.path.join(prev_ckpt_dir, "model.safetensors")
            if os.path.exists(st_path):
                sd = load_safetensors(st_path)
            else:
                pt_path = os.path.join(prev_ckpt_dir, "pytorch_model.bin")
                sd = torch.load(pt_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                logger.info("Load state_dict — missing keys: %d", len(missing))
            if unexpected:
                logger.info("Load state_dict — unexpected keys: %d", len(unexpected))

        params = model.count_parameters()
        logger.info("Parameters: %s total, %s trainable",
                    f"{params['total']:,}", f"{params['trainable']:,}")

        # ── Create Trainer ──────────────────────────────────────────

        # Vary DataLoader shuffle order across stages.
        # Safe: HelixGraph topology is already built, weights already loaded.
        torch.manual_seed(SEED + stage_num * 1000)

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
            min_tail_len=1,          # keep all docs, matches ablation stride=SEQ_LEN
            verbose=True,
        )

        # ── Train one epoch ─────────────────────────────────────────
        t0 = time.time()
        history = trainer.train(num_epochs=1)
        elapsed = time.time() - t0

        # ── Extract metrics ─────────────────────────────────────────
        train_loss = history.get("train_loss", [float("nan")])[-1]
        val_loss = history.get("val_loss", [float("nan")])[-1]
        train_ppl = math.exp(min(train_loss, 20)) if not math.isnan(train_loss) else float("nan")
        val_ppl = math.exp(min(val_loss, 20)) if not math.isnan(val_loss) else float("nan")

        logger.info("Stage %d complete — Train PPL: %.2f | Val PPL: %.2f | Time: %.0fs (%.2f h)",
                    stage_num, train_ppl, val_ppl, elapsed, elapsed / 3600)

        # ── Locate saved checkpoint ─────────────────────────────────
        # Trainer.save_checkpoint uses self.model.save_pretrained(path)
        # After train(num_epochs=1), it saves to {output_dir}/final_model/
        ckpt_dir = os.path.join(stage_output_dir, "final_model")
        prev_ckpt_dir = ckpt_dir

        # Also save an explicit copy to our canonical path
        canonical_ckpt = str(OUTPUT_DIR / f"ckpt_stage{stage_num}")
        model.save_pretrained(canonical_ckpt)
        tokenizer.save_pretrained(canonical_ckpt)

        # ── Push to HF Hub ──────────────────────────────────────────
        hub_repo = push_checkpoint(model, tokenizer, stage_num, canonical_ckpt)

        # ── Track results ───────────────────────────────────────────
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

        # Incremental save
        with open(RESULTS_JSON, "w") as f:
            json.dump(all_results, f, indent=2)

        # Clean up model to free memory before next stage
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
    logger.info("Results JSON:   %s", RESULTS_JSON)
    logger.info("Log file:       %s", LOG_FILE)

    # Final structured output
    final = {
        "run_ts": RUN_TS,
        "config": {
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "ffn_expansion": FFN_EXPANSION,
            "seq_len": SEQ_LEN,
            "n_loops": N_LOOPS,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "lr_stages": LR_STAGES,
            "warmup_stages": WARMUP_STAGES,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "grad_buffer_ratio": GRAD_BUFFER_RATIO,
        },
        "total_time_h": total_time_h,
        "best_val_ppl": best_stage["val_ppl"],
        "best_stage": best_stage["stage"],
        "stages": all_results,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(final, f, indent=2)

    # Quick load reference
    best_repo = best_stage["hub_repo"]
    if best_repo:
        logger.info("\n📌 Load your best model:")
        logger.info("   from transformers import AutoModelForCausalLM")
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
