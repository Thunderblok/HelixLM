"""Ablation: grad_buffer_ratio on 5M-token dataset, small_v2 preset, L4 GPU."""
import math
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from datasets import load_dataset

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer

# ── Config ──────────────────────────────────────────────────────────
VALUES = [1.0 / math.e, 0.0, 0.184]  # 1/e, standard tying, half-1/e
PRESET = "small_v2"
EPOCHS = 3
EVAL_EVERY = 1
OUTPUT = "/app/ablation_results.json"

# small_v2 defaults: d_model=256, n_columns=2, n_heads=4, n_loops=1, seq_len=512
# ~15M params tied — chinchilla-adjacent for 400M tokens


def main():
    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)
    print(f"Vocab: {vocab_size}")

    # Load 5M-token dataset — use pretrain_train / pretrain_val splits
    print("Loading 5M-token dataset (pretrain_train split)...")
    ds_train = load_dataset(
        "david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427",
        split="pretrain_train",
        trust_remote_code=True,
    )
    ds_val = load_dataset(
        "david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427",
        split="pretrain_val",
        trust_remote_code=True,
    )
    train_texts = ds_train["text"]
    val_texts = ds_val["text"]
    print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")

    results = []

    for ratio in VALUES:
        label = f"buffer={ratio:.4f}"
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"{'='*60}")

        cfg = getattr(HelixConfig, PRESET)(
            vocab_size=vocab_size,
            tokenizer_name="gpt2",
            use_titans_memory=False,
            lr=3e-4,
            weight_decay=0.1,
            epochs=EPOCHS,
            warmup_steps=100,
            grad_clip=1.0,
            grad_buffer_ratio=ratio,
        )
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.bos_token_id = tokenizer.bos_token_id

        model = HelixForCausalLM(cfg)
        n_params = model.count_parameters()["total"]
        print(f"Parameters: {n_params:,}")

        trainer = Trainer(
            model=model,
            cfg=cfg,
            train_texts=train_texts,
            val_texts=val_texts,
            tokenizer=tokenizer,
            output_dir=f"./checkpoints_abl_{label.replace('=', '_').replace('.', '_')}",
            example_prompts=["Once upon a time", "The cat sat"],
            generated_example_length=20,
            grad_accum_steps=1,
            use_amp=False,  # L4 supports AMP but keep it simple for ablations
            verbose=True,
        )

        t0 = time.time()
        history = trainer.train(num_epochs=EPOCHS, eval_every=EVAL_EVERY)
        elapsed = time.time() - t0

        train_loss = history["train_loss"][-1] if history.get("train_loss") else float("inf")
        val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
        val_ppl = math.exp(min(val_loss, 20))

        result = {
            "grad_buffer_ratio": ratio,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "params": n_params,
            "time_seconds": elapsed,
            "history": {k: v[-1] if v else None for k, v in history.items()},
        }
        results.append(result)
        print(f"[{label}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} ppl={val_ppl:.2f} time={elapsed:.0f}s")

        # Save incrementally
        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print("ABLATION RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"  buffer={r['grad_buffer_ratio']:.4f}  train={r['train_loss']:.4f}  val={r['val_loss']:.4f}  ppl={r['val_ppl']:.2f}  time={r['time_seconds']:.0f}s")

    best = min(results, key=lambda r: r["val_loss"])
    print(f"\nBest: grad_buffer_ratio={best['grad_buffer_ratio']:.4f} (val_ppl={best['val_ppl']:.2f})")
    print(f"Results saved to {OUTPUT}")


if __name__ == "__main__":
    main()
