"""Phase 1 hyperparameter tuning: LR × n_loops grid for HelixLM on 5M-token dataset."""
import math, time, json, sys, os, random

sys.path.insert(0, os.path.dirname(__file__))

import torch
from datasets import load_dataset

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer

# ── Config ──────────────────────────────────────────────────────────
DATASET_NAME = "david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427"
PRESET = "small_v2"
SEED = 42
SUBSET_SIZE = 1000   # samples for quick signal
EPOCHS = 3
OUTPUT = "/app/phase1_results.json"


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)

    # Load dataset — production pipeline uses the splits as-is
    ds_train = load_dataset(DATASET_NAME, split="pretrain_train")
    texts_all = [ex["text"] for ex in ds_train]

    # Subset for speed
    if SUBSET_SIZE and len(texts_all) > SUBSET_SIZE:
        texts_all = random.sample(texts_all, SUBSET_SIZE)

    split = int(len(texts_all) * 0.9)
    train_texts = texts_all[:split]
    val_texts = texts_all[split:]

    results = []

    # Phase 1: LR × n_loops grid
    runs = [
        (3e-4, 1, "lr3e-4_L1"),
        (2e-3, 1, "lr2e-3_L1"),
        (3e-4, 2, "lr3e-4_L2"),
        (2e-3, 2, "lr2e-3_L2"),
    ]

    for lr, n_loops, label in runs:
        print(f"\n{'='*60}")
        print(f"Running: {label} | lr={lr:.0e} | n_loops={n_loops}")
        print(f"{'='*60}")

        cfg = getattr(HelixConfig, PRESET)(
            vocab_size=vocab_size,
            tokenizer_name="gpt2",
            use_titans_memory=False,
            use_cca=False,
            use_ssm=False,
            lr=lr,
            n_loops=n_loops,
            dropout=0.05,
            weight_decay=0.1,
            epochs=EPOCHS,
            warmup_steps=50,
            grad_clip=1.0,
            grad_buffer_ratio=1.0 / math.e,
            batch_size=8,
        )
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.bos_token_id = tokenizer.bos_token_id

        model = HelixForCausalLM(cfg)
        n_params = model.count_parameters()["total"]

        trainer = Trainer(
            model=model,
            cfg=cfg,
            train_texts=train_texts,
            val_texts=val_texts,
            tokenizer=tokenizer,
            output_dir=f"./ckpt_{label}",
            example_prompts=["Once upon a time"],
            generated_example_length=15,
            grad_accum_steps=1,
            use_amp=False,
            verbose=True,
        )

        t0 = time.time()
        history = trainer.train(num_epochs=EPOCHS, eval_every=1)
        elapsed = time.time() - t0

        train_loss = history["train_loss"][-1]
        val_loss = history["val_loss"][-1] if history["val_loss"] else float("inf")
        val_ppl = math.exp(min(val_loss, 20))

        results.append({
            "label": label,
            "lr": lr,
            "n_loops": n_loops,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "params": n_params,
            "time_s": elapsed,
        })
        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=2)

    # Report
    print(f"\n{'='*60}")
    print("PHASE 1 RESULTS")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: x["val_loss"]):
        print(
            f"  {r['label']:12s}  lr={r['lr']:.0e}  loops={r['n_loops']}  "
            f"train={r['train_loss']:.4f}  val={r['val_loss']:.4f}  ppl={r['val_ppl']:.2f}  "
            f"time={r['time_s']:.0f}s"
        )
    best = min(results, key=lambda r: r["val_loss"])
    print(f"\nBest: {best['label']} (val_ppl={best['val_ppl']:.2f})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
