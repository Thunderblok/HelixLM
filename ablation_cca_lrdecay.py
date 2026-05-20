"""CCA × LR schedule factorial ablation on full 5M-token dataset."""
import math, time, json, sys, os, random

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from datasets import load_dataset

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer

DATASET = "david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427"
PRESET = "small_v2"
SEED = 42
EPOCHS = 3
OUTPUT = "/app/cca_lrdecay_results.json"

# Fixed winning config
BASE_KWARGS = dict(
    n_loops=2, dropout=0.1, weight_decay=0.1,
    warmup_steps=50, grad_clip=1.0,
    grad_buffer_ratio=1.0 / math.e, batch_size=8,
    use_ssm=False, use_titans_memory=False,
)

# CCA warmup: 10% of ~3,423 total optimizer steps
CCA_WARMUP = 342


def reset_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def build_cfg(vocab_size, lr, use_cca=False, tokenizer=None):
    tok = tokenizer or HelixTokenizer("gpt2")
    cfg = getattr(HelixConfig, PRESET)(
        vocab_size=vocab_size, tokenizer_name="gpt2",
        lr=lr, use_cca=use_cca,
        cca_warmup_steps=CCA_WARMUP if use_cca else 5000,
        cca_ramp_mode="quadratic", cca_min_scale=0.05,
        **BASE_KWARGS,
    )
    cfg.pad_token_id = tok.pad_token_id
    cfg.eos_token_id = tok.eos_token_id
    cfg.bos_token_id = tok.bos_token_id
    return cfg


def run_constant(train_texts, val_texts, tokenizer, vocab_size, lr, use_cca, label):
    """Standard 3-epoch training at constant LR."""
    reset_seeds()
    cfg = build_cfg(vocab_size, lr, use_cca, tokenizer)
    model = HelixForCausalLM(cfg)
    trainer = Trainer(model=model, cfg=cfg,
                      train_texts=train_texts, val_texts=val_texts,
                      tokenizer=tokenizer, output_dir=f"./ckpt_{label}",
                      grad_accum_steps=1, use_amp=False, verbose=True)
    t0 = time.time()
    history = trainer.train(num_epochs=3, eval_every=1)
    return {
        "label": label, "lr_schedule": "constant", "use_cca": use_cca,
        "train_loss": history["train_loss"][-1],
        "val_loss": history["val_loss"][-1],
        "val_ppl": math.exp(min(history["val_loss"][-1], 20)),
        "time_s": time.time() - t0,
    }


def run_two_stage(train_texts, val_texts, tokenizer, vocab_size, use_cca, label):
    """Epoch 1 at 2e-3, epochs 2-3 at 3e-4."""
    reset_seeds()
    # Stage 1
    cfg1 = build_cfg(vocab_size, 2e-3, use_cca)
    model = HelixForCausalLM(cfg1)
    trainer1 = Trainer(model=model, cfg=cfg1,
                       train_texts=train_texts, val_texts=val_texts,
                       tokenizer=tokenizer, output_dir=f"./ckpt_{label}_s1",
                       grad_accum_steps=1, use_amp=False, verbose=True)
    t0 = time.time()
    trainer1.train(num_epochs=1, eval_every=1)
    trainer1.save_checkpoint(1, "stage1")
    ckpt_path = f"./ckpt_{label}_s1/stage1/model.safetensors"

    # Stage 2
    cfg2 = build_cfg(vocab_size, 3e-4, use_cca)
    cfg2.warmup_steps = 10
    model2 = HelixForCausalLM(cfg2)
    from safetensors.torch import load_file
    sd = load_file(ckpt_path, device="cpu")
    model2.load_state_dict(sd, strict=False)
    trainer2 = Trainer(model=model2, cfg=cfg2,
                       train_texts=train_texts, val_texts=val_texts,
                       tokenizer=tokenizer, output_dir=f"./ckpt_{label}_s2",
                       grad_accum_steps=1, use_amp=False, verbose=True)
    history = trainer2.train(num_epochs=2, eval_every=1)
    return {
        "label": label, "lr_schedule": "two_stage_2e3_to_3e4", "use_cca": use_cca,
        "train_loss": history["train_loss"][-1],
        "val_loss": history["val_loss"][-1],
        "val_ppl": math.exp(min(history["val_loss"][-1], 20)),
        "time_s": time.time() - t0,
    }


def run_three_stage(train_texts, val_texts, tokenizer, vocab_size, label):
    """Epoch 1: 2e-3, epoch 2: 1e-3, epoch 3: 3e-4."""
    reset_seeds()
    # Stage 1: epoch 1 at 2e-3
    cfg1 = build_cfg(vocab_size, 2e-3, False)
    model = HelixForCausalLM(cfg1)
    trainer1 = Trainer(model=model, cfg=cfg1,
                       train_texts=train_texts, val_texts=val_texts,
                       tokenizer=tokenizer, output_dir=f"./ckpt_{label}_s1",
                       grad_accum_steps=1, use_amp=False, verbose=True)
    t0 = time.time()
    trainer1.train(num_epochs=1, eval_every=1)
    trainer1.save_checkpoint(1, "stage1")

    # Stage 2: epoch 2 at 1e-3
    cfg2 = build_cfg(vocab_size, 1e-3, False)
    cfg2.warmup_steps = 10
    model2 = HelixForCausalLM(cfg2)
    from safetensors.torch import load_file
    sd = load_file(f"./ckpt_{label}_s1/stage1/model.safetensors", device="cpu")
    model2.load_state_dict(sd, strict=False)
    trainer2 = Trainer(model=model2, cfg=cfg2,
                       train_texts=train_texts, val_texts=val_texts,
                       tokenizer=tokenizer, output_dir=f"./ckpt_{label}_s2",
                       grad_accum_steps=1, use_amp=False, verbose=True)
    trainer2.train(num_epochs=1, eval_every=1)
    trainer2.save_checkpoint(1, "stage2")

    # Stage 3: epoch 3 at 3e-4
    cfg3 = build_cfg(vocab_size, 3e-4, False)
    cfg3.warmup_steps = 10
    model3 = HelixForCausalLM(cfg3)
    sd = load_file(f"./ckpt_{label}_s2/stage2/model.safetensors", device="cpu")
    model3.load_state_dict(sd, strict=False)
    trainer3 = Trainer(model=model3, cfg=cfg3,
                       train_texts=train_texts, val_texts=val_texts,
                       tokenizer=tokenizer, output_dir=f"./ckpt_{label}_s3",
                       grad_accum_steps=1, use_amp=False, verbose=True)
    history = trainer3.train(num_epochs=1, eval_every=1)
    return {
        "label": label, "lr_schedule": "three_stage_2e3_1e3_3e4", "use_cca": False,
        "train_loss": history["train_loss"][-1],
        "val_loss": history["val_loss"][-1],
        "val_ppl": math.exp(min(history["val_loss"][-1], 20)),
        "time_s": time.time() - t0,
    }


def main():
    reset_seeds()

    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)

    # Load FULL splits
    ds_train = load_dataset(DATASET, split="pretrain_train")
    ds_val   = load_dataset(DATASET, split="pretrain_val")
    train_texts = ds_train["text"]
    val_texts   = ds_val["text"]
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # Resume from existing results if any
    if os.path.exists(OUTPUT):
        with open(OUTPUT, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results: {[r['label'] for r in results]}")
    else:
        results = []

    done_labels = {r["label"] for r in results}

    # Run 1: baseline (CCA off, constant LR)
    if "baseline_constant" not in done_labels:
        print("\n" + "="*60)
        print("RUN 1: baseline_constant")
        print("="*60)
        results.append(run_constant(train_texts, val_texts, tokenizer, vocab_size,
                                    2e-3, False, "baseline_constant"))
        with open(OUTPUT, "w") as f: json.dump(results, f, indent=2)

    # Run 2: CCA on, constant LR
    if "cca_constant" not in done_labels:
        print("\n" + "="*60)
        print("RUN 2: cca_constant")
        print("="*60)
        results.append(run_constant(train_texts, val_texts, tokenizer, vocab_size,
                                    2e-3, True, "cca_constant"))
        with open(OUTPUT, "w") as f: json.dump(results, f, indent=2)

    # Run 3: two-stage LR, CCA off
    if "lr2stage" not in done_labels:
        print("\n" + "="*60)
        print("RUN 3: lr2stage")
        print("="*60)
        results.append(run_two_stage(train_texts, val_texts, tokenizer, vocab_size,
                                     False, "lr2stage"))
        with open(OUTPUT, "w") as f: json.dump(results, f, indent=2)

    # Run 4: two-stage LR, CCA on
    if "cca_lr2stage" not in done_labels:
        print("\n" + "="*60)
        print("RUN 4: cca_lr2stage")
        print("="*60)
        results.append(run_two_stage(train_texts, val_texts, tokenizer, vocab_size,
                                     True, "cca_lr2stage"))
        with open(OUTPUT, "w") as f: json.dump(results, f, indent=2)

    # Run 5: three-stage LR, CCA off
    if "lr3stage" not in done_labels:
        print("\n" + "="*60)
        print("RUN 5: lr3stage")
        print("="*60)
        results.append(run_three_stage(train_texts, val_texts, tokenizer, vocab_size,
                                       "lr3stage"))
        with open(OUTPUT, "w") as f: json.dump(results, f, indent=2)

    # Report
    print(f"\n{'='*60}")
    print("CCA × LR SCHEDULE FACTORIAL RESULTS")
    for r in sorted(results, key=lambda x: x["val_loss"]):
        print(f"  {r['label']:20s}  cca={r['use_cca']}  lr={r['lr_schedule']:30s}  "
              f"ppl={r['val_ppl']:.2f}  time={r['time_s']:.0f}s")
    best = min(results, key=lambda r: r["val_loss"])
    print(f"\nBest: {best['label']} (val_ppl={best['val_ppl']:.2f})")


if __name__ == "__main__":
    main()
