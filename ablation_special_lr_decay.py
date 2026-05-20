"""Special test: Two-stage LR decay schedule.

Train epoch 1 at high LR, then epochs 2-3 at lower LR.
Uses the winning config from Phases 1-3: LR=2e-3 base, n_loops=2, dropout=0.1, no CCA.
"""
import math, time, json, sys, os, random

sys.path.insert(0, os.path.dirname(__file__))

import torch
from datasets import load_dataset

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer

DATASET_NAME = "david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427"
PRESET = "small_v2"
SEED = 42
SUBSET_SIZE = 1000
OUTPUT = "/app/special_lr_decay_results.json"


def train_two_stage(train_texts, val_texts, tokenizer, vocab_size,
                     lr_epoch1, lr_epochs23, label):
    """Train 1 epoch at lr_epoch1, then 2 epochs at lr_epochs23."""
    print(f"\n{'='*60}")
    print(f"Running: {label} | epoch1_lr={lr_epoch1:.0e} | epochs2-3_lr={lr_epochs23:.0e}")
    print(f"{'='*60}")

    # Phase 1: 1 epoch at high LR
    cfg1 = getattr(HelixConfig, PRESET)(
        vocab_size=vocab_size,
        tokenizer_name="gpt2",
        use_titans_memory=False,
        use_cca=False,
        use_ssm=False,
        lr=lr_epoch1,
        n_loops=2,
        dropout=0.1,
        weight_decay=0.1,
        epochs=1,
        warmup_steps=50,
        grad_clip=1.0,
        grad_buffer_ratio=1.0 / math.e,
        batch_size=8,
    )
    cfg1.pad_token_id = tokenizer.pad_token_id
    cfg1.eos_token_id = tokenizer.eos_token_id
    cfg1.bos_token_id = tokenizer.bos_token_id

    model = HelixForCausalLM(cfg1)
    n_params = model.count_parameters()["total"]

    trainer = Trainer(
        model=model,
        cfg=cfg1,
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        output_dir=f"./ckpt_{label}_stage1",
        example_prompts=["Once upon a time"],
        generated_example_length=15,
        grad_accum_steps=1,
        use_amp=False,
        verbose=True,
    )

    t0 = time.time()
    history1 = trainer.train(num_epochs=1, eval_every=1)
    stage1_time = time.time() - t0

    # Save checkpoint
    ckpt_path = f"./ckpt_{label}_stage1/best_model"
    trainer.save_checkpoint(1, "best_model")

    # Phase 2: 2 epochs at lower LR — reload model from checkpoint
    cfg2 = getattr(HelixConfig, PRESET)(
        vocab_size=vocab_size,
        tokenizer_name="gpt2",
        use_titans_memory=False,
        use_cca=False,
        use_ssm=False,
        lr=lr_epochs23,
        n_loops=2,
        dropout=0.1,
        weight_decay=0.1,
        epochs=2,
        warmup_steps=10,  # short warmup for restart
        grad_clip=1.0,
        grad_buffer_ratio=1.0 / math.e,
        batch_size=8,
    )
    cfg2.pad_token_id = tokenizer.pad_token_id
    cfg2.eos_token_id = tokenizer.eos_token_id
    cfg2.bos_token_id = tokenizer.bos_token_id

    model2 = HelixForCausalLM(cfg2)
    # Load weights from stage 1
    state_dict = torch.load(os.path.join(ckpt_path, "pytorch_model.bin"),
                           map_location="cpu")
    model2.load_state_dict(state_dict, strict=False)

    trainer2 = Trainer(
        model=model2,
        cfg=cfg2,
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        output_dir=f"./ckpt_{label}_stage2",
        example_prompts=["Once upon a time"],
        generated_example_length=15,
        grad_accum_steps=1,
        use_amp=False,
        verbose=True,
    )

    t0 = time.time()
    history2 = trainer2.train(num_epochs=2, eval_every=1)
    stage2_time = time.time() - t0

    # Combine results
    train_loss = history2["train_loss"][-1]
    val_loss = history2["val_loss"][-1] if history2["val_loss"] else float("inf")
    val_ppl = math.exp(min(val_loss, 20))

    return {
        "label": label,
        "lr_epoch1": lr_epoch1,
        "lr_epochs23": lr_epochs23,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "params": n_params,
        "time_s": stage1_time + stage2_time,
        "stage1_val_loss": history1["val_loss"][-1] if history1["val_loss"] else None,
        "stage1_val_ppl": math.exp(min(history1["val_loss"][-1], 20)) if history1["val_loss"] else None,
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)

    ds_train = load_dataset(DATASET_NAME, split="pretrain_train")
    texts_all = [ex["text"] for ex in ds_train]
    if SUBSET_SIZE and len(texts_all) > SUBSET_SIZE:
        texts_all = random.sample(texts_all, SUBSET_SIZE)

    split = int(len(texts_all) * 0.9)
    train_texts = texts_all[:split]
    val_texts = texts_all[split:]

    results = []

    # S1: 2e-3 -> 3e-4
    results.append(train_two_stage(
        train_texts, val_texts, tokenizer, vocab_size,
        lr_epoch1=2e-3, lr_epochs23=3e-4, label="lr_decay_2e3_to_3e4"
    ))
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    # S2: 1e-3 -> 3e-4
    results.append(train_two_stage(
        train_texts, val_texts, tokenizer, vocab_size,
        lr_epoch1=1e-3, lr_epochs23=3e-4, label="lr_decay_1e3_to_3e4"
    ))
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    # Report
    print(f"\n{'='*60}")
    print("SPECIAL TEST: LR DECAY SCHEDULE RESULTS")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: x["val_loss"]):
        print(
            f"  {r['label']:20s}  lr1={r['lr_epoch1']:.0e}  lr23={r['lr_epochs23']:.0e}  "
            f"train={r['train_loss']:.4f}  val={r['val_loss']:.4f}  ppl={r['val_ppl']:.2f}  "
            f"time={r['time_s']:.0f}s"
        )
    best = min(results, key=lambda r: r["val_loss"])
    print(f"\nBest: {best['label']} (val_ppl={best['val_ppl']:.2f})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
