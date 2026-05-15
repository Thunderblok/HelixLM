"""
Muon vs AdamW Focused Comparison
=================================

Direct comparison of Muon (hybrid) vs AdamW (only) on 3 configs at tiny scale.
Uses local bible.txt + char tokenizer for speed (no HF downloads).

Uses Trainer's native multi-optimizer support (no monkey-patching).
"""
import os
import sys
import math
import random
import json
import time
import warnings

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
import torch

torch.manual_seed(SEED)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer
from helix_lm.trainer import Trainer
from helix_lm.dataset import create_document_loader
from helix_lm.muon import Muon

# -- Load local data ---------------------------------------------------------
with open(os.path.join(REPO_DIR, "bible.txt"), "r") as f:
    text = f.read()

texts = [t.strip() for t in text.split("\n\n") if len(t.strip()) > 50][:200]
train_texts = texts[:160]
val_texts = texts[160:]

tok = HelixTokenizer("char")
tok.build_char_vocab(text)
vs = len(tok)
pad_id = tok.pad_token_id

ARCH_CONFIGS = [
    {"name": "A4", "d_model": 256, "n_loops": 2, "lr": 1e-3, "wd": 0.01, "dropout": 0.10},
    {"name": "A5", "d_model": 256, "n_loops": 2, "lr": 1.5e-3, "wd": 0.03, "dropout": 0.05},
    {"name": "A3", "d_model": 384, "n_loops": 1, "lr": 2e-3, "wd": 0.05, "dropout": 0.05},
]

SEQ_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3
N_TRAIN = 1000
N_VAL = 100


def build_hybrid_optimizer(model, cfg, adamw_lr=None, adamw_wd=None):
    """Build Muon + AdamW hybrid optimizer pair.

    Muon handles 2D weight matrices; AdamW handles embeddings, norms, biases.
    Returns a list [Muon, AdamW] compatible with Trainer multi-optimizer.
    """
    adamw_lr = adamw_lr or cfg.lr
    adamw_wd = adamw_wd or cfg.weight_decay

    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if (
            param.ndim == 2
            and "embed" not in name
            and "norm" not in name
            and param.requires_grad
        ):
            muon_params.append(param)
        elif param.requires_grad:
            adamw_params.append(param)

    muon_p_n = sum(p.numel() for p in muon_params)
    adam_p_n = sum(p.numel() for p in adamw_params)
    print(f"    Muon params: {muon_p_n:,} | AdamW params: {adam_p_n:,}")

    muon_opt = Muon(
        muon_params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5
    )
    adamw_opt = torch.optim.AdamW(
        adamw_params,
        lr=adamw_lr,
        weight_decay=adamw_wd,
        betas=(0.9, 0.999),
    )

    return [muon_opt, adamw_opt]


def run_experiment(arch_cfg, use_muon):
    """Run single experiment."""
    opt_name = "muon" if use_muon else "adamw"
    run_name = f"{arch_cfg['name']}_{opt_name}"

    print(f"\n{'='*55}")
    print(f"RUN: {run_name}")
    print(f"  d={arch_cfg['d_model']}, loops={arch_cfg['n_loops']}")
    print(f"  lr={arch_cfg['lr']}, wd={arch_cfg['wd']}, do={arch_cfg['dropout']}")
    print(f"  Optimizer: {'Muon hybrid' if use_muon else 'AdamW only'}")
    print(f"{'='*55}")

    t0 = time.time()

    d = arch_cfg["d_model"]
    if d <= 128:
        n_columns, nodes_per_column, n_heads = 2, (2, 2), 4
    elif d <= 256:
        n_columns, nodes_per_column, n_heads = 3, (2, 3, 2), 4
    elif d <= 384:
        n_columns, nodes_per_column, n_heads = 3, (2, 3, 2), 4
    else:
        n_columns, nodes_per_column, n_heads = 4, (3, 4, 4, 3), 8

    cfg = HelixConfig(
        vocab_size=vs,
        d_model=d,
        n_columns=n_columns,
        nodes_per_column=nodes_per_column,
        n_heads=n_heads,
        n_loops=arch_cfg["n_loops"],
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        use_titans_memory=False,
        attention_mode="hybrid",
        dropout=arch_cfg["dropout"],
        attn_dropout=min(arch_cfg["dropout"] + 0.05, 0.25),
        lr=arch_cfg["lr"],
        weight_decay=arch_cfg["wd"],
        epochs=EPOCHS,
        warmup_steps=50,
        grad_clip=1.0,
        device="cpu",
        dtype="float32",
        use_cca=False,
    )
    cfg.pad_token_id = pad_id
    cfg.eos_token_id = tok.eos_token_id

    model = HelixForCausalLM(cfg)
    params = model.count_parameters()["total"]
    print(f"  Parameters: {params:,}")

    # Data loaders
    train_loader = create_document_loader(
        train_texts[:N_TRAIN],
        tok,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        lazy=True,
    )
    val_loader = create_document_loader(
        val_texts[:N_VAL],
        tok,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        lazy=True,
    )

    # Build optimizer: hybrid Muon+AdamW or pure AdamW
    if use_muon:
        optimizer = build_hybrid_optimizer(model, cfg)
    else:
        optimizer = None  # Trainer will use default AdamW

    # Build trainer with custom optimizer if Muon
    out_dir = f"./results_muon_adamw/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    trainer_kwargs = dict(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tok,
        output_dir=out_dir,
        grad_accum_steps=1,
        use_amp=False,
        verbose=False,
    )
    if use_muon:
        trainer_kwargs["optimizer"] = optimizer

    trainer = Trainer(**trainer_kwargs)

    # For pure AdamW, ensure betas
    if not use_muon:
        for group in trainer.optimizer.param_groups:
            group["betas"] = (0.9, 0.999)

    # Train
    try:
        history = trainer.train(num_epochs=EPOCHS, eval_every=1)

        val_loss = (
            history["val_loss"][-1] if history.get("val_loss") else float("inf")
        )
        train_loss = (
            history["train_loss"][-1]
            if history.get("train_loss")
            else float("inf")
        )
        val_ppl = math.exp(min(val_loss, 20))

        result = {
            "run": run_name,
            "arch": arch_cfg["name"],
            "optimizer": opt_name,
            "d_model": d,
            "n_loops": arch_cfg["n_loops"],
            "lr": arch_cfg["lr"],
            "wd": arch_cfg["wd"],
            "dropout": arch_cfg["dropout"],
            "val_ppl": val_ppl,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "params": params,
            "time": time.time() - t0,
            "success": True,
        }
        print(
            f"  OK Val PPL: {val_ppl:.2f} | Train: {train_loss:.4f} | "
            f"{result['time']:.1f}s"
        )

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        result = {
            "run": run_name,
            "arch": arch_cfg["name"],
            "optimizer": opt_name,
            "d_model": d,
            "n_loops": arch_cfg["n_loops"],
            "lr": arch_cfg["lr"],
            "wd": arch_cfg["wd"],
            "dropout": arch_cfg["dropout"],
            "val_ppl": float("inf"),
            "val_loss": float("inf"),
            "train_loss": float("inf"),
            "params": params,
            "time": time.time() - t0,
            "success": False,
            "error": str(e),
        }

    del model, trainer
    return result


def main():
    print(f"{'='*60}")
    print("MUON vs ADAMW: FOCUSED ABLATION")
    print(f"{'='*60}")
    print(f"3 configs x 2 optimizers = 6 runs")
    print(
        f"Scale: {N_TRAIN} train, {N_VAL} val, {SEQ_LEN} seq, "
        f"{EPOCHS} epochs, CPU"
    )
    print(f"{'='*60}")

    os.makedirs("./results_muon_adamw", exist_ok=True)

    results = []
    idx = 0
    for arch_cfg in ARCH_CONFIGS:
        for use_muon in [False, True]:
            idx += 1
            print(f"\n[{idx}/6] Experiment {idx} of 6")
            r = run_experiment(arch_cfg, use_muon)
            results.append(r)
            with open("./results_muon_adamw/results_partial.json", "w") as f:
                json.dump(results, f, indent=2)

    # Summary
    print(f"\n\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    sorted_r = sorted(results, key=lambda x: x["val_ppl"])
    print(f"\n{'Rank':<6}{'Config':<22}{'Val PPL':<12}{'Train Loss':<14}{'Time':<10}")
    print("-" * 60)
    for i, r in enumerate(sorted_r):
        print(
            f"{i+1:<6}{r['run']:<22}{r['val_ppl']:<12.2f}"
            f"{r['train_loss']:<14.4f}{r['time']:<10.1f}s"
        )

    # Head-to-head
    print(f"\n{'='*60}")
    print("MUON vs ADAMW HEAD-TO-HEAD")
    print(f"{'='*60}")
    for arch in [c["name"] for c in ARCH_CONFIGS]:
        muon_r = next(
            (
                r
                for r in results
                if r["arch"] == arch and r["optimizer"] == "muon"
            ),
            None,
        )
        adam_r = next(
            (
                r
                for r in results
                if r["arch"] == arch and r["optimizer"] == "adamw"
            ),
            None,
        )
        if muon_r and adam_r:
            mp, ap = muon_r["val_ppl"], adam_r["val_ppl"]
            diff = ap - mp
            winner = "MUON" if mp < ap else "ADAMW"
            print(
                f"  {arch}: Muon={mp:.2f} vs AdamW={ap:.2f} "
                f"| delta={diff:+.2f} -> {winner}"
            )

    with open("./results_muon_adamw/results_final.json", "w") as f:
        json.dump(
            {
                "results": results,
                "sorted": sorted_r,
                "summary": {
                    "best": sorted_r[0] if sorted_r else None,
                    "n_success": sum(1 for r in results if r["success"]),
                },
            },
            f,
            indent=2,
        )

    print("\nResults saved to ./results_muon_adamw/results_final.json")


if __name__ == "__main__":
    main()
