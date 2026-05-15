"""
Card A: Factorial Grid — Muon vs AdamW Optimizer Ablations
============================================================

Tests 6 hyperparameter configs × 2 optimizers (Muon hybrid vs AdamW only)
at tiny scale (128 seq, 5000 samples, 3 epochs) for fast iteration.

Muon config: Muon on 2D weight matrices + AdamW on non-2D params
AdamW config: AdamW on all params (baseline)

Decision: Pick top 2 configs by PPL → lock architecture for Cards B-G.
"""
import os, sys, math, random, json, time, argparse

SEED = 42
random.seed(SEED)

import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer
from helix_lm.trainer import Trainer
from helix_lm.dataset import create_document_loader
from helix_lm.muon import Muon
from datasets import load_dataset

# ── Args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    default="david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427")
parser.add_argument("--output_dir", type=str, default="./card_a_muon_results")
parser.add_argument("--n_samples", type=int, default=5000,
                    help="Number of training samples (tiny=5000)")
parser.add_argument("--seq_len", type=int, default=128)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--device", type=str, default="auto")
args = parser.parse_args()

# ── Card A Configs (6-way factorial) ────────────────────────────────────
CARD_A_CONFIGS = [
    {"name": "A1", "d_model": 384, "n_loops": 1, "lr": 1.0e-3, "wd": 0.01, "dropout": 0.05},
    {"name": "A2", "d_model": 384, "n_loops": 1, "lr": 1.5e-3, "wd": 0.03, "dropout": 0.10},
    {"name": "A3", "d_model": 384, "n_loops": 1, "lr": 2.0e-3, "wd": 0.05, "dropout": 0.05},
    {"name": "A4", "d_model": 256, "n_loops": 2, "lr": 1.0e-3, "wd": 0.01, "dropout": 0.10},
    {"name": "A5", "d_model": 256, "n_loops": 2, "lr": 1.5e-3, "wd": 0.03, "dropout": 0.05},
    {"name": "A6", "d_model": 512, "n_loops": 1, "lr": 1.0e-3, "wd": 0.03, "dropout": 0.10},
]

OPTIMIZER_CONFIGS = [
    {"name": "muon_hybrid", "use_muon": True,  "muon_lr": 0.02, "adamw_lr_factor": 1.0},
    {"name": "adamw_only",  "use_muon": False, "muon_lr": None, "adamw_lr_factor": 1.0},
]

def build_optimizer(model, cfg, opt_cfg):
    """Build optimizer: Muon hybrid or AdamW only."""
    if opt_cfg["use_muon"]:
        # Split params: Muon for 2D weight matrices, AdamW for non-2D
        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            if param.ndim == 2 and 'embed' not in name and 'norm' not in name and param.requires_grad:
                muon_params.append(param)
            elif param.requires_grad:
                adamw_params.append(param)
        
        muon_optimizer = Muon(
            muon_params,
            lr=opt_cfg["muon_lr"],
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
        )
        adamw_optimizer = torch.optim.AdamW(
            adamw_params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )
        return [muon_optimizer, adamw_optimizer]
    else:
        # Pure AdamW baseline
        return [torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )]

def run_single_experiment(card_cfg, opt_cfg, dataset_texts, val_texts, tok, device):
    """Run a single experiment config."""
    run_name = f"{card_cfg['name']}_{opt_cfg['name']}"
    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"  Arch: d={card_cfg['d_model']}, loops={card_cfg['n_loops']}")
    print(f"  HP: lr={card_cfg['lr']}, wd={card_cfg['wd']}, do={card_cfg['dropout']}")
    print(f"  Optimizer: {opt_cfg['name']}")
    print(f"{'='*60}")
    
    vs = len(tok)
    
    # Determine model size
    if card_cfg["d_model"] <= 128:
        n_columns, nodes_per_column, n_heads = 2, (2, 2), 4
    elif card_cfg["d_model"] <= 256:
        n_columns, nodes_per_column, n_heads = 3, (2, 3, 2), 4
    elif card_cfg["d_model"] <= 384:
        n_columns, nodes_per_column, n_heads = 3, (2, 3, 2), 4
    else:
        n_columns, nodes_per_column, n_heads = 4, (3, 4, 4, 3), 8
    
    cfg = HelixConfig(
        vocab_size=vs,
        d_model=card_cfg["d_model"],
        n_columns=n_columns,
        nodes_per_column=nodes_per_column,
        n_heads=n_heads,
        n_loops=card_cfg["n_loops"],
        seq_len=args.seq_len,
        batch_size=8,
        use_titans_memory=False,
        attention_mode="hybrid",
        dropout=card_cfg["dropout"],
        attn_dropout=card_cfg["dropout"] + 0.05,
        lr=card_cfg["lr"],
        weight_decay=card_cfg["wd"],
        epochs=args.epochs,
        warmup_steps=100,
        grad_clip=1.0,
        device=device,
        dtype="float32",
        use_cca=False,  # Suppress CCA for tiny ablations
    )
    cfg.pad_token_id = tok.pad_token_id
    cfg.eos_token_id = tok.eos_token_id
    
    model = HelixForCausalLM(cfg)
    params = model.count_parameters()["total"]
    print(f"  Parameters: {params:,}")
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Build data loaders
    train_texts = dataset_texts[:args.n_samples]
    val_subset = val_texts[:min(500, len(val_texts))]
    
    train_loader = create_document_loader(
        train_texts, tok, seq_len=args.seq_len, batch_size=8,
        shuffle=True, drop_last=True, lazy=True,
    )
    val_loader = create_document_loader(
        val_subset, tok, seq_len=args.seq_len, batch_size=8,
        shuffle=False, drop_last=False, lazy=True,
    )
    
    # Build trainer with custom optimizer
    trainer = Trainer(
        model=model, cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        tokenizer=tok,
        output_dir=f"{args.output_dir}/{run_name}",
        grad_accum_steps=1,
        use_amp=False,
        verbose=True,
    )
    
    # Replace optimizer with our custom one
    optimizers = build_optimizer(model, cfg, opt_cfg)
    trainer.optimizer = optimizers[0] if len(optimizers) == 1 else {
        'muon': optimizers[0],
        'adamw': optimizers[1]
    }
    
    # Actually need to handle multiple optimizers properly
    # Trainer only supports single optimizer currently, so we monkey-patch
    if len(optimizers) > 1:
        # Custom multi-optimizer step
        class MultiOptimizer:
            def __init__(self, opts):
                self.opts = opts
                self.param_groups = []
                for o in opts:
                    self.param_groups.extend(o.param_groups)
            def step(self):
                for o in self.opts:
                    o.step()
            def zero_grad(self):
                for o in self.opts:
                    o.zero_grad()
            def state_dict(self):
                return [o.state_dict() for o in self.opts]
            def load_state_dict(self, state):
                for o, s in zip(self.opts, state):
                    o.load_state_dict(s)
        
        trainer.optimizer = MultiOptimizer(optimizers)
        
        # Rebuild scheduler with the multi-optimizer
        trainer.scheduler = None
        import math
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step):
            warmup_steps = max(1, cfg.warmup_steps)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        
        # Simple constant schedule after warmup
        trainer._scheduler_warmup = max(1, cfg.warmup_steps)
        trainer.scheduler = None  # Will be built in train_epoch
    
    # Train
    start_time = time.time()
    try:
        history = trainer.train(num_epochs=args.epochs, eval_every=1)
        
        final_val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
        final_train_loss = history["train_loss"][-1] if history.get("train_loss") else float("inf")
        final_val_ppl = math.exp(min(final_val_loss, 20))
        
        result = {
            "run_name": run_name,
            "card": card_cfg["name"],
            "optimizer": opt_cfg["name"],
            "d_model": card_cfg["d_model"],
            "n_loops": card_cfg["n_loops"],
            "lr": card_cfg["lr"],
            "wd": card_cfg["wd"],
            "dropout": card_cfg["dropout"],
            "val_ppl": final_val_ppl,
            "val_loss": final_val_loss,
            "train_loss": final_train_loss,
            "params": params,
            "time": time.time() - start_time,
            "success": True,
        }
        
        print(f"\n  Result: Val PPL = {final_val_ppl:.2f} | Train Loss = {final_train_loss:.4f}")
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        result = {
            "run_name": run_name,
            "card": card_cfg["name"],
            "optimizer": opt_cfg["name"],
            "d_model": card_cfg["d_model"],
            "n_loops": card_cfg["n_loops"],
            "lr": card_cfg["lr"],
            "wd": card_cfg["wd"],
            "dropout": card_cfg["dropout"],
            "val_ppl": float("inf"),
            "val_loss": float("inf"),
            "train_loss": float("inf"),
            "params": params,
            "time": time.time() - start_time,
            "success": False,
            "error": str(e),
        }
    
    # Cleanup
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


def main():
    print(f"{'='*70}")
    print("CARD A: MUON VS ADAMW FACTORIAL GRID")
    print(f"{'='*70}")
    print(f"Configs: {len(CARD_A_CONFIGS)} architecture configs")
    print(f"Optimizers: {len(OPTIMIZER_CONFIGS)} optimizer variants")
    print(f"Total experiments: {len(CARD_A_CONFIGS) * len(OPTIMIZER_CONFIGS)}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.n_samples} | Seq: {args.seq_len} | Epochs: {args.epochs}")
    print(f"{'='*70}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)
    all_train = list(ds["pretrain_train"]["text"])
    all_val = list(ds["pretrain_val"]["text"])
    print(f"Train docs: {len(all_train):,} | Val docs: {len(all_val):,}")
    
    # Tokenizer
    tok = HelixTokenizer("gpt2")
    print(f"Vocab: {len(tok)}")
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}\n")
    
    # Run all experiments
    all_results = []
    total = len(CARD_A_CONFIGS) * len(OPTIMIZER_CONFIGS)
    idx = 0
    
    for card_cfg in CARD_A_CONFIGS:
        for opt_cfg in OPTIMIZER_CONFIGS:
            idx += 1
            print(f"\n[{idx}/{total}] Starting experiment...")
            result = run_single_experiment(card_cfg, opt_cfg, all_train, all_val, tok, device)
            all_results.append(result)
            
            # Save intermediate results
            with open(f"{args.output_dir}/results_partial.json", "w") as f:
                json.dump(all_results, f, indent=2)
    
    # ── Analysis ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("CARD A RESULTS SUMMARY")
    print(f"{'='*70}")
    
    # Sort by val PPL
    results_sorted = sorted(all_results, key=lambda x: x["val_ppl"])
    
    print(f"\n{'Rank':<6}{'Config':<20}{'Val PPL':<12}{'Train Loss':<14}{'Time':<10}{'Status'}")
    print("-" * 70)
    for i, r in enumerate(results_sorted):
        status = "OK" if r["success"] else "FAIL"
        print(f"{i+1:<6}{r['run_name']:<20}{r['val_ppl']:<12.2f}{r['train_loss']:<14.4f}{r['time']:<10.1f}s{status}")
    
    # Compare Muon vs AdamW
    print(f"\n{'='*70}")
    print("MUON vs ADAMW HEAD-TO-HEAD")
    print(f"{'='*70}")
    
    for card_cfg in CARD_A_CONFIGS:
        card_name = card_cfg["name"]
        muon_result = next((r for r in all_results if r["card"] == card_name and r["optimizer"] == "muon_hybrid"), None)
        adam_result = next((r for r in all_results if r["card"] == card_name and r["optimizer"] == "adamw_only"), None)
        
        if muon_result and adam_result:
            muon_ppl = muon_result["val_ppl"]
            adam_ppl = adam_result["val_ppl"]
            diff = adam_ppl - muon_ppl
            winner = "MUON" if muon_ppl < adam_ppl else "ADAMW"
            print(f"  {card_name}: Muon={muon_ppl:.2f} vs AdamW={adam_ppl:.2f} | "
                  f"delta={diff:+.2f} → {winner}")
    
    # Pick top 2 configs
    top2 = results_sorted[:2]
    print(f"\n{'='*70}")
    print("TOP 2 CONFIGS (lock architecture for Cards B-G)")
    print(f"{'='*70}")
    for i, r in enumerate(top2):
        print(f"  #{i+1}: {r['run_name']} (Val PPL={r['val_ppl']:.2f})")
        print(f"       d={r['d_model']}, loops={r['n_loops']}, "
              f"lr={r['lr']}, wd={r['wd']}, do={r['dropout']}")
        print(f"       optimizer={r['optimizer']}")
    
    # Save final results
    with open(f"{args.output_dir}/results_card_a.json", "w") as f:
        json.dump({
            "results": all_results,
            "top2": top2,
            "summary": {
                "n_experiments": len(all_results),
                "n_success": sum(1 for r in all_results if r["success"]),
                "best_config": top2[0] if top2 else None,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/results_card_a.json")
    return top2


if __name__ == "__main__":
    main()
