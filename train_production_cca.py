"""
Production-scale CCA training for HelixLM.
Self-contained: no external dependencies besides standard packages.
Architecture: d=384, n_loops=1, 2 columns of 2 nodes, hybrid attention.
With: attention_mask fix + Curriculum Component Activation.
"""
import sys, os, math, random, json, time, argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm

# Import from local repo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer
from helix_lm.trainer import Trainer


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def build_cca_model(cfg):
    """Build model with native CCA gates in the graph (no manual patching needed)."""
    model = HelixForCausalLM(cfg)
    return model





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_loops", type=int, default=1)
    parser.add_argument("--n_columns", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_production_cca")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cca_warmup_steps", type=int, default=5000)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--use_amp", action="store_true")
    args = parser.parse_args()

    tok = HelixTokenizer("gpt2")
    vs = len(tok)
    print(f"Vocab={vs}")

    cfg = HelixConfig(
        vocab_size=vs, d_model=args.d_model, n_columns=args.n_columns,
        nodes_per_column=(2, 2),
        n_heads=4, n_loops=args.n_loops, seq_len=args.seq_len,
        batch_size=args.batch_size,
        use_titans_memory=False, attention_mode="hybrid", dropout=args.dropout,
        lr=args.lr, weight_decay=args.wd, epochs=args.epochs,
        warmup_steps=200, grad_clip=1.0,
        device=args.device,
    )
    cfg.pad_token_id = tok.pad_token_id
    cfg.eos_token_id = tok.eos_token_id

    model = build_cca_model(cfg)
    params = model.count_parameters()["total"]
    print(f"Params: {params:,}")

    # Load data
    ds = load_dataset("david-thrower/tiny-stories-mini-96-seq-len-50000-samples")
    all_texts = list(ds["train"]["text"])
    random.shuffle(all_texts)
    n_train = int(args.samples * 0.9)
    train_texts = all_texts[:n_train]
    val_texts = all_texts[n_train:args.samples]

    # CRITICAL: For short-text datasets at long seq_len, set min_tail_len=1
    # to prevent dropping nearly all documents. The dataset has ~96-token stories.
    from helix_lm.dataset import create_document_loader
    train_loader = create_document_loader(
        train_texts, tok, seq_len=args.seq_len, batch_size=args.batch_size,
        shuffle=True, drop_last=True, min_tail_len=1, lazy=True,
    )
    val_loader = create_document_loader(
        val_texts, tok, seq_len=args.seq_len, batch_size=args.batch_size,
        shuffle=False, drop_last=False, min_tail_len=1, lazy=True,
    )

    trainer = Trainer(
        model=model, cfg=cfg,
        train_loader=train_loader, val_loader=val_loader,
        tokenizer=tok,
        output_dir=args.output_dir,
        example_prompts=["Once upon a time", "The cat sat on the"],
        generated_example_length=20,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=args.use_amp,
        verbose=True,
    )
    for group in trainer.optimizer.param_groups:
        group["betas"] = (0.9, 0.999)

    history = trainer.train(num_epochs=args.epochs, eval_every=1)

    final_val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
    final_ppl = math.exp(min(final_val_loss, 20))
    final_train_loss = history["train_loss"][-1]

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Train loss: {final_train_loss:.4f}")
    print(f"Val loss:   {final_val_loss:.4f}")
    print(f"Val PPL:    {final_ppl:.2f}")
    print(f"Params:     {params:,}")

    # Save
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"train_loss": final_train_loss, "val_loss": final_val_loss,
                   "val_ppl": final_ppl, "params": params, "history": history,
                   "config": vars(args)}, f)

    if args.push_to_hub and args.hub_model_id:
        print(f"Pushing to {args.hub_model_id}...")
        model.push_to_hub(args.hub_model_id)

    print(f"\n{'='*60}")
    if final_ppl < 80:
        print(f"SHIP IT! PPL={final_ppl:.2f} < 80")
    elif final_ppl < 120:
        print(f"PRODUCTION GATE OPEN: PPL={final_ppl:.2f} < 120")
    elif final_ppl < 160:
        print(f"PROMISING: PPL={final_ppl:.2f}")
    else:
        print(f"NEEDS WORK: PPL={final_ppl:.2f}")


if __name__ == "__main__":
    main()
