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
from helix_lm.trainer import Trainer, get_cosine_schedule_with_warmup, compute_perplexity


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def build_cca_model(cfg):
    """Build model with native CCA gates in the graph."""
    model = HelixForCausalLM(cfg)
    graph = model.model.recurrent.graph

    # Register attention gates
    if not hasattr(graph, 'attention_gates'):
        graph.attention_gates = nn.ParameterDict()
        for name, (ci, idx, ntype) in graph.node_meta.items():
            if ntype in ("linear_attn", "full_attn"):
                graph.attention_gates[name] = nn.Parameter(torch.zeros(1))

    # Patch each attention node's forward to use curriculum gate
    for name, node in graph.nodes.items():
        _, _, ntype = graph.node_meta[name]
        if ntype in ("linear_attn", "full_attn") and name in graph.attention_gates:
            gate = graph.attention_gates[name]
            orig = node.forward

            def make_cca(orig_fn, g, graph_ref):
                def cca_fwd(x, state=None, cache=None, attention_mask=None, **kw):
                    attn_out, s = orig_fn(x, state=state, cache=cache, attention_mask=attention_mask, **kw)
                    step = getattr(graph_ref, '_cca_step', 0)
                    total = getattr(graph_ref, '_cca_total_steps', 5000)
                    progress = min(1.0, step / max(1, total))
                    scale = progress ** 2  # quadratic ramp
                    learned = torch.sigmoid(g)
                    cg = learned * scale
                    out = cg * attn_out + (1 - cg) * x
                    return out, s
                return cca_fwd

            node.forward = make_cca(orig, gate, graph)

    return model


class CCATrainer(Trainer):
    """Trainer with CCA step tracking and attention_mask support."""
    def __init__(self, *args, cca_warmup_steps=5000, **kwargs):
        self.cca_warmup_steps = cca_warmup_steps
        self._cca_total = None
        super().__init__(*args, **kwargs)

    def train_epoch(self, epoch):
        # Initialize CCA total steps
        if self._cca_total is None:
            spe = max(1, math.ceil(len(self.train_loader) / self.grad_accum_steps))
            self._cca_total = spe * self.cfg.epochs
            self.model.model.recurrent.graph._cca_total_steps = self.cca_warmup_steps

        self.model.train()
        total_loss = 0.0
        raw_count = 0
        accum_count = 0
        skipped = 0
        epoch_start = time.time()
        tokens_seen = 0

        self.optimizer.zero_grad()

        if self.scheduler is None:
            spe = max(1, math.ceil(len(self.train_loader) / self.grad_accum_steps))
            tot = spe * self.cfg.epochs
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self._scheduler_warmup,
                num_training_steps=tot, num_cycles=0.5, min_lr_ratio=0.1)

        pbar = tqdm(self.train_loader, desc=f"E{epoch}", unit="b", disable=not self.verbose)
        for batch_idx, batch in enumerate(pbar):
            # Update CCA step
            self.model.model.recurrent.graph._cca_step = self.global_step * self.grad_accum_steps + batch_idx

            ids = batch["input_ids"].to(self.device)
            lbl = batch["labels"].to(self.device)
            mask = batch.get("attention_mask")
            if mask is not None:
                mask = mask.to(self.device)
            tokens_seen += ids.numel()

            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(ids, labels=lbl, attention_mask=mask)
                    loss = out["loss"]
            else:
                out = self.model(ids, labels=lbl, attention_mask=mask)
                loss = out["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                skipped += 1
                continue

            # Gradient accumulation
            divisor = 1
            if self.grad_accum_steps > 1:
                is_last = (batch_idx + 1) == len(self.train_loader)
                if is_last and accum_count < self.grad_accum_steps - 1:
                    divisor = accum_count + 1
                else:
                    divisor = self.grad_accum_steps
                loss = loss / divisor

            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1
            total_loss += loss.item() * divisor
            raw_count += 1

            is_last = (batch_idx + 1) == len(self.train_loader)
            if accum_count >= self.grad_accum_steps or is_last:
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                accum_count = 0
                self.global_step += 1

            avg = total_loss / max(raw_count, 1)
            lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{avg:.4f}", "ppl": f"{compute_perplexity(avg):.2f}", "lr": f"{lr:.2e}"})

        return {"loss": total_loss / max(raw_count, 1),
                "perplexity": compute_perplexity(total_loss / max(raw_count, 1)),
                "time": time.time() - epoch_start, "skipped_batches": skipped}


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

    trainer = CCATrainer(
        model=model, cfg=cfg,
        train_texts=train_texts, val_texts=val_texts,
        tokenizer=tok,
        output_dir=args.output_dir,
        example_prompts=["Once upon a time", "The cat sat on the"],
        generated_example_length=20,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=args.use_amp,
        verbose=True,
        cca_warmup_steps=args.cca_warmup_steps,
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
