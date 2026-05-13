"""
Medium-scale CCA training for HF Jobs.
Run: python hf_job_medium_cca.py
"""
import sys, os, math, random, json, time

# Install dependencies
os.system("pip install -e . --quiet 2>&1 | tail -5")

import torch
import torch.nn as nn
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer
from helix_lm.trainer import Trainer, get_cosine_schedule_with_warmup, compute_perplexity


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def build_cca_model(cfg):
    model = HelixForCausalLM(cfg)
    graph = model.model.recurrent.graph

    if not hasattr(graph, 'attention_gates'):
        graph.attention_gates = nn.ParameterDict()
        for name, (ci, idx, ntype) in graph.node_meta.items():
            if ntype in ("linear_attn", "full_attn"):
                graph.attention_gates[name] = nn.Parameter(torch.zeros(1))

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
                    scale = progress ** 2
                    learned = torch.sigmoid(g)
                    cg = learned * scale
                    out = cg * attn_out + (1 - cg) * x
                    return out, s
                return cca_fwd

            node.forward = make_cca(orig, gate, graph)

    return model


class CCATrainer(Trainer):
    def __init__(self, *args, cca_warmup_steps=5000, **kwargs):
        self.cca_warmup_steps = cca_warmup_steps
        self._cca_total = None
        super().__init__(*args, **kwargs)

    def train_epoch(self, epoch):
        if self._cca_total is None:
            spe = max(1, math.ceil(len(self.train_loader) / self.grad_accum_steps))
            self._cca_total = spe * self.cfg.epochs
            graph = self.model.model.recurrent.graph
            graph._cca_total_steps = self.cca_warmup_steps

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
            graph = self.model.model.recurrent.graph
            graph._cca_step = self.global_step * self.grad_accum_steps + batch_idx

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
    tok = HelixTokenizer("gpt2")
    vs = len(tok)
    print(f"Vocab={vs}")

    cfg = HelixConfig(
        vocab_size=vs, d_model=384, n_columns=2, nodes_per_column=(2, 2),
        n_heads=4, n_loops=1, seq_len=512, batch_size=16,
        use_titans_memory=False, attention_mode="hybrid", dropout=0.05,
        lr=2e-3, weight_decay=0.05, epochs=3, warmup_steps=200,
        grad_clip=1.0, device="cuda" if torch.cuda.is_available() else "cpu",
    )
    cfg.pad_token_id = tok.pad_token_id
    cfg.eos_token_id = tok.eos_token_id

    model = build_cca_model(cfg)
    params = model.count_parameters()["total"]
    print(f"Params: {params:,}")

    ds = load_dataset("david-thrower/tiny-stories-mini-96-seq-len-50000-samples")
    all_texts = list(ds["train"]["text"])
    random.shuffle(all_texts)
    train_texts = all_texts[:45000]
    val_texts = all_texts[45000:50000]

    trainer = CCATrainer(
        model=model, cfg=cfg,
        train_texts=train_texts, val_texts=val_texts,
        tokenizer=tok,
        output_dir="./checkpoints_medium_cca",
        example_prompts=["Once upon a time", "The cat sat on the"],
        generated_example_length=20,
        grad_accum_steps=1,
        use_amp=torch.cuda.is_available(),
        verbose=True,
        cca_warmup_steps=5000,
    )
    for group in trainer.optimizer.param_groups:
        group["betas"] = (0.9, 0.999)

    history = trainer.train(num_epochs=3, eval_every=1)

    final_val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
    final_ppl = math.exp(min(final_val_loss, 20))
    final_train_loss = history["train_loss"][-1]

    print(f"\n{'='*60}")
    print("MEDIUM SCALE RESULTS")
    print(f"{'='*60}")
    print(f"Train loss: {final_train_loss:.4f}")
    print(f"Val loss:   {final_val_loss:.4f}")
    print(f"Val PPL:    {final_ppl:.2f}")
    print(f"Params:     {params:,}")

    hub_id = os.environ.get("HUB_MODEL_ID", "david-thrower/HelixLM-384d-cca-43M-50Kt-medium")
    print(f"Saving to {hub_id}...")
    model.save_pretrained("./checkpoints_medium_cca/final_model")
    model.push_to_hub(hub_id)

    with open("./checkpoints_medium_cca/results.json", "w") as f:
        json.dump({"train_loss": final_train_loss, "val_loss": final_val_loss,
                   "val_ppl": final_ppl, "params": params, "history": history}, f)

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
