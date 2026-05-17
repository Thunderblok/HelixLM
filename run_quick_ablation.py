#!/usr/bin/env python3
"""Quick ablation: tied vs untied with WeightWatcher and gradient analysis."""
import os, sys, math, json, time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helix_lm.config import HelixConfig
from helix_lm.hf_model import HelixForCausalLM
from helix_lm.tokenizer import HelixTokenizer
from helix_lm.dataset import create_document_loader
from helix_lm.trainer import Trainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEQ_LEN = 128       # Fast iteration; use 512 for real runs
BATCH_SIZE = 4
EPOCHS = 2
DEVICE = 'cpu'
MAX_CHUNKS = 100    # Subset of bible for speed

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
with open('helix_lm/bible.txt', 'r') as f:
    text = f.read()
chunks = [text[i:i+500] for i in range(0, len(text), 500)]
chunks = [c for c in chunks if len(c) > 50][:MAX_CHUNKS]
split = int(len(chunks) * 0.8)
train_chunks, val_chunks = chunks[:split], chunks[split:]

# Char tokenizer
tok = HelixTokenizer('char')
tok.build_char_vocab(text)

# ---------------------------------------------------------------------------
# Helper: train variant
# ---------------------------------------------------------------------------
def train_and_analyze(name, cfg):
    print(f"\n{'='*60}")
    print(f"Variant: {name}")
    print(f"{'='*60}")

    model = HelixForCausalLM(cfg)
    params = model.count_parameters()['total']
    print(f"  Params: {params:,}")

    train_loader = create_document_loader(
        train_chunks, tok, cfg.seq_len, cfg.batch_size,
        shuffle=True, min_tail_len=1, lazy=True,
    )
    val_loader = create_document_loader(
        val_chunks, tok, cfg.seq_len, cfg.batch_size,
        shuffle=False, drop_last=False, min_tail_len=1, lazy=True,
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    trainer = Trainer(
        model=model, cfg=cfg, tokenizer=tok, verbose=False,
        train_loader=train_loader, val_loader=val_loader,
    )

    t0 = time.time()
    history = trainer.train(num_epochs=cfg.epochs, eval_every=1)
    elapsed = time.time() - t0

    final_ppl = history['perplexity'][-1] if history['perplexity'] else float('inf')
    final_loss = history['train_loss'][-1] if history['train_loss'] else float('inf')

    print(f"  Train loss: {final_loss:.4f}")
    print(f"  Val PPL: {final_ppl:.2f}")
    print(f"  Time: {elapsed:.1f}s")

    # ---- WeightWatcher ----
    print(f"\n  [WeightWatcher] {name}...")
    import weightwatcher as ww
    watcher = ww.WeightWatcher(model=model)
    try:
        details = watcher.analyze(plot=False, mp_fit=True)
        summary = watcher.get_summary(details)
        alpha = summary.get('alpha', float('nan'))
        log_norm = summary.get('log_norm', float('nan'))
        num_spikes = summary.get('num_pl_spikes', float('nan'))
        print(f"    alpha: {alpha:.3f}, log_norm: {log_norm:.3f}, PL_spikes: {num_spikes}")
    except Exception as e:
        print(f"    WW error: {e}")
        alpha = log_norm = num_spikes = float('nan')

    # ---- Gradient analysis ----
    model.train()
    model.zero_grad()
    batch = next(iter(train_loader))
    out = model(batch['input_ids'].to(DEVICE), labels=batch['labels'].to(DEVICE))
    out['loss'].backward()

    grad_norms = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[n] = p.grad.norm().item()

    # Aggregate by component type
    embed_grad = grad_norms.get('model.embed.weight', 0)
    head_grad = grad_norms.get('lm_head.weight', grad_norms.get('lm_head.buffer.weight', 0))

    print(f"\n  Gradient analysis:")
    print(f"    Embedding grad: {embed_grad:.6f}")
    print(f"    Head/buffer grad: {head_grad:.6f}")
    if embed_grad > 0:
        print(f"    Head/Embed ratio: {head_grad/embed_grad:.3f}")

    # Top 5 highest gradient layers
    sorted_grads = sorted(grad_norms.items(), key=lambda x: -x[1])
    print(f"    Top gradient layers:")
    for name, gn in sorted_grads[:5]:
        print(f"      {name:50s}: {gn:.6f}")

    return {
        'name': name, 'params': params,
        'final_loss': final_loss, 'final_ppl': final_ppl,
        'elapsed': elapsed, 'alpha': alpha,
        'log_norm': log_norm, 'num_spikes': num_spikes,
        'embed_grad': embed_grad, 'head_grad': head_grad,
        'head_embed_ratio': head_grad / embed_grad if embed_grad > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Run variants
# ---------------------------------------------------------------------------
print("="*60)
print("Quick Ablation: Tied vs Untied")
print(f"seq_len={SEQ_LEN}, epochs={EPOCHS}, chunks={len(chunks)}")
print("="*60)

base = dict(
    vocab_size=len(tok), seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
    epochs=EPOCHS, device=DEVICE, lr=5e-4, tokenizer_name='char',
    use_titans_memory=False, use_ssm=False,
)

# Use smaller d_model for speed since we're on CPU
cfg_tied = HelixConfig(d_model=128, n_columns=2, n_loops=1, n_heads=4,
                        tie_word_embeddings=True, grad_buffer_ratio=0.5, **base)
cfg_untied = HelixConfig(d_model=128, n_columns=2, n_loops=1, n_heads=4,
                          tie_word_embeddings=False, **base)

res_tied = train_and_analyze("Tied (buffer=0.5)", cfg_tied)
res_untied = train_and_analyze("Untied baseline", cfg_untied)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"{'Metric':25s} {'Tied':>12s} {'Untied':>12s} {'Delta':>12s}")
print("-"*60)
for key in ['params', 'final_loss', 'final_ppl', 'alpha', 'log_norm',
            'embed_grad', 'head_grad', 'head_embed_ratio']:
    t = res_tied.get(key, 0)
    u = res_untied.get(key, 0)
    if isinstance(t, float):
        print(f"{key:25s} {t:12.4f} {u:12.4f} {t-u:12.4f}")
    else:
        print(f"{key:25s} {t:12,} {u:12,} {t-u:12,}")

param_reduction = (1 - res_tied['params']/res_untied['params'])*100
print(f"\nParameter reduction: {param_reduction:.1f}%")

# Save
with open('ablation_quick_summary.json', 'w') as f:
    json.dump({'tied': res_tied, 'untied': res_untied}, f, indent=2, default=str)
print("Saved to ablation_quick_summary.json")
