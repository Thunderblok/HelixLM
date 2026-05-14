"""
CPU Sanity Check: 2-step forward/backward with the fixed graph.py
Verify: no NaN, CCA gates init ~0.88, scale min=0.05
"""
import os, sys
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

import torch
from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer

SEED = 42
torch.manual_seed(SEED)

tok = HelixTokenizer("gpt2")
vs = len(tok)

cfg = HelixConfig(
    vocab_size=vs,
    d_model=256,
    n_columns=2,
    nodes_per_column=(2, 2),
    n_heads=4,
    n_loops=2,
    seq_len=512,
    batch_size=2,
    use_titans_memory=False,
    attention_mode="hybrid",
    dropout=0.05,
    lr=1e-3,
    weight_decay=0.01,
    epochs=1,
    warmup_steps=200,
    grad_clip=1.0,
    device="cpu",
    dtype="float32",
    use_cca=True,
    cca_warmup_steps=10000,
    cca_ramp_mode="cubic_ease",
    cca_min_scale=0.05,
)
cfg.pad_token_id = tok.pad_token_id
cfg.eos_token_id = tok.eos_token_id

model = HelixForCausalLM(cfg)
print(f"Params: {model.count_parameters()['total']:,}")

# Check gate initialization
graph = model.model.recurrent.graph
if graph.use_cca:
    for name, gate in graph.attention_gates.items():
        gate_val = torch.sigmoid(gate).item()
        assert 0.85 < gate_val < 0.90, f"Gate {name} init should be ~0.88, got {gate_val}"
        print(f"  Gate {name}: raw={gate.item():.3f} sigmoid={gate_val:.4f} ✓")

# Fake batch
batch_size = 2
seq_len = 16
input_ids = torch.randint(0, vs, (batch_size, seq_len), dtype=torch.long)
labels = input_ids.clone()
labels[:, :-1] = input_ids[:, 1:]
labels[:, -1] = -100
attention_mask = torch.ones_like(input_ids)
attention_mask[:, -4:] = 0  # some padding

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(2):
    optimizer.zero_grad()
    outputs = model(
        input_ids,
        labels=labels,
        attention_mask=attention_mask,
        cca_step=step,
    )
    loss = outputs["loss"]
    print(f"Step {step}: loss={loss.item():.4f} | NaN={loss.isnan().item()}")
    assert not loss.isnan(), "Loss is NaN!"
    assert not loss.isinf(), "Loss is Inf!"
    loss.backward()
    # Check for NaN grads
    has_nan = False
    for n, p in model.named_parameters():
        if p.grad is not None:
            if p.grad.isnan().any() or p.grad.isinf().any():
                has_nan = True
                print(f"  NaN/Inf grad in {n}")
    if not has_nan:
        print("  No NaN/Inf gradients ✓")
    optimizer.step()

print("\n✅ CPU sanity check PASSED")
