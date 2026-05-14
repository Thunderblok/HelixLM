"""Quick GPU smoke test for the updated code."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from helix_lm import HelixConfig, HelixForCausalLM

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

# Tiny config for fast test
cfg = HelixConfig(
    vocab_size=1000,
    d_model=128,
    n_columns=2,
    nodes_per_column=(2, 2),
    n_heads=4,
    n_loops=1,
    seq_len=64,
    batch_size=2,
    use_titans_memory=False,
    attention_mode="hybrid",
    dropout=0.05,
    attn_dropout=0.15,
    lr=1e-3,
    weight_decay=0.01,
    epochs=1,
    warmup_steps=10,
    grad_clip=1.0,
    device="cpu",  # Force CPU for this quick test
    dtype="float32",
    use_cca=True,
    cca_warmup_steps=100,
    cca_ramp_mode="quadratic",
    cca_min_scale=0.05,
)

model = HelixForCausalLM(cfg)
params = model.count_parameters()["total"]
print(f"Params: {params:,}")

# Test forward with dummy data
input_ids = torch.randint(0, 1000, (2, 64))
labels = input_ids.clone()

outputs = model(input_ids, labels=labels)
print(f"Loss: {outputs['loss'].item():.4f}")

# Test backward
outputs["loss"].backward()
print("Backward OK")

# Check for NaN
assert not torch.isnan(outputs["loss"]), "Loss is NaN!"
print("\n✅ Quick test PASSED - model compiles and trains without NaN")
