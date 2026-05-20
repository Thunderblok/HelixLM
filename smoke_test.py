"""Quick smoke test for HelixLM ablation script."""
import math, sys, os, random

sys.path.insert(0, os.path.dirname(__file__))

import torch
from datasets import load_dataset

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer

random.seed(42)
torch.manual_seed(42)

tokenizer = HelixTokenizer("gpt2")
vocab_size = len(tokenizer)

ds = load_dataset("david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427", split="pretrain_train")
texts = [ex["text"] for ex in ds]
texts = random.sample(texts, 100)
train_texts = texts[:90]
val_texts = texts[90:]

cfg = HelixConfig.small_v2(
    vocab_size=vocab_size,
    tokenizer_name="gpt2",
    use_titans_memory=False,
    use_cca=False,
    use_ssm=False,
    lr=3e-4,
    n_loops=1,
    dropout=0.05,
    weight_decay=0.1,
    epochs=1,
    warmup_steps=10,
    grad_clip=1.0,
    grad_buffer_ratio=1.0 / math.e,
    batch_size=8,
)
cfg.pad_token_id = tokenizer.pad_token_id
cfg.eos_token_id = tokenizer.eos_token_id
cfg.bos_token_id = tokenizer.bos_token_id

model = HelixForCausalLM(cfg)
print(f"Model params: {model.count_parameters()['total']:,}")

trainer = Trainer(
    model=model, cfg=cfg,
    train_texts=train_texts, val_texts=val_texts,
    tokenizer=tokenizer,
    output_dir="./ckpt_smoke",
    example_prompts=["Once upon a time"],
    generated_example_length=15,
    grad_accum_steps=1,
    use_amp=False,
    verbose=True,
)

history = trainer.train(num_epochs=1, eval_every=1)
print(f"\nSmoke test passed! Train loss: {history['train_loss'][-1]:.4f}, Val loss: {history['val_loss'][-1]:.4f}")
