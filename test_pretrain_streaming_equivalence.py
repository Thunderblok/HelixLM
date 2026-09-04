#!/usr/bin/env python3
"""
Test: Verify List[str] and IterableColumn produce comparable training results
using PretrainTrainer (continuous windows, no stride).

This test validates 3 cases:
1. List[str] (materialized) -> in-memory ContinuousWindowDataset
2. Streaming IterableColumn -> auto-compiled disk store + indexed loader
3. Streaming IterableColumn with 2 epochs -> to approximate the extra
   training steps of the original stride=seq_len//2 case.

Uses dataset: david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427
- 200 train samples (pretrain_train), ~500 tokens avg (longer than seq_len)
- 30 eval samples (pretrain_val)

Model: small_v2 (~15M params) with seq_len=96
Expected:
- Case 1 and Case 2 produce SIMILAR results (not identical, because the
  indexed path uses a global permutation while the in-memory path uses a
  shuffle buffer; both are deterministic, but different orders).
- Case 3 has PPL equal or better than Case 2 (more training steps).
"""
import sys
import os
import random
import numpy as np
import torch
from datasets import load_dataset
import tempfile
import uuid

from helix_lm import HelixTokenizer, HelixConfig, HelixForCausalLM, PretrainTrainer


MAX_SEQ_LEN = 96
NUM_TRAIN = 200
NUM_VAL = 30
RANDOM_SEED = 42
TOLERANCE = 0.1  # max absolute difference in loss between List and Iterable paths


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data_as_list():
    """Load data as List[str] (materialized)."""
    ds = load_dataset("david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427")
    train_texts = list(ds['pretrain_train']['text'])[:NUM_TRAIN]
    val_texts = list(ds['pretrain_val']['text'])[:NUM_VAL]
    return train_texts, val_texts


def load_data_as_iterable_column():
    """Load data as IterableColumn (streaming)."""
    ds = load_dataset("david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427", streaming=True)
    train_texts = ds['pretrain_train'].take(NUM_TRAIN)['text']
    val_texts = ds['pretrain_val'].take(NUM_VAL)['text']
    return train_texts, val_texts


def run_training(train_texts, val_texts, num_epochs, cfg, tokenizer, output_dir,
                 auto_compile=False, store_dir=None):
    """Run training using PretrainTrainer."""
    model = HelixForCausalLM(cfg)
    trainer = PretrainTrainer(
        model=model,
        cfg=cfg,
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        output_dir=output_dir,
        pretrain_store_dir=store_dir if auto_compile and store_dir else None,
        verbose=False,
        seed=RANDOM_SEED,
        num_workers=0,          # single worker to avoid iterator duplication
        count_first=False,      # let progress bar discover count
    )
    history = trainer.train(num_epochs=num_epochs)
    
    train_loss = history['train_loss'][-1]
    val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
    val_ppl = history['perplexity'][-1] if history['perplexity'] else float('inf')
    return {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_ppl': val_ppl,
    }


def main():
    print("="*60)
    print("HelixLM PretrainTrainer Streaming Equivalence Test")
    print("="*60)
    print(f"Dataset: david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427")
    print(f"Train samples: {NUM_TRAIN}, Val samples: {NUM_VAL}")
    print(f"Seq len: {MAX_SEQ_LEN}")
    
    set_seeds(RANDOM_SEED)
    tokenizer = HelixTokenizer("gpt2")
    
    cfg = HelixConfig.small_v2(
        vocab_size=len(tokenizer),
        seq_len=MAX_SEQ_LEN,
        tokenizer_name="gpt2",
        use_titans_memory=False,
        epochs=1,               # will be overridden per case
        batch_size=8,
        lr=0.001,
        seed=RANDOM_SEED,
    )
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.bos_token_id = tokenizer.bos_token_id
    
    train_list, val_list = load_data_as_list()
    train_iter, val_iter = load_data_as_iterable_column()
    
    # Case 1: List[str], 1 epoch
    print("\n--- Case 1: List[str] (in-memory continuous) ---")
    set_seeds(RANDOM_SEED)
    metrics1 = run_training(train_list, val_list, 1, cfg, tokenizer, "./pt_c1")
    print(f"Train Loss: {metrics1['train_loss']:.4f}, Val PPL: {metrics1['val_ppl']:.2f}")
    
    # Case 2: IterableColumn, 1 epoch (auto-compile to temporary store)
    # Use a non-existent path; the trainer will create it atomically.
    print("\n--- Case 2: IterableColumn (auto-compiled store) ---")
    set_seeds(RANDOM_SEED)
    temp_store = os.path.join(tempfile.gettempdir(), f"pt_stream_store_{uuid.uuid4().hex}")
    metrics2 = run_training(train_iter, val_iter, 1, cfg, tokenizer, "./pt_c2",
                            auto_compile=True, store_dir=temp_store)
    print(f"Train Loss: {metrics2['train_loss']:.4f}, Val PPL: {metrics2['val_ppl']:.2f}")
    
    # Re-create iterators for Case 3
    train_iter, val_iter = load_data_as_iterable_column()
    
    # Case 3: IterableColumn, 2 epochs (more steps)
    print("\n--- Case 3: IterableColumn, 2 epochs ---")
    set_seeds(RANDOM_SEED)
    temp_store3 = os.path.join(tempfile.gettempdir(), f"pt_stream_store_{uuid.uuid4().hex}")
    metrics3 = run_training(train_iter, val_iter, 2, cfg, tokenizer, "./pt_c3",
                            auto_compile=True, store_dir=temp_store3)
    print(f"Train Loss: {metrics3['train_loss']:.4f}, Val PPL: {metrics3['val_ppl']:.2f}")
    
    # Verification
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    all_passed = True
    
    # Check Case 1 and Case 2 are similar (not identical due to different shuffle)
    print("\n1. Case 1 vs Case 2 (should be CLOSE, not identical):")
    loss_diff = abs(metrics1['train_loss'] - metrics2['train_loss'])
    if loss_diff < TOLERANCE:
        print(f"   ✓ Train losses close: |{metrics1['train_loss']:.4f} - {metrics2['train_loss']:.4f}| = {loss_diff:.4f} < {TOLERANCE}")
    else:
        print(f"   ✗ Train losses differ too much: {loss_diff:.4f} >= {TOLERANCE}")
        all_passed = False
    
    # Check Case 3 <= Case 2 (more training should not increase PPL)
    print("\n2. Case 3 vs Case 2 (PPL should be <=):")
    if metrics3['val_ppl'] <= metrics2['val_ppl']:
        improvement = (metrics2['val_ppl'] - metrics3['val_ppl']) / metrics2['val_ppl'] * 100
        print(f"   ✓ Val PPL improved: {metrics3['val_ppl']:.2f} <= {metrics2['val_ppl']:.2f} ({improvement:.1f}% better)")
    else:
        print(f"   ✗ Val PPL worse: {metrics3['val_ppl']:.2f} > {metrics2['val_ppl']:.2f}")
        all_passed = False
    
    print("\n" + "="*60)
    print("✓ ALL CHECKS PASSED" if all_passed else "✗ SOME CHECKS FAILED")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
