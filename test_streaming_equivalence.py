#!/usr/bin/env python3
"""
Test: Verify List[str] and IterableColumn produce identical training results.

This test validates 3 cases with texts longer than seq_len:
1. List[str] (materialized) passed to Trainer
2. Streaming IterableColumn passed to Trainer with stride=seq_len  
3. Streaming IterableColumn passed to Trainer with stride=seq_len//2

Uses dataset: david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427
- 200 train samples (pretrain_train), ~500 tokens avg (longer than seq_len)
- 30 eval samples (pretrain_val)

Model: small_v2 (~15M params) with seq_len=96
Expected:
- Case 1 and 2 produce IDENTICAL results (since we fixed shuffle)
- Case 3 has PPL equal or better (more training data with overlap)
"""
import sys
import os
import random
import numpy as np
import torch
from datasets import load_dataset

from helix_lm import HelixTokenizer, HelixConfig, HelixForCausalLM, Trainer


MAX_SEQ_LEN = 96
NUM_TRAIN = 200
NUM_VAL = 30
RANDOM_SEED = 42


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
    # Return the actual IterableColumn, not a generator
    train_texts = ds['pretrain_train'].take(NUM_TRAIN)['text']
    val_texts = ds['pretrain_val'].take(NUM_VAL)['text']
    return train_texts, val_texts


def run_training(train_texts, val_texts, stride, cfg, tokenizer, output_dir):
    """Run training using ONLY the Trainer API."""
    model = HelixForCausalLM(cfg)
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        output_dir=output_dir,
        stride=stride,
        verbose=False,
    )
    history = trainer.train(num_epochs=1)
    
    return {
        'train_loss': history['train_loss'][-1],
        'val_loss': history['val_loss'][-1] if history['val_loss'] else float('inf'),
        'val_ppl': history['perplexity'][-1] if history['perplexity'] else float('inf'),
    }


def main():
    print("="*60)
    print("HelixLM Streaming Dataset Equivalence & Stride Test")
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
        epochs=1,
        batch_size=8,
        lr=0.001,
        seed=RANDOM_SEED,
    )
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.bos_token_id = tokenizer.bos_token_id
    
    train_list, val_list = load_data_as_list()
    train_iter, val_iter = load_data_as_iterable_column()
    
    # Case 1: List[str]
    print("\n--- Case 1: List[str] ---")
    set_seeds(RANDOM_SEED)
    metrics1 = run_training(train_list, val_list, MAX_SEQ_LEN, cfg, tokenizer, "./c1")
    print(f"Train Loss: {metrics1['train_loss']:.4f}, Val PPL: {metrics1['val_ppl']:.2f}")
    
    # Case 2: IterableColumn with stride=seq_len
    print("\n--- Case 2: IterableColumn stride=seq_len ---")
    set_seeds(RANDOM_SEED)
    metrics2 = run_training(train_iter, val_iter, MAX_SEQ_LEN, cfg, tokenizer, "./c2")
    print(f"Train Loss: {metrics2['train_loss']:.4f}, Val PPL: {metrics2['val_ppl']:.2f}")
    
    # Re-create iterators for Case 3
    train_iter, val_iter = load_data_as_iterable_column()
    
    # Case 3: IterableColumn with stride=seq_len//2
    print("\n--- Case 3: IterableColumn stride=seq_len//2 ---")
    set_seeds(RANDOM_SEED)
    metrics3 = run_training(train_iter, val_iter, MAX_SEQ_LEN // 2, cfg, tokenizer, "./c3")
    print(f"Train Loss: {metrics3['train_loss']:.4f}, Val PPL: {metrics3['val_ppl']:.2f}")
    
    # Verification
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    all_passed = True
    
    # Check Case 1 == Case 2 (identical)
    print("\n1. Case 1 vs Case 2 (should be IDENTICAL):")
    if metrics1['train_loss'] == metrics2['train_loss']:
        print(f"   ✓ Train Loss identical: {metrics1['train_loss']:.4f}")
    else:
        print(f"   ✗ Train Loss differs: {metrics1['train_loss']:.4f} vs {metrics2['train_loss']:.4f}")
        all_passed = False
    
    if metrics1['val_ppl'] == metrics2['val_ppl']:
        print(f"   ✓ Val PPL identical: {metrics1['val_ppl']:.2f}")
    else:
        print(f"   ✗ Val PPL differs: {metrics1['val_ppl']:.2f} vs {metrics2['val_ppl']:.2f}")
        all_passed = False
    
    # Check Case 3 <= Case 2 (better or equal)
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
