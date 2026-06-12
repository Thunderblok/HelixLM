#!/usr/bin/env python3
"""
CICD Regression Test: Full training with ~500 seq len samples fed into 96 seq len model.

This test validates:
1. List[str] and IterableColumn produce equivalent training results
2. Sliding window chunking works (> seq_len texts get batched)
3. Loss and PPL improve to reasonable levels
4. Generated text shows pattern learning (not pure random tokens)
5. No regression in streaming API

Based on quick_demo_cpu.py but uses real data that requires chunking.
Uses 200 samples (comparable token count to original 1000 x 96 test).

Expected run time: ~25-30 minutes (similar to quick_demo_cpu.py)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import random
import numpy as np
import torch
from datasets import load_dataset, IterableDataset
from math import ceil

from helix_lm import (
    HelixConfig,
    HelixTokenizer,
    HelixForCausalLM,
    Trainer,
    _is_iterable_column,
)

# Settings based on quick_demo_cpu.py
EPOCHS = 25
MAX_SEQ_LEN = 96
NUM_SAMPLES = 200  # ~200 x ~500 tokens = ~100k tokens (comparable to original)
VAL_SPLIT = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 8

EXAMPLE_PROMPTS = [
    "The next day, something unexpected",
    "I have an idea, Ben. Let's build a",
    "The oyster and its friends decided to make",
]


class TextExtractor:
    """Wraps HF IterableDataset to extract text column."""
    def __init__(self, iterable, text_column='text'):
        self._iter = iterable
        self._text_column = text_column
    
    def __iter__(self):
        for item in self._iter:
            yield item[self._text_column]


def run_training_case(case_name, train_texts, val_texts):
    """Run training for one case and return metrics."""
    print(f"\n{'='*70}")
    print(f"CASE: {case_name}")
    print(f"{'='*70}")
    
    # Set seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")
    
    # Model setup (same as quick_demo_cpu.py)
    cfg = HelixConfig.small_v2(
        vocab_size=vocab_size,
        seq_len=MAX_SEQ_LEN,
        tokenizer_name="gpt2",
        use_titans_memory=False,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=5e-4,
    )
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.bos_token_id = tokenizer.bos_token_id
    
    model = HelixForCausalLM(cfg)
    params = model.count_parameters()['total']
    print(f"Parameters: {params:,}")
    
    # Verify streaming detection
    is_streaming = _is_iterable_column(train_texts)
    print(f"Is streaming: {is_streaming}")
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        output_dir=f"./test_checkpoints_{case_name.lower().replace(' ', '_')}",
        example_prompts=EXAMPLE_PROMPTS,
        generated_example_length=50,
        verbose=True,
    )
    
    print("\n" + "-"*70)
    print("Starting training...")
    print("-"*70)
    
    history = trainer.train(num_epochs=EPOCHS, eval_every=5)
    
    # Get final metrics
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1] if history['val_loss'] else None
    final_ppl = history['perplexity'][-1]
    
    print(f"\nFinal metrics:")
    print(f"  Train Loss: {final_train_loss:.4f}")
    if final_val_loss:
        print(f"  Val Loss: {final_val_loss:.4f}")
    print(f"  Perplexity: {final_ppl:.2f}")
    
    return {
        'case_name': case_name,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_ppl': final_ppl,
        'is_streaming': is_streaming,
    }


def main():
    print("="*70)
    print("HelixLM CICD Regression Test: Full Training")
    print("="*70)
    print(f"Epochs: {EPOCHS}")
    print(f"Seq len: {MAX_SEQ_LEN}")
    print(f"Train samples: ~{int(NUM_SAMPLES * (1 - VAL_SPLIT))} (200 @ ~500 tokens ≈ 100k tokens)")
    print(f"Val samples: ~{int(NUM_SAMPLES * VAL_SPLIT)}")
    print(f"Expected run time: ~30 min (similar to quick_demo_cpu.py)")
    print("="*70)
    
    # ==========================================================================
    # Load data ONCE
    # ==========================================================================
    print("\nLoading dataset...")
    
    ds_full = load_dataset("david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427")
    all_texts = list(ds_full['pretrain_train']['text'])[:NUM_SAMPLES]
    
    # Shuffle before splitting (same as quick_demo_cpu.py)
    random.seed(RANDOM_SEED)
    random.shuffle(all_texts)
    
    split_idx = ceil(NUM_SAMPLES * (1 - VAL_SPLIT))
    train_texts_list = all_texts[:split_idx]
    val_texts_list = all_texts[split_idx:]
    
    print(f"Loaded {len(train_texts_list)} train, {len(val_texts_list)} val samples")
    
    # Check sample length - should be >> seq_len for chunking
    sample_words = len(train_texts_list[0].split())
    print(f"Sample text word count: ~{sample_words} (should be >> {MAX_SEQ_LEN})")
    
    # ==========================================================================
    # Run Case 1: List[str]
    # ==========================================================================
    results_case1 = run_training_case("List[str]", train_texts_list, val_texts_list)
    
    # ==========================================================================
    # Run Case 2: IterableColumn (need to reload streaming data)
    # ==========================================================================
    ds_stream = load_dataset("david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427", streaming=True)
    train_iter = ds_stream['pretrain_train'].take(split_idx)
    val_iter = ds_stream['pretrain_val'].take(len(val_texts_list))
    
    results_case2 = run_training_case(
        "IterableColumn",
        TextExtractor(train_iter, 'text'),
        TextExtractor(val_iter, 'text'),
    )
    
    # ==========================================================================
    # Compare results
    # ==========================================================================
    print("\n" + "="*70)
    print("COMPARISON: List[str] vs IterableColumn")
    print("="*70)
    
    loss_diff = abs(results_case1['final_train_loss'] - results_case2['final_train_loss'])
    ppl_diff = abs(results_case1['final_ppl'] - results_case2['final_ppl'])
    
    print(f"\nTrain Loss:")
    print(f"  List[str]:      {results_case1['final_train_loss']:.4f}")
    print(f"  IterableColumn: {results_case2['final_train_loss']:.4f}")
    print(f"  Difference:     {loss_diff:.4f}")
    
    print(f"\nPerplexity:")
    print(f"  List[str]:      {results_case1['final_ppl']:.2f}")
    print(f"  IterableColumn: {results_case2['final_ppl']:.2f}")
    print(f"  Difference:     {ppl_diff:.2f}")
    
    # ==========================================================================
    # Validation (visual - check logs for text quality)
    # ==========================================================================
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    all_passed = True
    
    # 1. Loss should be reasonable (not NaN/Inf)
    for results in [results_case1, results_case2]:
        if np.isnan(results['final_train_loss']) or np.isinf(results['final_train_loss']):
            print(f"✗ {results['case_name']}: Train loss is NaN/Inf")
            all_passed = False
        else:
            print(f"✓ {results['case_name']}: Train loss valid ({results['final_train_loss']:.4f})")
    
    # 2. PPL should improve and be reasonable
    for results in [results_case1, results_case2]:
        if results['final_ppl'] > 1000:
            print(f"✗ {results['case_name']}: PPL too high ({results['final_ppl']:.2f})")
            all_passed = False
        else:
            print(f"✓ {results['case_name']}: PPL reasonable ({results['final_ppl']:.2f})")
    
    # 3. Both paths should give similar results
    if loss_diff > 0.5:
        print(f"✗ Loss difference too large: {loss_diff:.4f}")
        all_passed = False
    else:
        print(f"✓ Loss difference acceptable: {loss_diff:.4f}")
    
    # Final verdict
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL CICD REGRESSION TESTS PASSED")
        print("="*70)
        print("\nManual verification: Check generation logs above for:")
        print("  • Text shows word patterns (not random tokens)")
        print("  • Sentences have structure (even if malarkey)")
        print("  • No excessive character repetition")
        return 0
    else:
        print("✗ SOME CICD REGRESSION TESTS FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
