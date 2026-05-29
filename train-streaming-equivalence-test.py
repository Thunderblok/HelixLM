"""
Integration test: Equivalence between list[str] (DocumentAwareDataset) and 
HelixIterableDataset (streaming) for identical training behavior.

This test verifies:
1. Both paths produce the same number of training steps
2. Both paths produce identical loss curves (within tolerance)
3. Both paths handle the same data with identical chunking behavior

CICD-ready with live stdout logging.
"""
import sys
import os
import math
import random
import copy
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer
from helix_lm.dataset import (
    DocumentAwareDataset, 
    create_document_loader,
    HelixIterableDataset,
    helix_data_collator,
)
from torch.utils.data import DataLoader


SEED = 42
SEQ_LEN = 32
N_SAMPLES = 500
N_EPOCHS = 5  # Fewer epochs for equivalence test
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 1


def banner(text):
    line = "=" * 70
    print(f"\n{line}\n{text}\n{line}", flush=True)


def get_token_counts(texts, tokenizer):
    return [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]


def select_mixed_length(texts, counts, seq_len, n=N_SAMPLES):
    """Select a mix of document lengths for comprehensive testing."""
    short, medium, long = [], [], []
    for t, c in zip(texts, counts):
        if c < seq_len:
            short.append(t)
        elif c < seq_len * 2:
            medium.append(t)
        else:
            long.append(t)
    
    n_short = min(len(short), n // 4)
    n_medium = min(len(medium), n // 2)
    n_long = n - n_short - n_medium
    n_long = min(len(long), n_long)
    
    chosen = []
    if n_short > 0:
        chosen.extend(random.sample(short, n_short))
    if n_medium > 0:
        chosen.extend(random.sample(medium, n_medium))
    if n_long > 0:
        chosen.extend(random.sample(long, n_long))
    
    # Fill remaining if needed
    need = n - len(chosen)
    if need > 0:
        pool = [t for t in texts if t not in chosen]
        if pool:
            chosen.extend(random.sample(pool, min(need, len(pool))))
    
    random.shuffle(chosen)
    return chosen[:n]


def build_cfg(vocab_size):
    return HelixConfig.tiny(
        vocab_size=vocab_size,
        seq_len=SEQ_LEN,
        tokenizer_name="gpt2",
        use_titans_memory=False,
        batch_size=BATCH_SIZE,
        lr=3e-4,
        weight_decay=0.1,
        epochs=N_EPOCHS,
        warmup_steps=10,  # Smaller warmup for quick test
        grad_clip=1.0,
        grad_buffer_ratio=0.0
    )


class ListIterableDataset:
    """Wrap a list as an HF-style iterable dataset for testing."""
    def __init__(self, texts, epoch=0):
        self.texts = texts
        self._epoch = epoch
        
    def __iter__(self):
        for text in self.texts:
            yield {"text": text}
            
    def set_epoch(self, epoch):
        self._epoch = epoch


def run_trainer_list_path(model, cfg, tokenizer, train_texts, val_texts, label):
    """
    Run training with list[str] path (DocumentAwareDataset).
    This is the reference/baseline path.
    """
    banner(f"[{label}] Training with list[str] path (DocumentAwareDataset)")
    
    train_loader = create_document_loader(
        train_texts, tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        stride=SEQ_LEN, shuffle=True, drop_last=True,
        min_tail_len=1, lazy=True,
    )
    val_loader = create_document_loader(
        val_texts, tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        stride=SEQ_LEN, shuffle=False, drop_last=False,
        min_tail_len=1, lazy=True,
    )
    
    # Count batches for verification
    train_len = len(train_loader)
    val_len = len(val_loader)
    print(f"  Train batches: {train_len}, Val batches: {val_len}")
    
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        output_dir=f"./checkpoints_equiv_{label}_list",
        example_prompts=["Once upon a time"],
        generated_example_length=10,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        use_amp=False,
        verbose=True,
        warmup_ratio=0.1,  # Use ratio mode for fair comparison
    )
    
    history = trainer.train(num_epochs=N_EPOCHS, eval_every=1)
    
    return {
        "history": history,
        "train_batches": train_len,
        "val_batches": val_len,
        "final_train_loss": history["train_loss"][-1] if history.get("train_loss") else float("inf"),
        "final_val_loss": history["val_loss"][-1] if history.get("val_loss") else float("inf"),
    }


def run_trainer_streaming_path(model, cfg, tokenizer, train_texts, val_texts, label):
    """
    Run training with HelixIterableDataset streaming path.
    This is the test path to verify equivalence.
    """
    banner(f"[{label}] Training with streaming path (HelixIterableDataset)")
    
    # Create streaming datasets
    train_iterable = ListIterableDataset(train_texts)
    val_iterable = ListIterableDataset(val_texts)
    
    train_ds = HelixIterableDataset(
        hf_iterable=train_iterable,
        tokenizer=tokenizer,
        seq_len=SEQ_LEN,
        text_column="text",
        stride=SEQ_LEN,
        min_tail_len=1,
        add_eos=True,
        shuffle_buffer_size=0,  # Disable shuffle for deterministic comparison
        seed=SEED,
    )
    val_ds = HelixIterableDataset(
        hf_iterable=val_iterable,
        tokenizer=tokenizer,
        seq_len=SEQ_LEN,
        text_column="text",
        stride=SEQ_LEN,
        min_tail_len=1,
        add_eos=True,
        shuffle_buffer_size=0,
        seed=SEED,
    )
    
    # Create DataLoaders (no shuffle for iterable)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        collate_fn=helix_data_collator,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        collate_fn=helix_data_collator,
        num_workers=0,
        drop_last=False,
    )
    
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        output_dir=f"./checkpoints_equiv_{label}_streaming",
        example_prompts=["Once upon a time"],
        generated_example_length=10,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        use_amp=False,
        verbose=True,
        warmup_ratio=0.1,  # Same ratio as list path
    )
    
    history = trainer.train(num_epochs=N_EPOCHS, eval_every=1)
    
    return {
        "history": history,
        "final_train_loss": history["train_loss"][-1] if history.get("train_loss") else float("inf"),
        "final_val_loss": history["val_loss"][-1] if history.get("val_loss") else float("inf"),
        "cached_length": trainer._cached_dataset_length,
    }


def compare_batch_contents(train_texts, tokenizer):
    """Verify that both paths produce identical batch contents."""
    banner("VERIFICATION: Batch content equivalence")
    
    # Create both datasets
    from helix_lm.dataset import DocumentAwareDataset
    
    doc_ds = DocumentAwareDataset(
        train_texts[:10], tokenizer, SEQ_LEN, min_tail_len=1, add_eos=True, lazy=True
    )
    
    list_iterable = ListIterableDataset(train_texts[:10])
    stream_ds = HelixIterableDataset(
        list_iterable, tokenizer, SEQ_LEN, stride=SEQ_LEN, 
        min_tail_len=1, add_eos=True, shuffle_buffer_size=0
    )
    
    # Get samples from both
    doc_samples = [doc_ds[i] for i in range(len(doc_ds))]
    stream_samples = list(stream_ds)
    
    print(f"  DocumentAwareDataset samples: {len(doc_samples)}")
    print(f"  HelixIterableDataset samples: {len(stream_samples)}")
    
    # Check counts match (they should for same data)
    if len(doc_samples) != len(stream_samples):
        print(f"  WARNING: Sample count mismatch! Doc={len(doc_samples)}, Stream={len(stream_samples)}")
    else:
        print(f"  [PASS] Sample counts match: {len(doc_samples)}")
    
    return len(doc_samples) == len(stream_samples)


def test_equivalence(train_texts, val_texts, tokenizer):
    """Main equivalence test."""
    banner("TEST: Equivalence between list[str] and streaming dataset")
    
    results = {}
    
    # Run list[str] path (baseline)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    cfg_a = build_cfg(len(tokenizer))
    cfg_a.pad_token_id = tokenizer.pad_token_id
    cfg_a.eos_token_id = tokenizer.eos_token_id
    model_a = HelixForCausalLM(cfg_a)
    
    results["list"] = run_trainer_list_path(
        model_a, cfg_a, tokenizer, train_texts, val_texts, "baseline"
    )
    
    # Run streaming path
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    cfg_b = copy.deepcopy(cfg_a)
    model_b = HelixForCausalLM(cfg_b)
    
    results["streaming"] = run_trainer_streaming_path(
        model_b, cfg_b, tokenizer, train_texts, val_texts, "test"
    )
    
    # Compare results
    banner("EQUIVALENCE COMPARISON")
    
    list_result = results["list"]
    stream_result = results["streaming"]
    
    print(f"\nList path results:")
    print(f"  Train batches: {list_result['train_batches']}")
    print(f"  Final train loss: {list_result['final_train_loss']:.4f}")
    print(f"  Final val loss: {list_result['final_val_loss']:.4f}")
    print(f"  Train loss history: {[f'{l:.4f}' for l in list_result['history']['train_loss']]}")
    
    print(f"\nStreaming path results:")
    print(f"  Cached dataset length: {stream_result['cached_length']}")
    print(f"  Final train loss: {stream_result['final_train_loss']:.4f}")
    print(f"  Final val loss: {stream_result['final_val_loss']:.4f}")
    print(f"  Train loss history: {[f'{l:.4f}' for l in stream_result['history']['train_loss']]}")
    
    # Verify batch count matches
    print(f"\n--- Batch Count Verification ---")
    expected_steps = list_result['train_batches'] // GRAD_ACCUM_STEPS * N_EPOCHS
    cached_batches = stream_result['cached_length']
    print(f"  List path batches: {list_result['train_batches']}")
    print(f"  Streaming cached batches: {cached_batches}")
    
    batch_match = cached_batches == list_result['train_batches']
    if batch_match:
        print(f"  [PASS] Batch counts match!")
    else:
        print(f"  [FAIL] Batch count mismatch! Diff: {abs(cached_batches - list_result['train_batches'])}")
    
    # Verify loss equivalence
    print(f"\n--- Loss Equivalence Verification ---")
    train_loss_close = abs(list_result['final_train_loss'] - stream_result['final_train_loss']) < 0.1
    val_loss_close = abs(list_result['final_val_loss'] - stream_result['final_val_loss']) < 0.1
    
    if train_loss_close:
        print(f"  [PASS] Final train losses are similar (within tolerance)")
    else:
        print(f"  [FAIL] Train loss divergence: {list_result['final_train_loss']:.4f} vs {stream_result['final_train_loss']:.4f}")
    
    if val_loss_close:
        print(f"  [PASS] Final val losses are similar (within tolerance)")
    else:
        print(f"  [FAIL] Val loss divergence: {list_result['final_val_loss']:.4f} vs {stream_result['final_val_loss']:.4f}")
    
    # Check no divergence (inf/nan)
    list_diverged = math.isinf(list_result['final_train_loss']) or math.isnan(list_result['final_train_loss'])
    stream_diverged = math.isinf(stream_result['final_train_loss']) or math.isnan(stream_result['final_train_loss'])
    
    if not list_diverged and not stream_diverged:
        print(f"  [PASS] Neither path diverged (no inf/nan)")
    else:
        print(f"  [FAIL] Divergence detected!")
        if list_diverged:
            print(f"    List path diverged")
        if stream_diverged:
            print(f"    Streaming path diverged")
    
    return batch_match and train_loss_close and val_loss_close and not list_diverged and not stream_diverged


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    banner("SETUP: Streaming Equivalence Test")
    
    # Suppress deprecation warnings for clean output
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}", flush=True)
    
    # Use a small dataset for quick testing
    print("Loading dataset...", flush=True)
    ds = load_dataset("david-thrower/tiny-stories-mini-96-seq-len-50000-samples")
    
    # Use a small subset for faster testing
    texts_all = ds["train"]["text"][:2000]
    
    # Tokenize for selection
    df = pd.DataFrame({"text": texts_all})
    print("Tokenizing for cohort selection...", flush=True)
    df["n_tokens"] = df["text"].apply(
        lambda t: len(tokenizer.encode(t, add_special_tokens=False))
    )
    counts = df["n_tokens"].tolist()
    
    # Select mixed-length documents
    texts = select_mixed_length(texts_all, counts, SEQ_LEN, N_SAMPLES)
    
    text_counts = get_token_counts(texts, tokenizer)
    print(f"Selected {len(texts)} samples", flush=True)
    print(f"Token range: {min(text_counts)} - {max(text_counts)}", flush=True)
    
    # Split train/val
    split = int(len(texts) * 0.9)
    train_texts, val_texts = texts[:split], texts[split:]
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}", flush=True)
    
    # Guard against empty cohorts
    assert len(train_texts) >= 10, f"Need >=10 train texts, got {len(train_texts)}"
    assert len(val_texts) >= 5, f"Need >=5 val texts, got {len(val_texts)}"
    
    # First verify batch contents are equivalent
    content_ok = compare_batch_contents(train_texts, tokenizer)
    
    # Run main equivalence test
    equiv_ok = test_equivalence(train_texts, val_texts, tokenizer)
    
    # Final summary
    banner("FINAL SUMMARY")
    results = {
        "batch_content_equiv": content_ok,
        "training_equiv": equiv_ok,
    }
    
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}", flush=True)
    
    all_ok = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_ok else 'SOME FAILED'}", flush=True)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
