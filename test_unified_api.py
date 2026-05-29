"""
Unified API Smoke Test for HelixLM Trainer with streaming and in-memory data.

Tests the 4 supported usage patterns:
1. List[str] -> Trainer (auto-detected, creates DataLoader internally)
2. IterableDataset (streaming) -> Trainer (auto-detected, streams)
3. List[str] -> create_data_loader() -> Trainer (DataLoader provided)
4. IterableDataset (streaming) -> create_data_loader() -> Trainer (DataLoader provided)

Uses real HF dataset: david-thrower/tiny-stories-mini-96-seq-len-50000-samples

Public API imports only:
- Trainer (from helix_lm.trainer)
- HelixTokenizer, create_unified_data_loader (from helix_lm)
- load_dataset (from datasets)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from helix_lm import HelixTokenizer, create_unified_data_loader
from helix_lm.trainer import Trainer
from helix_lm.config import HelixConfig
from helix_lm.hf_model import HelixForCausalLM


def get_test_data_as_list(num_samples=100):
    """Load data as List[str] (in-memory)."""
    ds = load_dataset("david-thrower/tiny-stories-mini-96-seq-len-50000-samples", split="train", streaming=False)
    texts = [ds[i]["text"] for i in range(min(num_samples, len(ds)))]
    return texts


def get_test_data_as_streaming(num_samples=100):
    """Load data as streaming IterableDataset."""
    ds = load_dataset("david-thrower/tiny-stories-mini-96-seq-len-50000-samples", split="train", streaming=True)
    # Take only first N samples for smoke test
    ds = ds.take(num_samples)
    return ds


def test_list_str_to_trainer():
    """Test 1: List[str] passed directly to Trainer."""
    print("\n=== Test 1: List[str] -> Trainer ===")
    
    tokenizer = HelixTokenizer("gpt2")
    texts = get_test_data_as_list(num_samples=50)
    
    print(f"  Loaded {len(texts)} texts as List[str]")
    
    # Trainer auto-creates DataLoader internally via create_unified_data_loader
    cfg = HelixConfig(
        vocab_size=tokenizer.vocab_size,
        seq_len=64,
        batch_size=4,
        epochs=1,
        lr=1e-4,
    )
    
    model = HelixForCausalLM(cfg)
    
    # This should auto-detect List[str] and create appropriate DataLoader
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_texts=texts,
        tokenizer=tokenizer,
        verbose=False,
    )
    
    # Verify DataLoader was created
    assert trainer.train_loader is not None
    print(f"  DataLoader created: {type(trainer.train_loader).__name__}")
    print(f"  [PASS] List[str] -> Trainer works!")
    return True


def test_streaming_to_trainer():
    """Test 2: Streaming IterableDataset passed directly to Trainer."""
    print("\n=== Test 2: IterableDataset (streaming) -> Trainer ===")
    
    tokenizer = HelixTokenizer("gpt2")
    ds = get_test_data_as_streaming(num_samples=50)
    
    print(f"  Loaded streaming dataset")
    
    cfg = HelixConfig(
        vocab_size=tokenizer.vocab_size,
        seq_len=64,
        batch_size=4,
        epochs=1,
        lr=1e-4,
    )
    
    model = HelixForCausalLM(cfg)
    
    # Trainer auto-detects streaming datasets via create_unified_data_loader
    try:
        trainer = Trainer(
            model=model,
            cfg=cfg,
            train_texts=ds,  # Pass streaming dataset
            tokenizer=tokenizer,
            verbose=False,
        )
        print(f"  DataLoader created: {type(trainer.train_loader).__name__}")
        print(f"  [PASS] IterableDataset -> Trainer works!")
        return True
    except Exception as e:
        print(f"  [FAIL] IterableDataset -> Trainer error: {e}")
        return False


def test_list_str_via_data_loader():
    """Test 3: List[str] -> create_data_loader() -> Trainer."""
    print("\n=== Test 3: List[str] -> create_data_loader() -> Trainer ===")
    
    tokenizer = HelixTokenizer("gpt2")
    texts = get_test_data_as_list(num_samples=50)
    
    loader = create_unified_data_loader(
        texts,
        tokenizer,
        seq_len=64,
        batch_size=4,
        shuffle=True,
        drop_last=True,
    )
    
    print(f"  DataLoader created from List[str]")
    print(f"  Dataset type: {type(loader.dataset).__name__}")
    
    # Pass to Trainer via train_loader argument
    cfg = HelixConfig(
        vocab_size=tokenizer.vocab_size,
        seq_len=64,
        batch_size=4,
        epochs=1,
        lr=1e-4,
    )
    
    model = HelixForCausalLM(cfg)
    
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=loader,  # Pass DataLoader directly
        tokenizer=tokenizer,
        verbose=False,
    )
    
    print(f"  [PASS] List[str] -> DataLoader -> Trainer works!")
    return True


def test_streaming_via_data_loader():
    """Test 4: IterableDataset -> create_data_loader() -> Trainer."""
    print("\n=== Test 4: IterableDataset -> create_data_loader() -> Trainer ===")
    
    tokenizer = HelixTokenizer("gpt2")
    ds = get_test_data_as_streaming(num_samples=50)
    
    loader = create_unified_data_loader(
        ds,
        tokenizer,
        seq_len=64,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        shuffle_buffer_size=100,
    )
    
    print(f"  DataLoader created from streaming dataset")
    print(f"  Dataset type: {type(loader.dataset).__name__}")
    
    cfg = HelixConfig(
        vocab_size=tokenizer.vocab_size,
        seq_len=64,
        batch_size=4,
        epochs=1,
        lr=1e-4,
    )
    
    model = HelixForCausalLM(cfg)
    
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=loader,  # Pass DataLoader directly
        tokenizer=tokenizer,
        verbose=False,
    )
    
    print(f"  [PASS] IterableDataset -> DataLoader -> Trainer works!")
    return True


def test_iterations():
    """Test that DataLoaders can actually iterate data."""
    print("\n=== Test: DataLoader iteration ===")
    
    tokenizer = HelixTokenizer("gpt2")
    
    # Test List[str] path
    texts = get_test_data_as_list(num_samples=10)
    loader = create_unified_data_loader(
        texts, tokenizer, seq_len=64, batch_size=2, shuffle=False, drop_last=True
    )
    
    batch = next(iter(loader))
    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
    print(f"  List[str] DataLoader batch shape: {batch['input_ids'].shape}")
    
    # Test streaming path
    ds = get_test_data_as_streaming(num_samples=10)
    loader = create_unified_data_loader(
        ds, tokenizer, seq_len=64, batch_size=2, shuffle=False, drop_last=True
    )
    
    batch = next(iter(loader))
    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
    print(f"  IterableDataset DataLoader batch shape: {batch['input_ids'].shape}")
    
    print(f"  [PASS] DataLoader iteration works for both paths!")
    return True


def main():
    print("=" * 60)
    print("Unified API Smoke Tests - Real HF Dataset")
    print("=" * 60)
    
    results = []
    
    # Test all 4 patterns
    results.append(("1. List[str] -> Trainer", test_list_str_to_trainer()))
    results.append(("2. IterableDataset -> Trainer", test_streaming_to_trainer()))
    results.append(("3. List[str] -> DataLoader -> Trainer", test_list_str_via_data_loader()))
    results.append(("4. IterableDataset -> DataLoader -> Trainer", test_streaming_via_data_loader()))
    results.append(("5. DataLoader iteration", test_iterations()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
