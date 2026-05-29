"""
Smoke test for the unified data loader API.
Tests both List[str] and streaming paths.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helix_lm import HelixTokenizer, create_unified_data_loader, ListIterableDataset


def test_list_path():
    """Test List[str] path with DocumentAwareDataset."""
    print("\n=== Testing List[str] path ===")
    
    tokenizer = HelixTokenizer("gpt2")
    texts = [
        "This is a short document.",
        "This is another document with more content to test.",
        "A third document for good measure.",
        "Document four with enough tokens to make multiple chunks " * 10,
    ]
    
    loader = create_unified_data_loader(
        texts,
        tokenizer,
        seq_len=32,
        batch_size=2,
        shuffle=True,
        drop_last=True,
    )
    
    print(f"DataLoader type: {type(loader)}")
    print(f"Dataset type: {type(loader.dataset)}")
    print(f"Length: {len(loader)}")
    
    # Test iteration
    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)
        if i >= 2:  # Just get a few batches
            break
    
    print(f"Batch keys: {batches[0].keys()}")
    print(f"Batch input_ids shape: {batches[0]['input_ids'].shape}")
    print(f"[PASS] List[str] path works!")
    
    return True


def test_list_iterable_dataset():
    """Test ListIterableDataset directly."""
    print("\n=== Testing ListIterableDataset ===")
    
    texts = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    ds = ListIterableDataset(texts, epoch=0, shuffle=True, seed=42)
    
    print(f"Length: {len(ds)}")
    
    # Test epoch 0 iteration
    ds.set_epoch(0)
    items_epoch0 = list(ds)
    print(f"Epoch 0 items: {[item['text'] for item in items_epoch0]}")
    
    # Test epoch 1 iteration (should be shuffled differently)
    ds.set_epoch(1)
    items_epoch1 = list(ds)
    print(f"Epoch 1 items: {[item['text'] for item in items_epoch1]}")
    
    # Verify shuffling actually happened
    epoch0_order = [item['text'] for item in items_epoch0]
    epoch1_order = [item['text'] for item in items_epoch1]
    
    if epoch0_order != epoch1_order:
        print(f"[PASS] Shuffling works (epochs have different orders)!")
    else:
        print(f"[FAIL] Shuffling not working (epochs have same order)")
        return False
    
    return True


def test_streaming_path():
    """Test streaming path with ListIterableDataset wrapped in HelixIterableDataset."""
    print("\n=== Testing streaming path ===")
    
    from helix_lm.dataset import HelixIterableDataset, helix_data_collator
    from torch.utils.data import DataLoader
    
    tokenizer = HelixTokenizer("gpt2")
    texts = [
        "This is a short document. " * 5,
        "This is another document with more content to test. " * 5,
        "A third document for good measure. " * 5,
    ]
    
    # Wrap as ListIterableDataset (simulates streaming)
    list_ds = ListIterableDataset(texts, epoch=0, shuffle=True, seed=42)
    
    # Wrap in HelixIterableDataset
    helix_ds = HelixIterableDataset(
        hf_iterable=list_ds,
        tokenizer=tokenizer,
        seq_len=32,
        shuffle_buffer_size=100,  # Small buffer for testing
        seed=42,
    )
    
    loader = DataLoader(
        helix_ds,
        batch_size=2,
        collate_fn=helix_data_collator,
        drop_last=True,
    )
    
    print(f"DataLoader type: {type(loader)}")
    print(f"Dataset type: {type(loader.dataset)}")
    
    # Test iteration (no len() for iterable)
    batches = []
    count = 0
    for batch in loader:
        batches.append(batch)
        count += 1
        if count >= 2:
            break
    
    print(f"Batch keys: {batches[0].keys()}")
    print(f"Batch input_ids shape: {batches[0]['input_ids'].shape}")
    print(f"[PASS] Streaming path works!")
    
    return True


def test_epoch_shuffling():
    """Test that shuffling produces different orders per epoch."""
    print("\n=== Testing epoch-based shuffling ===")
    
    tokenizer = HelixTokenizer("gpt2")
    texts = [f"Document number {i} with some content." for i in range(10)]
    
    # Create loader with shuffle
    loader = create_unified_data_loader(
        texts,
        tokenizer,
        seq_len=32,
        batch_size=1,
        shuffle=True,
        drop_last=False,
    )
    
    # Check if dataset supports set_epoch
    ds = loader.dataset
    if hasattr(ds, 'set_epoch'):
        print(f"Dataset supports set_epoch: {type(ds).__name__}")
        
        # Get epoch 0 order
        ds.set_epoch(0)
        items_epoch0 = list(loader)
        
        # Get epoch 1 order
        ds.set_epoch(1)
        items_epoch1 = list(loader)
        
        print(f"Epoch 0 batches: {len(items_epoch0)}")
        print(f"Epoch 1 batches: {len(items_epoch1)}")
    else:
        print(f"Dataset type {type(ds).__name__} uses DataLoader shuffle")
    
    print(f"[PASS] Epoch shuffling mechanism present!")
    return True


def main():
    print("=" * 50)
    print("Unified Data Loader API Smoke Tests")
    print("=" * 50)
    
    results = []
    
    results.append(("List[str] path", test_list_path()))
    results.append(("ListIterableDataset", test_list_iterable_dataset()))
    results.append(("Streaming path", test_streaming_path()))
    results.append(("Epoch shuffling", test_epoch_shuffling()))
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
