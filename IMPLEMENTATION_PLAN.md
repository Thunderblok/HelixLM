# Implementation Plan: Streaming with Exact Steps & No Bottleneck

**Branch:** `36-streaming-with-exact-steps-preprocessed`

## Goal
Single entry point API: user passes "collection of strings" regardless of source:
- `List[str]` (in-memory)
- `datasets.Dataset['column']` (map-style Column)
- `datasets.IterableDataset['column']` (streaming IterableColumn)

Trainer handles each appropriately without user concern.

## Key Requirements

1. **Stream**: Support IterableColumn without materializing full corpus
2. **Exact steps**: Deterministic batch count for scheduler/progress bar
3. **No bottleneck**: Parallel tokenization preprocessing avoids CPU starvation

## Design

### Unified Data Loader Factory

```python
def create_helix_data_loader(
    data: Union[List[str], datasets.Dataset, datasets.IterableDataset, Any],
    tokenizer,
    seq_len: int = 2048,
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    min_tail_len: Optional[int] = None,
    add_eos: bool = True,
    stride: Optional[int] = None,
    streaming_buffer_size: int = 10_000,
    seed: int = 42,
    # Preprocessing options for iterables
    preprocess_num_proc: int = 4,
    preprocess_batch_size: int = 1000,
    max_memory_gb: float = 8.0,  # Threshold for auto-materialize
) -> DataLoader:
    """
    Single entry point for all data types.
    
    Auto-detects input type:
    - List[str] or duck-typed list-like → DocumentAwareDataset (historical)
    - datasets.Dataset (map-style) → HelixHFDataset with optional materialization
    - datasets.IterableDataset/streaming → Preprocess with map(num_proc) + count
    
    For streaming datasets, determines strategy based on size estimate:
    - Small (< max_memory_gb): Materialize + map(num_proc) for parallelism
    - Large (>= max_memory_gb): Stream with preprocess_to_shards() pipeline
    """
```

## Path 1: List[str] (Historical)

**No changes needed.**

Uses `DocumentAwareDataset` with lazy chunking. When `__len__` is called, it builds the chunk index (not ideal but users expect this behavior for in-memory data).

## Path 2: map-style datasets.Dataset (Optional Optimization)

If user passes `ds['train']['text']` from a loaded (non-streaming) dataset:

```python
if isinstance(data, datasets.Dataset):
    # Check if we can use num_proc parallelization
    texts = data[text_column]  # This IS a list-like
    return create_helix_data_loader(texts, ...)  # Delegate to Path 1
```

## Path 3: IterableDataset / IterableColumn (The New Work)

This is where the new implementation lives.

### The Strategy

**Step 1: Size Estimation**

Peek at first N samples to estimate total tokens:

```python
def _estimate_iterable_size(iterable, sample_size: int = 100):
    """Estimate total documents and tokens from stream peek."""
    samples = []
    total = 0
    for item in iterable:
        samples.append(item)
        total += len(item.get(text_column, ""))
        if len(samples) >= sample_size:
            break
    
    # Estimate: avg_chars_per_doc * estimated_docs
    # If estimated < threshold: materialize
    # If estimated > threshold: shard-based preprocessing
```

**Step 2: Two Sub-Paths**

| Size | Strategy | Parallel | Memory |
|------|----------|----------|--------|
| Small (< threshold) | Materialize + `Dataset.map(num_proc)` | ✅ Yes | O(n) |
| Large (>= threshold) | `preprocess_to_shards()` two-stage | ✅ Yes | O(shard) |

### Small Streaming Datasets (Auto-Materialize)

```python
def _handle_small_iterable(iterable, text_column, tokenizer, seq_len, ...):
    """
    For small iterables, materialize then use parallel map.
    """
    # Materialize texts
    texts = [item.get(text_column, "") for item in iterable if item.get(text_column, "")]
    
    # Convert to Dataset for parallel processing
    from datasets import Dataset
    ds = Dataset.from_dict({"text": texts})
    
    # Use Dataset.map with num_proc for tokenization
    tokenized = ds.map(
        lambda batch: {"input_ids": [tokenizer.encode(t) for t in batch["text"]]},
        batched=True,
        batch_size=1000,
        num_proc=num_proc,  # ✅ Parallel!
    )
    
    # Now use HelixPrechunkedDataset or convert to List[str]
    return create_helix_data_loader(texts, ...)  # Or optimized path
```

### Large Streaming Datasets (Shard Pipeline)

```python
def _handle_large_iterable(iterable, text_column, tokenizer, seq_len, output_dir, ...):
    """
    For large iterables, use two-stage preprocessing:
    1. Stream to shards (pre-tokenized)
    2. Load shards as fast map-style dataset
    """
    # Stage 1: Preprocess to disk shards
    shard_dir = preprocess_to_shards(
        iterable,
        tokenizer,
        output_dir=output_dir,
        seq_len=seq_len,
        num_proc=num_proc,  # ✅ Parallel!
        max_shard_size="500MB",
    )
    
    # Stage 2: Create loader from shards
    return create_helix_prechunked_loader(
        shard_dir,
        tokenizer,  # Not needed for pre-tokenized
        batch_size=batch_size,
        shuffle=shuffle,
        ...
    )
```

## The Exact Steps Problem

For **both** sub-paths, we now have exact counts:

| Path | How we get exact batch count |
|------|------------------------------|
| Small materialized | `len(dataset)` after preprocessing |
| Large sharded | Sum of samples across all shards |

```python
def _count_preprocessed(preprocessed_ds):
    """Get exact sample count from preprocessed dataset."""
    return len(preprocessed_ds)  # Now fast for both paths
```

## Progress Bar Integration

Since we have exact counts before training starts:

```python
# In Trainer._count_iterable_dataset(...)
if is_preprocessed:
    # Fast path: preprocessed dataset has __len__
    return len(train_loader.dataset)
```

## File Changes

### Modified

1. `helix_lm/dataset.py`
   - Update `create_unified_data_loader()` with automatic materialization decision
   - Add `_estimate_iterable_size()` helper
   - Add `_handle_small_iterable()` with Dataset.map(num_proc)
   - Add `_handle_large_iterable()` with shard pipeline

2. `helix_lm/trainer.py`
   - Update `_count_iterable_dataset()` to use preprocessed count
   - Remove silent counting overhead when dataset is preprocessed

### API Contract

```python
# All of these work the same way:

# 1. List[str] - historical
from helix_lm import create_helix_data_loader
train_loader = create_helix_data_loader(
    texts, tokenizer, seq_len=512, batch_size=8
)

# 2. Map-style dataset Column
from datasets import load_dataset
ds = load_dataset("my_data", split="train")
loader = create_helix_data_loader(
    ds["text"], tokenizer, seq_len=512, batch_size=8
)

# 3. Streaming dataset Column  ← NEW!
ds = load_dataset("my_data", split="train", streaming=True)
loader = create_helix_data_loader(
    ds["text"], tokenizer, seq_len=512, batch_size=8,
    preprocess_num_proc=4, max_memory_gb=8.0
)
# Automatically chooses materialize vs shard based on size
```

## Testing Plan

1. **Small iterable** (< 8GB): Verify auto-materialize path works
2. **Large iterable** (> 8GB): Verify shard pipeline works  
3. **Exact steps**: Verify progress bar shows percentage
4. **No bottleneck**: Verify GPU utilization stays high (>80%)
5. **Backward compat**: Verify List[str] still works unchanged

## Open Questions

1. **Threshold**: Is 8GB a good default? Make configurable?
2. **Temp storage**: Where do shards go? User-specified or `/tmp`?
3. **Resume**: Do we need checkpoint/resume for streaming paths?
4. **Validation**: Same preprocessing for validation data?

## Implementation Order

1. ✅ Branch created
2. ⏳ Update `create_unified_data_loader()` with detection logic
3. ⏳ Implement `_handle_small_iterable()`
4. ⏳ Implement `_handle_large_iterable()`
5. ⏳ Update `Trainer` to use preprocessed counts
6. ⏳ Tests
7. ⏳ Documentation
