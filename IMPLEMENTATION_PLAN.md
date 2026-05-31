# Implementation Plan: Streaming with Exact Steps & No Bottleneck

**Branch:** `36-streaming-with-exact-steps-preprocessed`

## Goal
Single entry point API: user passes "collection of strings" regardless of source:
- `List[str]` (in-memory)
- `datasets.Dataset['column']` (map-style Column)
- `datasets.IterableDataset['column']` (streaming IterableColumn)

Trainer handles each appropriately without user concern.

## Key Requirements

1. **Stream**: Support IterableColumn without materializing full corpus at once
2. **Exact steps**: Deterministic batch count for scheduler/progress bar  
3. **No bottleneck**: Parallel tokenization preprocessing via sharding avoids CPU starvation
4. **Simple API**: Sharding by default - no memory threshold guessing

## Design Decision: Sharding by Default

**All iterables go through the shard pipeline.** This:
- Eliminates "what's the threshold?" complexity
- Guarantees O(shard) memory bound regardless of corpus size
- Enables parallel processing for everything via `num_proc`
- Gives exact step counts before training starts

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
    # Sharding options (for IterableDataset path)
    preprocess_num_proc: int = 4,
    preprocess_batch_size: int = 1000,
    shard_cache_dir: Optional[str] = None,  # Where to save shards, auto if None
    max_shard_size: str = "500MB",
    cleanup_shards: bool = True,  # Remove after training
) -> DataLoader:
    """
    Single entry point for all data types.
    
    Auto-detects input type:
    - List[str] or duck-typed list-like → DocumentAwareDataset (historical)
    - datasets.Dataset (map-style) → DocumentAwareDataset after extraction
    - datasets.IterableDataset/streaming → Sharded preprocessing pipeline
    
    For streaming datasets: ALWAYS uses preprocess_to_shards() pipeline.
    This gives O(shard) memory + parallel processing + exact counts.
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

**All streaming data goes through the shard pipeline.**

### The Strategy

**Stream → Shards → Fast Loader**

1. Stream from source (HF hub, local files, etc.)
2. Pre-tokenize and chunk into Arrow shards on disk (parallel via `num_proc`)
3. Load shards as fast map-style dataset with exact `__len__`
4. Cleanup shards after training (optional)

### Implementation

```python
def _handle_streaming_iterable(
    iterable,
    text_column: str,
    tokenizer,
    seq_len: int,
    shard_cache_dir: Optional[str],
    preprocess_num_proc: int,
    batch_size: int,
    max_shard_size: str,
    **kwargs
) -> DataLoader:
    """
    Handle IterableDataset by streaming to shards first.
    
    This is the ONLY path for streaming data - no threshold guessing.
    """
    import tempfile
    import os
    
    # Determine shard cache location
    if shard_cache_dir is None:
        shard_cache_dir = tempfile.mkdtemp(prefix="helixlm_training_shards_")
    else:
        os.makedirs(shard_cache_dir, exist_ok=True)
    
    # Stage 1: Stream to shards with parallel tokenization
    print(f"[HelixLM] Preprocessing streaming data to shards...")
    print(f"  Cache: {shard_cache_dir}")
    print(f"  Workers: {preprocess_num_proc}")
    
    shard_path = preprocess_to_shards(
        iterable,
        tokenizer,
        output_dir=shard_cache_dir,
        seq_len=seq_len,
        text_column=text_column,
        num_proc=preprocess_num_proc,
        batch_size=preprocess_batch_size,  # Batch for map(), not training
        max_shard_size=max_shard_size,
    )
    
    # Stage 2: Create fast loader from shards
    loader = _create_prechunked_loader(
        shard_path,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # For DataLoader batch collation
        drop_last=drop_last,
    )
    
    # Attach cleanup info (optional)
    loader._helix_shard_path = shard_path
    loader._helix_cleanup = cleanup_shards
    
    return loader
```

### Why Sharding is Better

| Aspect | Materialize-then-process | Sharding-by-default |
|--------|--------------------------|---------------------|
| Memory bound | O(corpus size) | O(shard size) |
| Parallel processing | Only if materialized | Always via `num_proc` |
| Step counting | After materialization | Exact before training |
| Complexity | Threshold guessing | Simple: always shard |
| Disk usage | Temporary materialization | Persistent shards (cleaned) |
| Restart/recovery | Re-download | Reuse existing shards |

## The Exact Steps Problem (SOLVED)

With sharding-by-default, we always have exact counts from the preprocessed shards:

```python
def _count_preprocessed(preprocessed_ds):
    """Get exact sample count from preprocessed dataset."""
    return len(preprocessed_ds)  # Fast for sharded data
```

The preprocessing step (before training) gives us:
1. Total samples across all shards
2. Exact batch count: `ceil(samples / batch_size)` if `drop_last=False`, else floor
3. Total optimizer steps: `batches // grad_accum_steps * epochs`

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

# 1. List[str] - historical (unchanged)
from helix_lm import create_helix_data_loader
train_loader = create_helix_data_loader(
    texts, tokenizer, seq_len=512, batch_size=8
)

# 2. Map-style dataset Column (extracts texts, then Path 1)
from datasets import load_dataset
ds = load_dataset("my_data", split="train")
loader = create_helix_data_loader(
    ds["text"], tokenizer, seq_len=512, batch_size=8
)

# 3. Streaming dataset Column  ← NEW!
# ALWAYS uses sharded preprocessing for consistent behavior
ds = load_dataset("my_data", split="train", streaming=True)
loader = create_helix_data_loader(
    ds["text"], tokenizer, seq_len=512, batch_size=8,
    preprocess_num_proc=4,  # Parallel tokenization workers
    shard_cache_dir="/tmp/my_training_shards",  # Optional
    cleanup_shards=True,  # Remove after training
)
# Outputs: Fast map-style loader with exact __len__
```

## Testing Plan

1. **Small iterable** (5M tokens): Verify shard pipeline works fast
2. **Large iterable** (500M+ tokens): Verify shard pipeline works  
3. **Exact steps**: Verify progress bar shows percentage
4. **No bottleneck**: Verify GPU utilization stays high (>80%)
5. **Shard cleanup**: Verify temp shards are removed
6. **Backward compat**: Verify List[str] still works unchanged

## Open Questions

1. ✅ **Temp storage**: `shard_cache_dir` parameter - auto-temp if None, user-specified otherwise
2. **Resume**: Skip preprocessing if shards exist? (`overwrite_shards=False` option)
3. **Validation**: Same sharding preprocessing for validation data - keep shards separate?
4. **Shard size**: Is 500MB a good default? Should it scale with `seq_len`?

## Implementation Order

1. ✅ Branch created
2. ⏳ Update `create_unified_data_loader()` with detection logic
3. ⏳ Implement `_handle_small_iterable()`
4. ⏳ Implement `_handle_large_iterable()`
5. ⏳ Update `Trainer` to use preprocessed counts
6. ⏳ Tests
7. ⏳ Documentation
