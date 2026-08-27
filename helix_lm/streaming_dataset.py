"""
HelixLM Streaming Dataset — Contiguous chunk storage for IterableColumn inputs.

Incorporates cofounder review decisions (2026-08-27):
  1. Shard size: 50k–100k samples (maximizes OS page-cache locality)
  2. No "group reads by shard" optimization (sparse reads are native to mmap)
  3. RETAINED: Multi-reservoir producer-consumer thread (background batch collation)
  4. Shard cache holds actual numpy arrays in RAM (not just mmap handles)
  5. Checkpoint metadata (corpus_hash, data_seed) attached to model config
  6. Atomic shard writes (.tmp → rename), disk-space admission, manifest validation
  7. Resumable mid-epoch: deferred to V1 (epoch-boundary resumption only)
  8. Benchmark memmap vs Arrow deferred; memmap kept as default with pluggable backend

Seed isolation:
  The DataLoader / producer thread creates its own torch.Generator seeded explicitly.
  It does NOT touch the global torch RNG, so model initialization order
  and data-loader creation order are fully decoupled.
"""
import os
import json
import bisect
import shutil
import tempfile
import threading
import queue as queue_module
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Iterator, Dict, Any, Tuple, Union, Protocol
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Re-use existing collate and mask utilities from dataset.py
# ---------------------------------------------------------------------------

def _get_dataset_utils():
    """Lazy import to avoid circular dependency with dataset.py."""
    from . import dataset as _dataset_module
    return (
        _dataset_module._collate_batch,
        _dataset_module._build_attention_mask,
        _dataset_module._process_and_shard_batch,
    )


# ---------------------------------------------------------------------------
# ShardBackend Protocol (for future Arrow / alternative backends)
# ---------------------------------------------------------------------------

class ShardBackend(Protocol):
    """Pluggable backend for shard I/O. Memmap default; Arrow can be swapped later."""

    def write_shard(
        self,
        shard_dir: Path,
        input_ids: np.ndarray,
        pad_len: np.ndarray,
        overlap_mask: np.ndarray,
        is_natural_stop: np.ndarray,
    ) -> None: ...

    def read_shard(self, shard_dir: Path) -> Dict[str, np.ndarray]: ...

    def validate_shard(self, shard_dir: Path, expected_size: int) -> bool: ...


class NumpyBackend:
    """Default backend: numpy .npy files with atomic writes."""

    @staticmethod
    def write_shard(shard_dir, input_ids, pad_len, overlap_mask, is_natural_stop):
        tmp_dir = shard_dir.with_suffix(".tmp")
        tmp_dir.mkdir(exist_ok=True)
        np.save(tmp_dir / "input_ids.npy", input_ids)
        np.save(tmp_dir / "pad_len.npy", pad_len)
        np.save(tmp_dir / "overlap_mask.npy", overlap_mask)
        np.save(tmp_dir / "is_natural_stop.npy", is_natural_stop)
        # Atomic commit on POSIX
        os.replace(str(tmp_dir), str(shard_dir))

    @staticmethod
    def read_shard(shard_dir):
        return {
            "input_ids": np.load(shard_dir / "input_ids.npy"),
            "pad_len": np.load(shard_dir / "pad_len.npy"),
            "overlap_mask": np.load(shard_dir / "overlap_mask.npy"),
            "is_natural_stop": np.load(shard_dir / "is_natural_stop.npy"),
        }

    @staticmethod
    def validate_shard(shard_dir, expected_size):
        try:
            ids = np.load(shard_dir / "input_ids.npy", mmap_mode="r")
            return ids.shape[0] == expected_size
        except Exception:
            return False


# ---------------------------------------------------------------------------
# ChunkWriter — sequential, contiguous shard writer with atomic commits
# ---------------------------------------------------------------------------

class ChunkWriter:
    """
    Writes preprocessed (token_ids, is_natural, pad_len, overlap_mask) tuples
    to contiguous numpy files on disk with atomic shard commits.

    Each shard is a directory containing:
        input_ids.npy        (n_samples, seq_len)  uint32
        pad_len.npy          (n_samples,)          uint16
        overlap_mask.npy     (n_samples,)          uint16
        is_natural_stop.npy  (n_samples,)          uint8

    meta.json is written ONLY after all shards are atomically committed.
    """

    def __init__(
        self,
        output_dir: str,
        seq_len: int,
        shard_size: int = 50_000,
        dtype=np.uint32,
        backend: Optional[ShardBackend] = None,
        max_disk_bytes: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len
        self.shard_size = shard_size
        self.dtype = dtype
        self.backend = backend or NumpyBackend()
        self.max_disk_bytes = max_disk_bytes
        self.current_shard_idx = 0
        self.current_buffer: List[Tuple[List[int], bool, int, int]] = []
        self.shard_sizes: List[int] = []
        self._bytes_written = 0

    def _check_disk_space(self, additional_bytes: int) -> None:
        if self.max_disk_bytes is None:
            return
        if self._bytes_written + additional_bytes > self.max_disk_bytes:
            raise RuntimeError(
                f"Disk quota exceeded: {self._bytes_written + additional_bytes} > "
                f"{self.max_disk_bytes} bytes. Increase max_disk_bytes or reduce corpus."
            )

    def _flush_shard(self) -> None:
        if not self.current_buffer:
            return

        n = len(self.current_buffer)
        self.shard_sizes.append(n)

        shard_dir = self.output_dir / f"shard_{self.current_shard_idx:05d}"

        # Allocate contiguous arrays
        input_ids = np.empty((n, self.seq_len), dtype=self.dtype)
        pad_len = np.empty(n, dtype=np.uint16)
        overlap_mask = np.empty(n, dtype=np.uint16)
        is_natural_stop = np.empty(n, dtype=np.uint8)

        for i, (chunk, is_natural, pl, om) in enumerate(self.current_buffer):
            input_ids[i] = chunk
            pad_len[i] = pl
            overlap_mask[i] = om
            is_natural_stop[i] = 1 if is_natural else 0

        self._check_disk_space(
            input_ids.nbytes + pad_len.nbytes + overlap_mask.nbytes + is_natural_stop.nbytes
        )

        self.backend.write_shard(shard_dir, input_ids, pad_len, overlap_mask, is_natural_stop)
        self._bytes_written += (
            input_ids.nbytes + pad_len.nbytes + overlap_mask.nbytes + is_natural_stop.nbytes
        )

        self.current_buffer = []
        self.current_shard_idx += 1

    def write(
        self,
        chunk: List[int],
        is_natural: bool,
        pad_len: int,
        overlap_mask: int,
    ) -> None:
        self.current_buffer.append((chunk, is_natural, pad_len, overlap_mask))
        if len(self.current_buffer) >= self.shard_size:
            self._flush_shard()

    def close(self) -> None:
        self._flush_shard()
        meta = {
            "seq_len": self.seq_len,
            "total_samples": int(sum(self.shard_sizes)),
            "num_shards": len(self.shard_sizes),
            "shard_sizes": [int(s) for s in self.shard_sizes],
            "dtype": str(self.dtype),
            "version": "1.0",
        }
        # Atomic meta write
        meta_tmp = self.output_dir / "meta.json.tmp"
        with open(meta_tmp, "w") as f:
            json.dump(meta, f)
        os.replace(str(meta_tmp), str(self.output_dir / "meta.json"))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# HelixChunkDataset — map-style dataset with RAM-resident shard cache
# ---------------------------------------------------------------------------

class HelixChunkDataset(Dataset):
    """
    Map-style Dataset that reads preprocessed chunks from contiguous numpy files.

    Key properties:
      - No per-sample threads.
      - No pickle deserialization.
      - Compact index: numpy cumulative offsets (not Python tuples).
      - RAM-resident shard cache: actual numpy arrays, not just mmap handles.
        By default (max_cache_shards=None), loads ALL shards into RAM.
        For corpora larger than RAM, set max_cache_shards to an LRU limit.
      - Sequential storage; shuffling is performed by the consumer
        (DataLoader or producer thread) with an independent torch.Generator.

    This produces sample dicts IDENTICAL to DocumentAwareDataset.__getitem__
    (same keys, same tensor shapes, same masking logic).
    """

    def __init__(
        self,
        chunks_dir: str,
        seq_len: Optional[int] = None,
        max_cache_shards: Optional[int] = None,
        backend: Optional[ShardBackend] = None,
    ):
        self.chunks_dir = Path(chunks_dir)
        self.backend = backend or NumpyBackend()

        meta_path = self.chunks_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No meta.json found in {chunks_dir}")

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.seq_len = seq_len or self.meta["seq_len"]
        self.total_samples = self.meta["total_samples"]
        self.num_shards = self.meta["num_shards"]
        self.shard_sizes = self.meta["shard_sizes"]

        # Validate shards against meta.json
        for i, expected_size in enumerate(self.shard_sizes):
            shard_dir = self.chunks_dir / f"shard_{i:05d}"
            if not self.backend.validate_shard(shard_dir, expected_size):
                raise ValueError(
                    f"Shard {i} validation failed: size mismatch or corruption. "
                    f"Delete {chunks_dir} and re-preprocess."
                )

        # Compact O(log S) index — numpy array, not Python list of tuples
        self.shard_offsets = np.cumsum([0] + self.shard_sizes, dtype=np.int64)

        # RAM cache strategy
        self.max_cache_shards = max_cache_shards
        if self.max_cache_shards is None or self.max_cache_shards >= self.num_shards:
            # Load ALL shards into RAM (default for V0 — eliminates disk I/O during training)
            self._all_shards: Dict[int, Dict[str, np.ndarray]] = {
                i: self.backend.read_shard(self.chunks_dir / f"shard_{i:05d}")
                for i in range(self.num_shards)
            }
            self._shard_cache = None
        else:
            # LRU cache for corpora larger than RAM
            self._all_shards = None
            self._shard_cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()

        # Import utilities once
        _, self._build_attention_mask, _ = _get_dataset_utils()

    def _load_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        if self._all_shards is not None:
            return self._all_shards[shard_idx]

        if shard_idx in self._shard_cache:
            self._shard_cache.move_to_end(shard_idx)
            return self._shard_cache[shard_idx]

        shard_dir = self.chunks_dir / f"shard_{shard_idx:05d}"
        shard_data = self.backend.read_shard(shard_dir)

        # LRU eviction
        while len(self._shard_cache) >= self.max_cache_shards:
            self._shard_cache.popitem(last=False)

        self._shard_cache[shard_idx] = shard_data
        return shard_data

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0:
            idx += self.total_samples
        if not 0 <= idx < self.total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.total_samples})")

        # O(log S) lookup via numpy array
        shard_idx = int(bisect.bisect_right(self.shard_offsets, idx) - 1)
        local_idx = int(idx - self.shard_offsets[shard_idx])

        shard = self._load_shard(shard_idx)

        # Read from RAM-resident arrays — zero-copy view
        chunk = shard["input_ids"][local_idx]
        pad_len = int(shard["pad_len"][local_idx])
        overlap_mask = int(shard["overlap_mask"][local_idx])
        is_natural = bool(shard["is_natural_stop"][local_idx])

        # Build sample — IDENTICAL logic to DocumentAwareDataset.__getitem__
        x = torch.from_numpy(chunk.astype(np.int64, copy=False))
        labels = x.clone()

        if overlap_mask > 0:
            labels[:overlap_mask] = -100
        if pad_len > 0:
            labels[-pad_len:] = -100

        attention_mask = self._build_attention_mask(self.seq_len, pad_len)

        return {
            "input_ids": x,
            "labels": labels,
            "attention_mask": attention_mask,
            "is_natural_stop": torch.tensor(is_natural, dtype=torch.bool),
        }


# ---------------------------------------------------------------------------
# HelixStreamingDataLoader — producer-consumer with background batch collation
# ---------------------------------------------------------------------------

class HelixStreamingDataLoader:
    """
    Iterable that produces batches via a background producer thread.

    Mimics the PyTorch DataLoader interface (__iter__, __len__) so it can be
    passed directly to Trainer(train_loader=...) without any Trainer changes.

    Producer thread responsibilities:
      1. Generate global permutation from seed (independent torch.Generator)
      2. Read samples from HelixChunkDataset (RAM-resident, fast)
      3. Collate batches using the existing _collate_batch function
      4. Pin memory for async H2D transfer
      5. Place batches in a bounded queue

    Training thread simply pops batches from the queue.
    """

    def __init__(
        self,
        dataset: HelixChunkDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        queue_size: int = 64,
        pin_memory: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.queue: "queue_module.Queue[Optional[Dict[str, torch.Tensor]]]" = (
            queue_module.Queue(maxsize=queue_size)
        )
        self._collate_batch, _, _ = _get_dataset_utils()
        self._producer = threading.Thread(target=self._produce, daemon=True)
        self._producer.start()

    def _produce(self) -> None:
        # Seed isolation: fresh Generator, explicit seed, no global RNG touch
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        if self.shuffle:
            perm = torch.randperm(len(self.dataset), generator=generator)
            indices = perm.tolist()
        else:
            indices = list(range(len(self.dataset)))

        batch: List[Dict[str, torch.Tensor]] = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                self._put_batch(batch)
                batch = []

        # Remainder
        if batch and not self.drop_last:
            self._put_batch(batch)

        self.queue.put(None)  # Sentinel

    def _put_batch(self, batch: List[Dict[str, torch.Tensor]]) -> None:
        collated = self._collate_batch(batch)
        if self.pin_memory:
            collated = {k: v.pin_memory() for k, v in collated.items()}
        self.queue.put(collated)

    def __iter__(self):
        while True:
            batch = self.queue.get()
            if batch is None:
                break
            yield batch

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# Preprocessing API
# ---------------------------------------------------------------------------

def preprocess_streaming_data(
    iterable,
    tokenizer,
    seq_len: int,
    output_dir: str,
    stride: Optional[int] = None,
    min_tail_len: int = 1,
    add_eos: bool = True,
    text_column: str = "text",
    shard_size: int = 50_000,
    max_disk_bytes: Optional[int] = None,
    show_progress: bool = True,
) -> str:
    """
    Preprocess an IterableColumn (or any iterator of texts) into contiguous
    chunk files on disk with atomic shard commits.

    This is a single sequential pass over the iterable. Tokenization happens
    in batches for efficiency, but order is strictly preserved.

    Returns:
        output_dir: path to the directory containing shards + meta.json
    """
    from tqdm import tqdm

    def extract_text(example):
        if isinstance(example, dict):
            return example.get(text_column, "")
        return str(example)

    _, _, _process_and_shard_batch = _get_dataset_utils()

    stride = stride if stride is not None else seq_len
    writer = ChunkWriter(
        output_dir, seq_len, shard_size=shard_size, max_disk_bytes=max_disk_bytes
    )

    batch = []
    batch_size = 1_000

    iterator = iterable
    if show_progress:
        iterator = tqdm(iterator, desc="Preprocessing shards", unit="doc")

    for example in iterator:
        text = extract_text(example)
        if text:
            batch.append(text)
        if len(batch) >= batch_size:
            chunks = _process_and_shard_batch(
                batch, tokenizer, seq_len, stride, min_tail_len, add_eos
            )
            for chunk, is_natural, pl, om in chunks:
                writer.write(chunk, is_natural, pl, om)
            batch = []

    # Final batch
    if batch:
        chunks = _process_and_shard_batch(
            batch, tokenizer, seq_len, stride, min_tail_len, add_eos
        )
        for chunk, is_natural, pl, om in chunks:
            writer.write(chunk, is_natural, pl, om)

    writer.close()
    return output_dir


def compute_corpus_hash(chunks_dir: str) -> str:
    """Compute a stable hash of the corpus manifest for checkpoint binding."""
    import hashlib
    meta_path = Path(chunks_dir) / "meta.json"
    with open(meta_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Loader creation — matches create_document_loader semantics exactly
# ---------------------------------------------------------------------------

def create_streaming_loader(
    chunks_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = 42,
    queue_size: int = 64,
    max_cache_shards: Optional[int] = None,
) -> HelixStreamingDataLoader:
    """
    Create a HelixStreamingDataLoader from preprocessed chunk files.

    This produces the EXACT same sample sequence as create_document_loader
    when given the same seed, because both use an independent torch.Generator
    with the same seed value.

    The returned loader is DataLoader-compatible (__iter__, __len__) and can
    be passed directly to Trainer(train_loader=...).

    Seed isolation note:
        We create a fresh torch.Generator() and seed it explicitly.
        This does NOT consume or alter the global torch RNG state,
        so model weight initialization and data ordering are fully decoupled.
    """
    dataset = HelixChunkDataset(chunks_dir, max_cache_shards=max_cache_shards)
    return HelixStreamingDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
        queue_size=queue_size,
    )
