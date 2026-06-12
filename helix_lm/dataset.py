"""
HelixLM Dataset with rolling text chunking and natural stop detection.

Key fixes in this revision
  * DocumentAwareDataset now tracks exact pad_len in every chunk tuple.
    It NEVER scans backwards for pad_token_id, so GPT-2 (pad_id == eos_id)
    cannot accidentally mask a real EOS.
  * Optional within-document overlap (stride) added to DocumentAwareDataset.
    stride == seq_len  -> non-overlapping (default).
    stride <  seq_len  -> overlapping windows; overlap is masked in labels.
  * No cross-document boundaries are ever crossed.

Compatible with both eager and lazy loading.
"""
import random
from typing import List, Optional, Iterator, Dict, Any, Union, Tuple

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from tqdm import tqdm


class HelixDataset(Dataset):
    """
    Index-based Dataset with rolling chunking for language model pretraining.
    Compatible with DataLoader(shuffle=True) natively.

    Handles three scenarios:
      1. Text >> seq_len: rolling window with configurable stride
      2. Text == seq_len: exact match
      3. Text < seq_len: padding to seq_len (with attention mask)

    For each chunk, produces:
      - input_ids: (seq_len,)
      - labels: (seq_len,) — shifted by 1 for next-token prediction
      - attention_mask: (seq_len,) — 1 for real tokens, 0 for padding
      - is_natural_stop: scalar bool — True if chunk ends at document boundary
    """
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        seq_len: int = 2048,
        stride: Optional[int] = None,
        lazy: bool = True,
        add_eos: bool = True,
        natural_stop_threshold: float = 0.8,
    ):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or max(1, seq_len // 2)
        self.lazy = lazy
        self.add_eos = add_eos
        self.natural_stop_threshold = natural_stop_threshold

        if lazy:
            self._tokenized_docs = None
            self._chunk_index = None
        else:
            self._tokenized_docs = self._tokenize_all()
            self._chunk_index = self._build_chunk_index()

    def _tokenize_all(self) -> List[Dict[str, Any]]:
        docs = []
        iterable = tqdm(self.texts, desc="Tokenizing", unit="doc", disable=len(self.texts) < 1000)
        for text in iterable:
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.add_eos and hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                ids.append(self.tokenizer.eos_token_id)
            docs.append({"ids": ids, "length": len(ids)})
        return docs

    def _build_chunk_index(self) -> List[Tuple[int, int, bool]]:
        index = []
        for doc_idx, doc in enumerate(self._tokenized_docs):
            length = doc["length"]
            if length == 0:
                continue
            ids = doc["ids"]
            if length >= self.seq_len:
                for start_idx in range(0, length - self.seq_len + 1, self.stride):
                    end_idx = start_idx + self.seq_len
                    is_natural_stop = end_idx >= length * self.natural_stop_threshold
                    index.append((doc_idx, start_idx, is_natural_stop))
                    if end_idx >= length:
                        break
            else:
                index.append((doc_idx, 0, True))
        return index

    def _build_lazy_chunk_index(self) -> List[Tuple[int, int, int, bool]]:
        index = []
        for doc_idx, text in enumerate(self.texts):
            text = text.strip()
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.add_eos and hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                ids.append(self.tokenizer.eos_token_id)
            length = len(ids)
            if length == 0:
                continue
            if length >= self.seq_len:
                for start_idx in range(0, length - self.seq_len + 1, self.stride):
                    end_idx = start_idx + self.seq_len
                    is_natural_stop = end_idx >= length * self.natural_stop_threshold
                    index.append((doc_idx, start_idx, length, is_natural_stop))
                    if end_idx >= length:
                        break
            else:
                index.append((doc_idx, 0, length, True))
        return index

    def __len__(self) -> int:
        if self._chunk_index is not None:
            return len(self._chunk_index)
        if not hasattr(self, '_lazy_index') or self._lazy_index is None:
            self._lazy_index = self._build_lazy_chunk_index()
        return len(self._lazy_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._chunk_index is not None:
            doc_idx, start_idx, is_natural_stop = self._chunk_index[idx]
            ids = self._tokenized_docs[doc_idx]["ids"]
            length = self._tokenized_docs[doc_idx]["length"]
        else:
            if not hasattr(self, '_lazy_index') or self._lazy_index is None:
                self._lazy_index = self._build_lazy_chunk_index()
            doc_idx, start_idx, length, is_natural_stop = self._lazy_index[idx]
            text = self.texts[doc_idx].strip()
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.add_eos and hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                ids.append(self.tokenizer.eos_token_id)

        if length >= self.seq_len:
            end_idx = start_idx + self.seq_len
            chunk = ids[start_idx:end_idx]
            labels = list(chunk)
            if start_idx > 0 and self.stride < self.seq_len:
                warmup_len = self.seq_len - self.stride
                labels[:warmup_len] = [-100] * warmup_len
            return self._make_sample(chunk, labels, is_natural_stop)
        else:
            chunk = ids[:length]
            pad_len = self.seq_len - length
            chunk = chunk + [self.tokenizer.pad_token_id] * pad_len
            labels = list(chunk)
            if pad_len > 0:
                labels[-pad_len:] = [-100] * pad_len
            return self._make_sample(chunk, labels, is_natural_stop=True)

    def _make_sample(self, chunk, labels, is_natural_stop):
        input_ids = torch.tensor(chunk[:self.seq_len], dtype=torch.long)
        labels_t = torch.tensor(labels[:self.seq_len], dtype=torch.long)
        # BUG FIX: Derive attention_mask from ANY -100 position (both padding AND overlap)
        # This ensures attention_mask is consistent with labels
        attention_mask = (labels_t != -100).long()
        return {
            "input_ids": input_ids,
            "labels": labels_t,
            "attention_mask": attention_mask,
            "is_natural_stop": torch.tensor(is_natural_stop, dtype=torch.bool),
        }


class DocumentAwareDataset(Dataset):
    """
    Per-document chunking with no cross-document boundaries.

    Chunk tuple format (built in _build_chunks):
        (token_ids, is_natural_stop, pad_len, overlap_mask_len)

      * pad_len: exact number of padded positions at the TAIL.
                 Used to set labels[-pad_len:] = -100.
                 Always 0 for full chunks.
      * overlap_mask_len: number of positions at the HEAD to mask with -100.
                 Used when stride < seq_len so overlapping tokens are not
                 double-counted in loss. Always 0 when stride == seq_len.

    For each document:
      - Long documents: split into seq_len chunks (optionally overlapping).
      - Short documents: kept as-is, padded to seq_len.
      - Only padding positions are masked in labels (-100).
      - No label masking for overlap regions except the explicit overlap head.
    """
    def __init__(
        self,
        texts,
        tokenizer,
        seq_len,
        min_tail_len=None,
        add_eos=True,
        lazy=True,
        stride: Optional[int] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # Robust pad_id fallback: if unset, fall back to eos_id, then 0.
        self.pad_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_id is None:
            self.pad_id = getattr(tokenizer, "eos_token_id", 0)
        self.eos_id = getattr(tokenizer, "eos_token_id", None)

        self.lazy = lazy
        self.stride = stride if stride is not None else seq_len
        if not (1 <= self.stride <= self.seq_len):
            raise ValueError(f"stride must be in [1, seq_len], got {self.stride}")

        if min_tail_len is None:
            min_tail_len = 1  # Keep all documents; was seq_len//4 which dropped most docs at 512
        self.min_tail_len = min_tail_len
        self.add_eos = add_eos

        if lazy:
            self.texts = texts
            self.chunks = None
            self._stats = None
        else:
            self.texts = None
            self.chunks, self._stats = self._build_chunks(texts)

    def _build_chunks(self, texts):
        """
        Build chunk tuples:
          (token_ids: List[int], is_natural: bool, pad_len: int, overlap_mask: int)
        """
        chunks = []
        dropped_short = dropped_tail = kept = 0
        stride = self.stride

        for text in tqdm(texts, desc="Chunking", unit="doc", disable=len(texts) < 1000):
            text = text.strip()
            if not text:
                continue

            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < self.min_tail_len:
                dropped_short += 1
                continue

            if self.add_eos and self.eos_id is not None:
                ids.append(self.eos_id)

            length = len(ids)

            if length >= self.seq_len:
                # Sliding-window chunks fully inside the document
                starts = list(range(0, length - self.seq_len + 1, stride))
                for i, start in enumerate(starts):
                    chunk = ids[start:start + self.seq_len]
                    is_last_start = (i == len(starts) - 1)
                    reaches_end = (start + self.seq_len == length)

                    # Natural stop logic: if this is the last chunk we will emit
                    # and there is no tail (or tail is too short), mark it.
                    if not is_last_start:
                        is_natural = False
                    else:
                        tail_len = length - (start + self.seq_len)
                        has_tail = tail_len >= self.min_tail_len
                        is_natural = reaches_end or (not has_tail)

                    overlap_mask = 0
                    if start > 0 and stride < self.seq_len:
                        overlap_mask = self.seq_len - stride

                    chunks.append((chunk, is_natural, 0, overlap_mask))
                    kept += 1

                # Tail: tokens after the last sliding chunk
                last_covered_end = starts[-1] + self.seq_len if starts else 0
                remainder_len = length - last_covered_end

                if remainder_len >= self.min_tail_len:
                    tail = ids[last_covered_end:]
                    pad_len = self.seq_len - remainder_len
                    tail = tail + [self.pad_id] * pad_len
                    chunks.append((tail, True, pad_len, 0))
                    kept += 1
                elif remainder_len > 0:
                    # Tail too short to keep: last sliding chunk becomes the doc end
                    if starts:
                        last_chunk, _, last_pad, last_overlap = chunks[-1]
                        chunks[-1] = (last_chunk, True, last_pad, last_overlap)
                    dropped_tail += 1

            else:
                # Short document: pad once, everything is a natural stop
                pad_len = self.seq_len - length
                chunk = ids + [self.pad_id] * pad_len
                chunks.append((chunk, True, pad_len, 0))
                kept += 1

        stats = {
            "kept": kept,
            "dropped_short": dropped_short,
            "dropped_tail": dropped_tail,
        }
        return chunks, stats

    def _ensure_chunks(self):
        if self.chunks is None:
            self.chunks, self._stats = self._build_chunks(self.texts)
            self.texts = None  # free memory

    def __len__(self):
        self._ensure_chunks()
        return len(self.chunks)

    def __getitem__(self, idx):
        self._ensure_chunks()
        chunk, is_natural, pad_len, overlap_mask = self.chunks[idx]

        x = torch.tensor(chunk, dtype=torch.long)
        labels = x.clone()

        # 1. Mask overlapping head (only when stride < seq_len)
        if overlap_mask > 0:
            labels[:overlap_mask] = -100

        # 2. Mask exact trailing padding count (robust to pad_id == eos_id)
        if pad_len > 0:
            labels[-pad_len:] = -100

        # BUG FIX: Derive attention_mask from ANY -100 position (both padding AND overlap)
        # This ensures attention_mask is consistent with labels
        attention_mask = (labels != -100).long()
        return {
            "input_ids": x,
            "labels": labels,
            "attention_mask": attention_mask,
            "is_natural_stop": torch.tensor(is_natural, dtype=torch.bool),
        }

    def get_stats(self):
        self._ensure_chunks()
        return self._stats


class HelixDatasetFromTokens(Dataset):
    """
    Dataset from pre-tokenized token stream (e.g., from HF datasets).
    Handles rolling chunking over a long token sequence.
    """
    def __init__(
        self,
        tokens: Union[List[int], torch.Tensor],
        seq_len: int = 2048,
        stride: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride or max(1, seq_len // 2)

        n = len(self.tokens)
        self.indices = list(range(0, max(1, n - seq_len), self.stride))
        if n >= seq_len and (n - seq_len) % self.stride != 0:
            self.indices.append(n - seq_len)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = x.clone()
        return {
            "input_ids": x,
            "labels": y,
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "is_natural_stop": torch.tensor(end >= len(self.tokens) * 0.9, dtype=torch.bool),
        }


class HelixHFDataset(Dataset):
    """
    Wrapper for HuggingFace datasets with streaming and non-streaming support.
    Uses DocumentAwareDataset internally, so no cross-document boundaries.
    """
    def __init__(
        self,
        hf_dataset: Union[str, Any],
        tokenizer,
        seq_len: int = 2048,
        text_column: str = "text",
        stride: Optional[int] = None,
        max_samples: Optional[int] = None,
        lazy: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_column = text_column
        self.lazy = lazy
        self.max_samples = max_samples

        if isinstance(hf_dataset, str):
            from datasets import load_dataset
            dataset_name = hf_dataset
            known_script_datasets = {"openai_humaneval", "bigcode/the-stack", "bigcode/the-stack-v2"}
            load_kwargs = dict(kwargs)
            if dataset_name not in known_script_datasets:
                load_kwargs.pop("trust_remote_code", None)
            self.dataset = load_dataset(dataset_name, **load_kwargs)
        else:
            self.dataset = hf_dataset

        if hasattr(self.dataset, "keys") and hasattr(self.dataset, "__getitem__"):
            if "train" in self.dataset:
                self.dataset = self.dataset["train"]
            else:
                self.dataset = self.dataset[list(self.dataset.keys())[0]]

        if max_samples is not None:
            if hasattr(self.dataset, "take") and hasattr(self.dataset, "__iter__"):
                self.dataset = self.dataset.take(max_samples)
            elif hasattr(self.dataset, "select"):
                indices = list(range(min(max_samples, len(self.dataset))))
                self.dataset = self.dataset.select(indices)

        # Extract texts
        if hasattr(self.dataset, "__iter__") and not hasattr(self.dataset, "__getitem__"):
            self._texts = []
            iterable = tqdm(self.dataset, desc="Loading HF dataset", unit="sample",
                            total=max_samples, disable=max_samples is not None and max_samples < 1000)
            for example in iterable:
                text = example.get(self.text_column, "")
                if text:
                    self._texts.append(text)
                if max_samples is not None and len(self._texts) >= max_samples:
                    break
            self._dataset_type = "list"
        else:
            if hasattr(self.dataset, "__getitem__") and hasattr(self.dataset, "__len__"):
                iterable = tqdm(range(len(self.dataset)), desc="Loading HF dataset", unit="sample",
                                disable=len(self.dataset) < 1000)
                self._texts = [self.dataset[i].get(self.text_column, "") for i in iterable]
                self._texts = [t for t in self._texts if t]
                self._dataset_type = "map"
            else:
                self._texts = list(self.dataset)
                self._dataset_type = "list"

        self._doc_dataset = DocumentAwareDataset(
            self._texts, tokenizer, seq_len,
            min_tail_len=1, add_eos=True, lazy=lazy, stride=stride,
        )

    def __len__(self) -> int:
        return len(self._doc_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._doc_dataset[idx]

    def get_stats(self):
        return self._doc_dataset.get_stats()


def create_helix_dataloader(
    texts: List[str],
    tokenizer,
    seq_len: int = 2048,
    batch_size: int = 8,
    stride: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    lazy: bool = True,
    **kwargs,
) -> torch.utils.data.DataLoader:
    dataset = HelixDataset(texts, tokenizer, seq_len, stride, lazy=lazy, **kwargs)

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        is_natural_stop = torch.stack([b["is_natural_stop"] for b in batch])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "is_natural_stop": is_natural_stop,
        }

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def create_document_loader(
    texts: List[str],
    tokenizer,
    seq_len: int = 2048,
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    min_tail_len: Optional[int] = None,
    add_eos: bool = True,
    lazy: bool = True,
    stride: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader using DocumentAwareDataset (no boundary crossings).

    Args:
        stride: If < seq_len, enables within-document overlap (default: seq_len).
                This restores more optimizer steps per epoch without ever
                crossing document boundaries.
        seed: RNG seed for shuffling (uses torch.Generator for determinism).
    """
    ds = DocumentAwareDataset(
        texts, tokenizer, seq_len,
        min_tail_len=min_tail_len, add_eos=add_eos, lazy=lazy, stride=stride,
    )

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        is_natural_stop = torch.stack([b["is_natural_stop"] for b in batch])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "is_natural_stop": is_natural_stop,
        }

    # Use torch.Generator for deterministic shuffling with given seed
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=drop_last,
            generator=generator,
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=drop_last,
        )


# =============================================================================
# Streaming Dataset Support (New API for 87M/3B Token Scale-Up)
# =============================================================================

def _is_iterable_column(data) -> bool:
    """
    Detect if data should use streaming path (vs materialized List[str]).
    
    Streaming data:
        - Has __iter__ (can be iterated)
        - Does NOT have working __len__ (len() raises or doesn't exist)
    
    Materialized data:
        - Has working __len__ (len() returns a number)
    
    Examples:
        - List[str]: materialized (has len)
        - IterableColumn: streaming (no len, has iter)
        - Generator: streaming (no len, has iter)
    """
    # Must be iterable
    if not hasattr(data, '__iter__'):
        return False
    
    # Check if len() works
    if hasattr(data, '__len__'):
        try:
            len(data)  # If this works, it's materialized
            return False
        except (TypeError, NotImplementedError):
            pass  # No working len, continue to check
    
    # Has iter but no working len → streaming
    return True


def _chunk_text_streaming(
    texts_iter: Iterator[str],
    tokenizer,
    seq_len: int,
    stride: Optional[int] = None,
    min_tail_len: int = 1,
    add_eos: bool = True,
) -> Iterator[Tuple[List[int], bool, int, int]]:
    """
    Streaming text chunker that yields (token_ids, is_natural_stop, pad_len, overlap_mask).
    
    Memory bounded: processes one document at a time.
    """
    stride = stride if stride is not None else seq_len
    if not (1 <= stride <= seq_len):
        raise ValueError(f"stride must be in [1, seq_len], got {stride}")
    
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    
    for text in texts_iter:
        text = text.strip() if text else ""
        if not text:
            continue
        
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < min_tail_len:
            continue
        
        if add_eos and eos_id is not None:
            ids.append(eos_id)
        
        length = len(ids)
        
        if length >= seq_len:
            # Sliding-window chunks
            starts = list(range(0, length - seq_len + 1, stride))
            for i, start in enumerate(starts):
                chunk = ids[start:start + seq_len]
                is_last_start = (i == len(starts) - 1)
                reaches_end = (start + seq_len == length)
                
                if not is_last_start:
                    is_natural = False
                else:
                    tail_len = length - (start + seq_len)
                    has_tail = tail_len >= min_tail_len
                    is_natural = reaches_end or (not has_tail)
                
                overlap_mask = 0
                if start > 0 and stride < seq_len:
                    overlap_mask = seq_len - stride
                
                yield (chunk, is_natural, 0, overlap_mask)
            
            # Tail handling
            last_covered_end = starts[-1] + seq_len if starts else 0
            remainder_len = length - last_covered_end
            
            if remainder_len >= min_tail_len:
                tail = ids[last_covered_end:]
                pad_len = seq_len - remainder_len
                tail = tail + [pad_id] * pad_len
                yield (tail, True, pad_len, 0)
            elif remainder_len > 0 and starts:
                # Tail too short: last chunk becomes natural stop
                # (handled by setting is_natural=True in last yielded)
                pass
        else:
            # Short document
            pad_len = seq_len - length
            chunk = ids + [pad_id] * pad_len
            yield (chunk, True, pad_len, 0)


class HelixPrechunkedDataset(Dataset):
    """
    Dataset from pre-chunked token sequences (for sharded preprocessing).
    
    Each sample is a tuple: (token_ids, is_natural_stop, pad_len, overlap_mask)
    """
    def __init__(self, chunks: List[Tuple[List[int], bool, int, int]], seq_len: int):
        super().__init__()
        self.chunks = chunks
        self.seq_len = seq_len
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk, is_natural, pad_len, overlap_mask = self.chunks[idx]
        
        x = torch.tensor(chunk, dtype=torch.long)
        labels = x.clone()
        
        # Mask overlapping head (stride < seq_len)
        if overlap_mask > 0:
            labels[:overlap_mask] = -100
        
        # Mask trailing padding
        if pad_len > 0:
            labels[-pad_len:] = -100
        
        # Derive attention_mask from labels (any -100 position)
        attention_mask = (labels != -100).long()
        
        return {
            "input_ids": x,
            "labels": labels,
            "attention_mask": attention_mask,
            "is_natural_stop": torch.tensor(is_natural, dtype=torch.bool),
        }


class HelixShardedDataset(Dataset):
    """
    Dataset that reads pre-chunked token sequences from multiple shard files on disk.
    
    This allows handling datasets larger than memory by:
    - Loading only one shard at a time into memory
    - Maintaining a global index across all shards
    - Random access via shard + local_index calculation
    
    Each shard file contains pickled List[Tuple[...]] saved by _handle_streaming_iterable.
    """
    def __init__(
        self,
        shard_paths: List[str],
        seq_len: int,
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.shard_paths = shard_paths
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed
        
        # Build shard index: cumulative offsets for O(1) __getitem__
        self.shard_sizes = []
        self.shard_offsets = [0]
        total = 0
        
        # Get sizes without loading full data
        import pickle
        for path in shard_paths:
            with open(path, 'rb') as f:
                chunks = pickle.load(f)
                size = len(chunks)
                self.shard_sizes.append(size)
                total += size
                self.shard_offsets.append(total)
        
        self.total_size = total
        
        # If shuffling, build a permutation index
        self._permutation = None
        if shuffle:
            self._build_permutation()
        
        # Cache for current shard to avoid repeated disk reads
        self._cache_shard_idx: Optional[int] = None
        self._cache_shard_data: Optional[List] = None
    
    def _build_permutation(self):
        """Build deterministic permutation for shuffling."""
        indices = list(range(self.total_size))
        random.Random(self.seed).shuffle(indices)
        self._permutation = indices
    
    def __len__(self) -> int:
        return self.total_size
    
    def _global_to_local(self, idx: int) -> Tuple[int, int]:
        """Convert global index to (shard_idx, local_idx)."""
        # Binary search for shard
        lo, hi = 0, len(self.shard_offsets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self.shard_offsets[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        shard_idx = lo
        local_idx = idx - self.shard_offsets[shard_idx]
        return shard_idx, local_idx
    
    def _load_shard(self, shard_idx: int) -> List:
        """Load shard data with caching."""
        if self._cache_shard_idx == shard_idx:
            return self._cache_shard_data
        
        import pickle
        with open(self.shard_paths[shard_idx], 'rb') as f:
            self._cache_shard_data = pickle.load(f)
            self._cache_shard_idx = shard_idx
        return self._cache_shard_data
    
    def _item_from_chunk(self, chunk_data: Tuple) -> Dict[str, torch.Tensor]:
        """Convert chunk tuple to sample dict."""
        chunk, is_natural, pad_len, overlap_mask = chunk_data
        
        x = torch.tensor(chunk, dtype=torch.long)
        labels = x.clone()
        
        # Mask overlapping head (stride < seq_len)
        if overlap_mask > 0:
            labels[:overlap_mask] = -100
        
        # Mask trailing padding
        if pad_len > 0:
            labels[-pad_len:] = -100
        
        # Derive attention_mask from labels (any -100 position)
        attention_mask = (labels != -100).long()
        
        return {
            "input_ids": x,
            "labels": labels,
            "attention_mask": attention_mask,
            "is_natural_stop": torch.tensor(is_natural, dtype=torch.bool),
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Handle negative indices
        if idx < 0:
            idx = self.total_size + idx
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range [0, {self.total_size})")
        
        # If shuffling, map to permuted index
        if self._permutation is not None:
            idx = self._permutation[idx]
        
        shard_idx, local_idx = self._global_to_local(idx)
        shard_data = self._load_shard(shard_idx)
        chunk_data = shard_data[local_idx]
        
        return self._item_from_chunk(chunk_data)


def _process_and_shard_batch(
    texts: List[str],
    tokenizer,
    seq_len: int,
    stride: Optional[int] = None,
    min_tail_len: int = 1,
    add_eos: bool = True,
) -> List[Tuple[List[int], bool, int, int]]:
    """
    Process a batch of texts and return pre-chunked samples.
    
    This is used during streaming preprocessing to build shards.
    """
    chunks = []
    stride = stride if stride is not None else seq_len
    
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    
    for text in texts:
        text = text.strip() if text else ""
        if not text:
            continue
        
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < min_tail_len:
            continue
        
        if add_eos and eos_id is not None:
            ids.append(eos_id)
        
        length = len(ids)
        
        if length >= seq_len:
            # Sliding-window chunks
            starts = list(range(0, length - seq_len + 1, stride))
            for i, start in enumerate(starts):
                chunk = ids[start:start + seq_len]
                is_last_start = (i == len(starts) - 1)
                reaches_end = (start + seq_len == length)
                
                if not is_last_start:
                    is_natural = False
                else:
                    tail_len = length - (start + seq_len)
                    has_tail = tail_len >= min_tail_len
                    is_natural = reaches_end or (not has_tail)
                
                # BUG FIX: Calculate overlap_mask correctly
                overlap_mask = 0
                if start > 0 and stride < seq_len:
                    overlap_mask = seq_len - stride
                
                chunks.append((chunk, is_natural, 0, overlap_mask))
            
            # Tail handling
            last_covered_end = starts[-1] + seq_len if starts else 0
            remainder_len = length - last_covered_end
            
            if remainder_len >= min_tail_len:
                tail = ids[last_covered_end:]
                pad_len = seq_len - remainder_len
                tail = tail + [pad_id] * pad_len
                chunks.append((tail, True, pad_len, 0))
        else:
            # Short document
            pad_len = seq_len - length
            chunk = ids + [pad_id] * pad_len
            chunks.append((chunk, True, pad_len, 0))
    
    return chunks


def _handle_streaming_iterable(
    iterable,
    tokenizer,
    seq_len: int = 2048,
    batch_size: int = 8,
    stride: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    min_tail_len: Optional[int] = None,
    add_eos: bool = True,
    seed: int = 42,
    text_column: str = "text",
    shard_cache_dir: Optional[str] = None,
    preprocess_num_proc: Optional[int] = None,
    preprocess_batch_size: int = 1000,
    max_shard_size: str = "500MB",
    cleanup_shards: bool = True,
) -> Tuple[DataLoader, Optional[str]]:
    """
    Handle streaming IterableColumn by preprocessing to sharded Dataset.
    
    Strategy: Stream -> Shards -> Fast Loader
    1. Stream data in batches to shards on disk
    2. Pre-tokenize each shard
    3. Return DataLoader from concatenated shards
    
    Returns:
        Tuple of (DataLoader, shard_cache_dir)
        shard_cache_dir should be cleaned up after training.
    """
    import tempfile
    import os
    import pickle
    from datetime import datetime
    
    if min_tail_len is None:
        min_tail_len = 1
    
    # Create cache directory
    if shard_cache_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shard_cache_dir = os.path.join(tempfile.gettempdir(), f"helix_shards_{timestamp}")
    os.makedirs(shard_cache_dir, exist_ok=True)
    
    # Determine text extraction method
    def extract_text(example):
        if isinstance(example, dict):
            return example.get(text_column, "")
        return str(example)
    
    # Stream and shard
    all_chunks = []
    shard_idx = 0
    batch = []
    
    for example in iterable:
        text = extract_text(example)
        if text:
            batch.append(text)
        
        if len(batch) >= preprocess_batch_size:
            # Process batch
            chunks = _process_and_shard_batch(
                batch, tokenizer, seq_len, stride, min_tail_len, add_eos
            )
            all_chunks.extend(chunks)
            batch = []
            
            # Save shard if it gets large
            if len(all_chunks) >= 10000:  # ~10k sequences per shard
                shard_path = os.path.join(shard_cache_dir, f"shard_{shard_idx:04d}.pkl")
                with open(shard_path, 'wb') as f:
                    pickle.dump(all_chunks, f)
                all_chunks = []
                shard_idx += 1
    
    # Process remaining batch
    if batch:
        chunks = _process_and_shard_batch(
            batch, tokenizer, seq_len, stride, min_tail_len, add_eos
        )
        all_chunks.extend(chunks)
    
    # Save final shard
    if all_chunks:
        shard_path = os.path.join(shard_cache_dir, f"shard_{shard_idx:04d}.pkl")
        with open(shard_path, 'wb') as f:
            pickle.dump(all_chunks, f)
        shard_idx += 1
    
    # Build list of shard paths (do NOT load them into memory)
    shard_paths = []
    for i in range(shard_idx):
        shard_path = os.path.join(shard_cache_dir, f"shard_{i:04d}.pkl")
        if os.path.exists(shard_path):
            shard_paths.append(shard_path)
    
    # Create sharded dataset that reads from disk on-demand
    dataset = HelixShardedDataset(shard_paths, seq_len, shuffle=shuffle, seed=seed)
    
    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        is_natural_stop = torch.stack([b["is_natural_stop"] for b in batch])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "is_natural_stop": is_natural_stop,
        }
    
    # Create DataLoader with proper shuffle
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=drop_last,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=drop_last,
        )
    
    return loader, shard_cache_dir


def create_unified_data_loader(
    data: Union[List[str], Any],
    tokenizer,
    seq_len: int = 2048,
    batch_size: int = 8,
    stride: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    min_tail_len: Optional[int] = None,
    add_eos: bool = True,
    lazy: bool = True,
    seed: int = 42,
    text_column: str = "text",
    # Sharding options (for IterableColumn path)
    shard_cache_dir: Optional[str] = None,
    preprocess_num_proc: Optional[int] = None,
    preprocess_batch_size: int = 1000,
    max_shard_size: str = "500MB",
    cleanup_shards: bool = True,
) -> Union[DataLoader, Tuple[DataLoader, Optional[str]]]:
    """
    Create DataLoader that automatically detects data type:
    - List[str] -> DocumentAwareDataset
    - Column[str] (with __getitem__) -> DocumentAwareDataset  
    - IterableColumn[str] (streaming) -> Sharded preprocessing -> HelixPrechunkedDataset
    
    For streaming data, returns (DataLoader, shard_cache_dir) tuple.
    The caller should clean up shard_cache_dir after training.
    
    Args:
        data: Input data - List[str], Column[str], or IterableColumn[str]
        tokenizer: Tokenizer instance
        seq_len: Sequence length for model
        batch_size: Batch size for DataLoader
        stride: If < seq_len, enables within-document overlap
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop last incomplete batch
        num_workers: Number of worker processes for DataLoader
        min_tail_len: Minimum tail length for document handling
        add_eos: Whether to add EOS token
        lazy: Whether to use lazy loading for DocumentAwareDataset
        seed: Random seed for shuffling
        text_column: Column name for text extraction from dicts
        shard_cache_dir: Directory for temporary shard storage (streaming only)
        preprocess_num_proc: Number of processes for preprocessing (streaming only)
        preprocess_batch_size: Batch size for streaming preprocessing
        max_shard_size: Maximum shard size (for compatibility)
        cleanup_shards: Whether to auto-cleanup shards (for compatibility)
    
    Returns:
        DataLoader for List[str]/Column[str] inputs
        Tuple[DataLoader, str] for IterableColumn inputs (includes shard_cache_dir for cleanup)
    """
    # Detect data type
    if _is_iterable_column(data):
        # Streaming path
        return _handle_streaming_iterable(
            data, tokenizer, seq_len, batch_size, stride,
            shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,
            min_tail_len=min_tail_len, add_eos=add_eos, seed=seed,
            text_column=text_column,
            shard_cache_dir=shard_cache_dir,
            preprocess_num_proc=preprocess_num_proc,
            preprocess_batch_size=preprocess_batch_size,
            max_shard_size=max_shard_size,
            cleanup_shards=cleanup_shards,
        )
    
    # List[str] or Column[str] path
    # Convert Column to list if needed (for non-iterable columns with __getitem__)
    if hasattr(data, '__getitem__') and hasattr(data, '__len__') and not isinstance(data, list):
        data = list(data)
    
    loader = create_document_loader(
        data, tokenizer, seq_len, batch_size,
        shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,
        min_tail_len=min_tail_len, add_eos=add_eos, lazy=lazy, stride=stride,
    )
    
    return loader
  
