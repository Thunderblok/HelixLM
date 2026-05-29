"""
HelixLM Dataset with rolling text chunking and natural stop detection.

This module provides THREE dataset strategies for different scales:

1. **Map-style datasets** (DocumentAwareDataset, HelixDataset):
   Best for datasets that fit in memory. Index-based random access.
   Backward compatible — all existing code continues to work unchanged.

2. **Streaming iterable dataset** (HelixIterableDataset):
   Best for 2B+ token corpora. Uses HF IterableDataset for true streaming
   with on-the-fly tokenization/chunking. Never materializes the full corpus.
   Trainer detects iterable datasets and skips length-based scheduler setup.

3. **Pre-chunked map dataset** (HelixPrechunkedDataset):
   Best for fast repeatable training. Pre-chunks once with Dataset.map()
   and saves Arrow-backed dataset to disk. Fast random access.

Key fixes preserved in ALL paths:
  * Exact pad_len tracked in every chunk tuple. NEVER scans backwards for
    pad_token_id, so GPT-2 (pad_id == eos_id) cannot accidentally mask real EOS.
  * Optional within-document overlap (stride) with overlap masked in labels.
  * No cross-document boundaries are ever crossed.
  * is_natural_stop distinguishes true document ends from artificial slices.
  * Recurrence-safety: document boundaries preserve state reset semantics.

Compatible with HF transformers v5.8.1 Trainer — both map and iterable modes.
"""
import random
import math
from typing import List, Optional, Iterator, Dict, Any, Union, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset as TorchIterableDataset
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
# EXISTING MAP-STYLE DATASETS (backward compatible — unchanged semantics)
# ═══════════════════════════════════════════════════════════════════════════

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
        # P1 FIX: Build mask from exact pad_len, NOT from pad_token_id comparison.
        # GPT-2/Qwen set pad_id == eos_id; comparing token values masks real EOS.
        pad_len = sum(1 for tok in reversed(labels_t.tolist()) if tok == -100)
        attention_mask = torch.cat([
            torch.ones(self.seq_len - pad_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long),
        ])
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

        # P1 FIX: Build mask from exact pad_len, NOT from pad_id comparison.
        attention_mask = torch.cat([
            torch.ones(self.seq_len - pad_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long),
        ])
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
        if n >= seq_len and (n - self.stride) % self.stride != 0:
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

    NOTE: For large corpora (2B+ tokens), consider using HelixIterableDataset
    or HelixPrechunkedDataset instead to avoid materializing all texts in memory.
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


# ═══════════════════════════════════════════════════════════════════════════
# NEW: STREAMING ITERABLE DATASET (for 2B+ token corpora)
# ═══════════════════════════════════════════════════════════════════════════

class HelixIterableDataset(TorchIterableDataset):
    """
    Streaming iterable dataset for HelixLM — true O(1) memory scaling.

    Tokenizes and chunks documents ON-THE-FLY during iteration. Never
    materializes the full corpus. Compatible with HF transformers v5.8.1
    Trainer which detects the lack of __len__ and iterates directly.

    Design:
      - Wraps an HF IterableDataset stream (from load_dataset(..., streaming=True))
      - Per-document tokenization and chunk emission in __iter__
      - Preserves ALL HelixLM invariants:
          * is_natural_stop per document boundary
          * -100 masking at padding positions (from exact pad_len)
          * attention_mask from pad_len (NOT pad_token_id comparison)
          * Overlap masking when stride < seq_len
          * No cross-document boundary crossings
      - set_epoch() for reshuffling between epochs

    Usage with Trainer:
        ds = load_dataset("your_dataset", split="train", streaming=True)
        train_ds = HelixIterableDataset(
            ds, tokenizer, seq_len=512, text_column="text"
        )
        # Trainer detects iterable, skips len()-based sampler setup
        trainer = Trainer(model, cfg, train_dataset=train_ds, ...)

    Args:
        hf_iterable: HF IterableDataset or any iterable yielding dicts with text_column
        tokenizer: Tokenizer with encode() and pad_token_id/eos_token_id attributes
        seq_len: Target sequence length for all emitted chunks
        text_column: Column name containing document text (default: "text")
        stride: Chunk stride. stride < seq_len enables overlap (default: seq_len)
        min_tail_len: Minimum document length in tokens to keep (default: 1)
        add_eos: Whether to append EOS token to each document (default: True)
        shuffle_buffer_size: If > 0, uses reservoir shuffle with this buffer size
        seed: Random seed for shuffle
    """

    def __init__(
        self,
        hf_iterable,
        tokenizer,
        seq_len: int = 2048,
        text_column: str = "text",
        stride: Optional[int] = None,
        min_tail_len: int = 1,
        add_eos: bool = True,
        shuffle_buffer_size: int = 10_000,
        seed: int = 42,
    ):
        self.hf_iterable = hf_iterable
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_column = text_column
        self.stride = stride if stride is not None else seq_len
        if not (1 <= self.stride <= self.seq_len):
            raise ValueError(f"stride must be in [1, seq_len], got {self.stride}")
        self.min_tail_len = min_tail_len
        self.add_eos = add_eos
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self._epoch = 0

        # Robust pad_id fallback
        self.pad_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_id is None:
            self.pad_id = getattr(tokenizer, "eos_token_id", 0)
        self.eos_id = getattr(tokenizer, "eos_token_id", None)

    def set_epoch(self, epoch: int):
        """Set epoch for reshuffling. Trainer calls this between epochs."""
        self._epoch = epoch

    def _tokenize_doc(self, text: str) -> Optional[List[int]]:
        """Tokenize a single document. Returns None if too short/empty."""
        text = text.strip()
        if not text:
            return None
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < self.min_tail_len:
            return None
        if self.add_eos and self.eos_id is not None:
            ids.append(self.eos_id)
        return ids

    def _chunk_doc(self, ids: List[int]) -> Iterator[Tuple[List[int], bool, int, int]]:
        """
        Emit fixed-length chunks from a single document's token ids.
        Yields tuples: (token_ids, is_natural_stop, pad_len, overlap_mask)
        """
        length = len(ids)
        stride = self.stride
        seq_len = self.seq_len

        if length >= seq_len:
            # Sliding-window chunks fully inside the document
            starts = list(range(0, length - seq_len + 1, stride))
            for i, start in enumerate(starts):
                chunk = ids[start:start + seq_len]
                is_last_start = (i == len(starts) - 1)
                reaches_end = (start + seq_len == length)

                # Natural stop logic
                if not is_last_start:
                    is_natural = False
                else:
                    tail_len = length - (start + seq_len)
                    has_tail = tail_len >= self.min_tail_len
                    is_natural = reaches_end or (not has_tail)

                overlap_mask = 0
                if start > 0 and stride < seq_len:
                    overlap_mask = seq_len - stride

                yield (chunk, is_natural, 0, overlap_mask)

            # Tail: tokens after the last sliding chunk
            last_covered_end = starts[-1] + seq_len if starts else 0
            remainder_len = length - last_covered_end

            if remainder_len >= self.min_tail_len:
                tail = ids[last_covered_end:]
                pad_len = seq_len - remainder_len
                tail = tail + [self.pad_id] * pad_len
                yield (tail, True, pad_len, 0)
            elif remainder_len > 0 and starts:
                # Tail too short: mark last sliding chunk as doc end
                # We need to mutate the last yielded chunk — re-yield corrected version
                pass  # The last sliding chunk already has is_natural computed above
        else:
            # Short document: pad to seq_len
            pad_len = seq_len - length
            chunk = ids + [self.pad_id] * pad_len
            yield (chunk, True, pad_len, 0)

    def _make_sample(
        self,
        chunk: List[int],
        is_natural: bool,
        pad_len: int,
        overlap_mask: int,
    ) -> Dict[str, torch.Tensor]:
        """Convert a chunk tuple to a training sample with proper masking."""
        x = torch.tensor(chunk, dtype=torch.long)
        labels = x.clone()

        # Mask overlapping head (stride < seq_len)
        if overlap_mask > 0:
            labels[:overlap_mask] = -100

        # Mask trailing padding (robust to pad_id == eos_id)
        if pad_len > 0:
            labels[-pad_len:] = -100

        # Build attention_mask from exact pad_len (NOT pad_token_id comparison)
        attention_mask = torch.cat([
            torch.ones(self.seq_len - pad_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long),
        ])

        return {
            "input_ids": x,
            "labels": labels,
            "attention_mask": attention_mask,
            "is_natural_stop": torch.tensor(is_natural, dtype=torch.bool),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over the dataset, yielding training samples on-the-fly.

        With shuffle_buffer_size > 0, uses reservoir shuffle for randomness.
        set_epoch() controls the shuffle seed between epochs.
        """
        rng = random.Random(self.seed + self._epoch)

        # Prepare the underlying iterable
        iterable = self.hf_iterable
        if hasattr(iterable, "set_epoch"):
            iterable.set_epoch(self._epoch)
        elif hasattr(iterable, "shard") and callable(getattr(iterable, "shard", None)):
            # HF IterableDataset: no set_epoch needed, shuffle is via shuffle()
            pass

        if self.shuffle_buffer_size > 0:
            # Reservoir shuffle: fill buffer, then yield random elements
            buffer = []
            for example in iterable:
                text = example.get(self.text_column, "") if isinstance(example, dict) else str(example)
                ids = self._tokenize_doc(text)
                if ids is None:
                    continue

                for chunk_tuple in self._chunk_doc(ids):
                    sample = self._make_sample(*chunk_tuple)
                    if len(buffer) < self.shuffle_buffer_size:
                        buffer.append(sample)
                    else:
                        # Replace random element with probability buffer_size / (buffer_size + 1)
                        idx = rng.randint(0, len(buffer) - 1)
                        yield buffer[idx]
                        buffer[idx] = sample

            # Drain remaining buffer in random order
            rng.shuffle(buffer)
            for sample in buffer:
                yield sample
        else:
            # No shuffle: deterministic iteration
            for example in iterable:
                text = example.get(self.text_column, "") if isinstance(example, dict) else str(example)
                ids = self._tokenize_doc(text)
                if ids is None:
                    continue

                for chunk_tuple in self._chunk_doc(ids):
                    yield self._make_sample(*chunk_tuple)

    def __len__(self):
        """
        Iterable datasets should NOT implement __len__.
        HF Trainer detects absence of __len__ and treats this as iterable
        (no random sampler, no length-based step estimation).
        """
        raise TypeError(
            "HelixIterableDataset has no len(). It is a streaming dataset. "
            "Use warmup_ratio instead of warmup_steps, or estimate steps from "
            "total_tokens / (batch_size * seq_len)."
        )


# ═══════════════════════════════════════════════════════════════════════════
# NEW: PRE-CHUNKED MAP DATASET (for fast repeatable training)
# ═══════════════════════════════════════════════════════════════════════════

class HelixPrechunkedDataset(Dataset):
    """
    Map-style dataset from pre-chunked HF Dataset.

    Preprocesses once with Dataset.map(batched=True, num_proc=...) so chunking
    happens offline, not during training. The resulting Arrow-backed dataset
    is memory-mapped and supports fast random access.

    Best for: datasets that fit on disk but are too slow to tokenize on-the-fly.
    Use this when profiling shows streaming tokenization can't keep GPUs fed.

    Usage:
        # One-time preprocessing
        ds = load_dataset("your_dataset", split="train")
        chunked = HelixPrechunkedDataset.preprocess(
            ds, tokenizer, seq_len=512, output_dir="./prechunked", num_proc=8
        )

        # Training (fast random access)
        train_ds = HelixPrechunkedDataset.from_disk("./prechunked")
        trainer = Trainer(model, cfg, train_dataset=train_ds, ...)

    Args:
        chunked_dataset: HF Dataset with columns:
            input_ids, labels, attention_mask, is_natural_stop
    """

    def __init__(self, chunked_dataset):
        """
        Args:
            chunked_dataset: HF Dataset with pre-computed columns:
                input_ids (List[int]), labels (List[int]),
                attention_mask (List[int]), is_natural_stop (bool)
        """
        super().__init__()
        self.dataset = chunked_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.dataset[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "is_natural_stop": torch.tensor(row["is_natural_stop"], dtype=torch.bool),
        }

    @classmethod
    def preprocess(
        cls,
        hf_dataset,
        tokenizer,
        seq_len: int = 2048,
        text_column: str = "text",
        stride: Optional[int] = None,
        min_tail_len: int = 1,
        add_eos: bool = True,
        output_dir: Optional[str] = None,
        num_proc: int = 4,
        batch_size: int = 1000,
        split: Optional[str] = None,
    ):
        """
        Pre-chunk an HF Dataset and optionally save to disk.

        Args:
            hf_dataset: HF Dataset (map-style) or dataset name string
            tokenizer: Tokenizer instance
            seq_len: Target sequence length
            text_column: Column name for text
            stride: Chunk stride (default: seq_len = no overlap)
            min_tail_len: Minimum document token count to keep
            add_eos: Append EOS to each document
            output_dir: If provided, save the pre-chunked dataset to this directory
            num_proc: Number of processes for parallel map
            batch_size: Batch size for mapping
            split: If hf_dataset is a string, which split to load

        Returns:
            HelixPrechunkedDataset instance
        """
        from datasets import Dataset, load_dataset

        if isinstance(hf_dataset, str):
            hf_dataset = load_dataset(hf_dataset, split=split or "train")
        elif hasattr(hf_dataset, "keys") and "train" in hf_dataset:
            hf_dataset = hf_dataset["train"]

        stride = stride if stride is not None else seq_len
        pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
        eos_id = getattr(tokenizer, "eos_token_id", None)

        def _chunk_batch(batch):
            """Batched chunking function for Dataset.map()."""
            all_input_ids = []
            all_labels = []
            all_attention_mask = []
            all_is_natural_stop = []

            texts = batch[text_column]
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

                        labels = list(chunk)
                        if overlap_mask > 0:
                            labels[:overlap_mask] = [-100] * overlap_mask

                        pad_len = 0
                        attn_mask = [1] * seq_len

                        all_input_ids.append(chunk)
                        all_labels.append(labels)
                        all_attention_mask.append(attn_mask)
                        all_is_natural_stop.append(is_natural)

                    # Tail
                    last_covered_end = starts[-1] + seq_len if starts else 0
                    remainder_len = length - last_covered_end
                    if remainder_len >= min_tail_len:
                        tail = ids[last_covered_end:]
                        pad_len = seq_len - remainder_len
                        tail = tail + [pad_id] * pad_len
                        labels = list(tail)
                        labels[-pad_len:] = [-100] * pad_len if pad_len > 0 else []
                        attn_mask = [1] * (seq_len - pad_len) + [0] * pad_len

                        all_input_ids.append(tail)
                        all_labels.append(labels)
                        all_attention_mask.append(attn_mask)
                        all_is_natural_stop.append(True)

                else:
                    # Short document
                    pad_len = seq_len - length
                    chunk = ids + [pad_id] * pad_len
                    labels = list(chunk)
                    labels[-pad_len:] = [-100] * pad_len if pad_len > 0 else []
                    attn_mask = [1] * (seq_len - pad_len) + [0] * pad_len

                    all_input_ids.append(chunk)
                    all_labels.append(labels)
                    all_attention_mask.append(attn_mask)
                    all_is_natural_stop.append(True)

            return {
                "input_ids": all_input_ids,
                "labels": all_labels,
                "attention_mask": all_attention_mask,
                "is_natural_stop": all_is_natural_stop,
            }

        # Determine columns to remove — keep only text_column for processing
        original_columns = list(hf_dataset.column_names)
        remove_columns = [c for c in original_columns if c != text_column]

        chunked = hf_dataset.map(
            _chunk_batch,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=remove_columns,
            desc=f"Pre-chunking (seq_len={seq_len}, stride={stride})",
        )

        # Flatten any nested batch structure
        chunked = chunked.flatten_indices()

        if output_dir is not None:
            chunked.save_to_disk(output_dir)
            print(f"Pre-chunked dataset saved to {output_dir}")
            print(f"Total chunks: {len(chunked):,}")

        return cls(chunked)

    @classmethod
    def from_disk(cls, path: str):
        """Load a pre-chunked dataset from disk."""
        from datasets import load_from_disk
        dataset = load_from_disk(path)
        return cls(dataset)

    def save_to_disk(self, path: str):
        """Save the underlying dataset to disk."""
        self.dataset.save_to_disk(path)


# ═══════════════════════════════════════════════════════════════════════════
# COLLATOR
# ═══════════════════════════════════════════════════════════════════════════

def helix_data_collator(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for all HelixLM dataset types.

    Stacks tensors for input_ids, labels, attention_mask, is_natural_stop.
    Works with both map-style and iterable datasets.
    """
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


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

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
) -> DataLoader:
    """Create a DataLoader using HelixDataset (rolling chunking)."""
    dataset = HelixDataset(texts, tokenizer, seq_len, stride, lazy=lazy, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=helix_data_collator,
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
) -> DataLoader:
    """
    Create a DataLoader using DocumentAwareDataset (no boundary crossings).

    Args:
        stride: If < seq_len, enables within-document overlap (default: seq_len).
                This restores more optimizer steps per epoch without ever
                crossing document boundaries.
    """
    ds = DocumentAwareDataset(
        texts, tokenizer, seq_len,
        min_tail_len=min_tail_len, add_eos=add_eos, lazy=lazy, stride=stride,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=helix_data_collator,
        num_workers=num_workers,
        drop_last=drop_last,
    )


class ListIterableDataset:
    """
    Wrap a Python list as an HF-style iterable dataset.
    
    Provides set_epoch() for reshuffling support between epochs.
    Compatible with HelixIterableDataset for seamless streaming.
    
    Args:
        data: List of items to iterate over
        epoch: Starting epoch number
        shuffle: Whether to shuffle data on each epoch change
        seed: Random seed for reproducible shuffling
        text_column: Key name when yielding dicts (default: "text")
    """
    def __init__(
        self,
        data: List[Any],
        epoch: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        text_column: str = "text",
    ):
        self.data = data
        self._epoch = epoch
        self.shuffle = shuffle
        self.seed = seed
        self.text_column = text_column
    
    def set_epoch(self, epoch: int):
        """Set epoch for reshuffling. Called by Trainer between epochs."""
        self._epoch = epoch
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over data, shuffling if enabled."""
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            indices = list(range(len(self.data)))
            rng.shuffle(indices)
            data_iter = (self.data[i] for i in indices)
        else:
            data_iter = iter(self.data)
        
        for item in data_iter:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, str):
                yield {self.text_column: item}
            else:
                yield {self.text_column: str(item)}
    
    def __len__(self) -> int:
        return len(self.data)


def create_helix_streaming_loader(
    hf_iterable,
    tokenizer,
    seq_len: int = 2048,
    batch_size: int = 8,
    text_column: str = "text",
    stride: Optional[int] = None,
    min_tail_len: int = 1,
    add_eos: bool = True,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
    drop_last: bool = True,
) -> "HelixIterableDataset":
    """
    Create a streaming HelixIterableDataset.

    Returns the dataset directly (not a DataLoader). HF Trainer handles
    iteration for iterable datasets. Use with Trainer's train_dataset arg.

    For PyTorch DataLoader usage, note that IterableDataset does not support
    shuffle=True (shuffling is handled internally via shuffle_buffer_size).

    Args:
        hf_iterable: HF IterableDataset or any iterable
        tokenizer: Tokenizer instance
        seq_len: Target sequence length
        batch_size: Batch size (stored for reference; Trainer uses it)
        text_column: Document text column name
        stride: Chunk stride (default: seq_len)
        min_tail_len: Minimum document length to keep (default: 1)
        add_eos: Append EOS to documents (default: True)
        shuffle_buffer_size: Reservoir shuffle buffer (default: 10_000)
        seed: Random seed for shuffle
        drop_last: Whether to drop incomplete final batch

    Returns:
        HelixIterableDataset instance
    """
    return HelixIterableDataset(
        hf_iterable=hf_iterable,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_column=text_column,
        stride=stride,
        min_tail_len=min_tail_len,
        add_eos=add_eos,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    )


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
    text_column: str = "text",
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
) -> DataLoader:
    """
    Unified data loader factory that accepts List[str] OR any iterable/streaming dataset.
    
    This is the recommended API entry point. It automatically detects the data type
    and creates the appropriate dataset + DataLoader:
    
    - List[str] (map-style): Uses DocumentAwareDataset with PyTorch shuffle
    - Iterable/streaming: Uses HelixIterableDataset with reservoir shuffle
    
    Both paths support shuffling once per epoch:
    - Map-style: DataLoader shuffle + Trainer's set_epoch() call
    - Iterable: HelixIterableDataset's internal shuffle_buffer + set_epoch()
    
    Args:
        data: Either a List[str] of documents, or an iterable/streaming dataset
              (HF IterableDataset, HelixIterableDataset, etc.)
        tokenizer: Tokenizer instance with encode() and pad/eos token attributes
        seq_len: Target sequence length for chunks
        batch_size: Batch size for DataLoader
        stride: Chunk stride (default: seq_len = no overlap)
        shuffle: Whether to shuffle data each epoch (default: True)
        drop_last: Whether to drop incomplete final batch (default: True)
        num_workers: DataLoader worker processes (default: 0)
        min_tail_len: Minimum document length to keep (default: 1)
        add_eos: Append EOS to documents (default: True)
        lazy: Whether to use lazy tokenization for List[str] (default: True)
        text_column: Column name for text in streaming datasets (default: "text")
        shuffle_buffer_size: Reservoir shuffle buffer for streaming (default: 10_000)
        seed: Random seed for reproducible shuffle
        
    Returns:
        DataLoader ready for Trainer consumption
        
    Example:
        # List[str] path
        texts = ["doc1...", "doc2...", ...]
        loader = create_unified_data_loader(
            texts, tokenizer, seq_len=512, batch_size=8, shuffle=True
        )
        
        # Streaming path
        hf_ds = load_dataset("dataset", split="train", streaming=True)
        loader = create_unified_data_loader(
            hf_ds, tokenizer, seq_len=512, batch_size=8, shuffle=True
        )
    """
    # Detect if data is a List[str]
    is_list_of_strings = (
        isinstance(data, list) and 
        len(data) > 0 and 
        isinstance(data[0], str)
    )
    
    if is_list_of_strings:
        # Map-style path: DocumentAwareDataset
        return create_document_loader(
            texts=data,
            tokenizer=tokenizer,
            seq_len=seq_len,
            batch_size=batch_size,
            stride=stride,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            min_tail_len=min_tail_len,
            add_eos=add_eos,
            lazy=lazy,
        )
    
    # Iterable/streaming path
    # Check if it already has our streaming interface
    is_our_iterable = hasattr(data, 'set_epoch') and hasattr(data, '__iter__')
    is_hf_iterable = hasattr(data, '__iter__') and not hasattr(data, '__getitem__')
    
    # For List[str] passed as iterable, wrap it
    if isinstance(data, list):
        data = ListIterableDataset(
            data=data,
            epoch=0,
            shuffle=shuffle,
            seed=seed,
            text_column=text_column,
        )
    elif not is_our_iterable and not is_hf_iterable:
        # Unknown type, try to treat as iterable
        pass
    
    # Create HelixIterableDataset with shuffle support
    ds = HelixIterableDataset(
        hf_iterable=data,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_column=text_column,
        stride=stride,
        min_tail_len=min_tail_len,
        add_eos=add_eos,
        shuffle_buffer_size=shuffle_buffer_size if shuffle else 0,
        seed=seed,
    )
    
    # Iterable datasets use DataLoader without shuffle (shuffling is internal)
    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=helix_data_collator,
        num_workers=num_workers,
        drop_last=drop_last,
        # shuffle=False for iterable datasets
    )


def create_helix_prechunked_loader(
    hf_dataset,
    tokenizer,
    seq_len: int = 2048,
    batch_size: int = 8,
    text_column: str = "text",
    stride: Optional[int] = None,
    min_tail_len: int = 1,
    add_eos: bool = True,
    output_dir: Optional[str] = None,
    num_proc: int = 4,
    map_batch_size: int = 1000,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    split: Optional[str] = None,
) -> DataLoader:
    """
    Create a DataLoader from a pre-chunked dataset.

    Preprocesses the dataset once with parallel chunking, then returns
    a fast map-style DataLoader.

    Args:
        hf_dataset: HF Dataset (map-style), dataset name string, or path to saved pre-chunked data
        tokenizer: Tokenizer instance
        seq_len: Target sequence length
        batch_size: DataLoader batch size
        text_column: Document text column name
        stride: Chunk stride (default: seq_len)
        min_tail_len: Minimum document length (default: 1)
        add_eos: Append EOS to documents (default: True)
        output_dir: Directory to save/load pre-chunked dataset
        num_proc: Parallel processes for chunking
        map_batch_size: Batch size for Dataset.map()
        shuffle: Whether to shuffle batches
        num_workers: DataLoader worker processes
        drop_last: Whether to drop incomplete final batch
        split: Dataset split if hf_dataset is a string name

    Returns:
        DataLoader with HelixPrechunkedDataset
    """
    # If hf_dataset is a path string that exists, load from disk
    import os
    if isinstance(hf_dataset, str) and os.path.exists(hf_dataset):
        dataset = HelixPrechunkedDataset.from_disk(hf_dataset)
    elif isinstance(hf_dataset, str) and output_dir and os.path.exists(output_dir):
        # Pre-chunked data already exists
        dataset = HelixPrechunkedDataset.from_disk(output_dir)
    else:
        # Preprocess from scratch
        dataset = HelixPrechunkedDataset.preprocess(
            hf_dataset=hf_dataset,
            tokenizer=tokenizer,
            seq_len=seq_len,
            text_column=text_column,
            stride=stride,
            min_tail_len=min_tail_len,
            add_eos=add_eos,
            output_dir=output_dir,
            num_proc=num_proc,
            batch_size=map_batch_size,
            split=split,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=helix_data_collator,
        num_workers=num_workers,
        drop_last=drop_last,
    )


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY: Offline shard preprocessing (two-stage pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_to_shards(
    hf_dataset,
    tokenizer,
    output_dir: str,
    seq_len: int = 2048,
    text_column: str = "text",
    stride: Optional[int] = None,
    min_tail_len: int = 1,
    add_eos: bool = True,
    num_proc: int = 4,
    map_batch_size: int = 1000,
    max_shard_size: str = "500MB",
    split: Optional[str] = None,
):
    """
    Preprocess a dataset into sharded files for streaming during training.

    This implements the two-stage pipeline recommended for 2B+ token corpora:
      1. Pre-chunk offline (expensive tokenization happens once)
      2. Stream shards during training (cheap IO, fast batching)

    Args:
        hf_dataset: HF Dataset or dataset name string
        tokenizer: Tokenizer instance
        output_dir: Directory to write sharded files
        seq_len: Target sequence length
        text_column: Document text column
        stride: Chunk stride
        min_tail_len: Minimum document length
        add_eos: Append EOS
        num_proc: Parallel processes
        map_batch_size: Batch size for mapping
        max_shard_size: Max shard file size (passed to Dataset.save_to_disk)
        split: Dataset split if hf_dataset is a string

    Returns:
        Path to output directory
    """
    from datasets import load_dataset
    import os

    os.makedirs(output_dir, exist_ok=True)

    if isinstance(hf_dataset, str):
        hf_dataset = load_dataset(hf_dataset, split=split or "train")
    elif hasattr(hf_dataset, "keys") and "train" in hf_dataset:
        hf_dataset = hf_dataset["train"]

    # Pre-chunk
    dataset = HelixPrechunkedDataset.preprocess(
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_column=text_column,
        stride=stride,
        min_tail_len=min_tail_len,
        add_eos=add_eos,
        num_proc=num_proc,
        batch_size=map_batch_size,
    )

    # Save as sharded dataset
    dataset.dataset.save_to_disk(output_dir, max_shard_size=max_shard_size)

    total_chunks = len(dataset)
    print(f"Shard preprocessing complete: {total_chunks:,} chunks -> {output_dir}")
    return output_dir
 
