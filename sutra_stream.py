#!/usr/bin/env python3
"""Deterministic, checkpointable packing for the pinned Sutra stream."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator, Protocol

import torch


class TokenizerLike(Protocol):
    eos_token_id: int

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class SutraStreamOffset:
    """Exact next unread token within the exact ordered dataset revision."""

    row_index: int = 0
    token_offset: int = 0
    raw_utf8_bytes_read: int = 0
    tokens_emitted: int = 0
    causal_targets_emitted: int = 0
    sequences_emitted: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def iter_packed_sequences(
    rows_from_offset: Iterable[dict[str, Any]],
    tokenizer: TokenizerLike,
    *,
    seq_len: int,
    start: SutraStreamOffset = SutraStreamOffset(),
) -> Iterator[tuple[torch.Tensor, SutraStreamOffset]]:
    """Pack a continuous token stream and yield only resumable boundaries.

    `rows_from_offset` must begin at `start.row_index`.  A checkpoint is emitted
    only with a full sequence, so resumption needs only the row and token offset;
    there is no hidden partial buffer to reconstruct.
    """
    if seq_len <= 1:
        raise ValueError("seq_len must be greater than one")
    if start.row_index < 0 or start.token_offset < 0:
        raise ValueError("stream offsets cannot be negative")

    buffer: list[int] = []
    row_index = start.row_index
    token_offset = start.token_offset
    raw_bytes = start.raw_utf8_bytes_read
    tokens_emitted = start.tokens_emitted
    causal_targets = start.causal_targets_emitted
    sequences = start.sequences_emitted

    for row_position, row in enumerate(rows_from_offset):
        text = row.get("text") if isinstance(row, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"dataset row {row_index} has no nonempty text")
        encoded = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        if row_position == 0 and token_offset > len(encoded):
            raise RuntimeError(
                f"token offset {token_offset} exceeds encoded row length {len(encoded)}"
            )
        if token_offset == 0:
            raw_bytes += len(text.encode("utf-8"))

        while token_offset < len(encoded):
            take = min(seq_len - len(buffer), len(encoded) - token_offset)
            buffer.extend(encoded[token_offset : token_offset + take])
            token_offset += take
            if len(buffer) != seq_len:
                continue
            tokens_emitted += seq_len
            causal_targets += seq_len - 1
            sequences += 1
            yield torch.tensor(buffer, dtype=torch.int64), SutraStreamOffset(
                row_index=row_index,
                token_offset=token_offset,
                raw_utf8_bytes_read=raw_bytes,
                tokens_emitted=tokens_emitted,
                causal_targets_emitted=causal_targets,
                sequences_emitted=sequences,
            )
            buffer.clear()

        row_index += 1
        token_offset = 0


def resume_rows(dataset: Any, offset: SutraStreamOffset) -> Any:
    """Position a Hugging Face iterable at the exact checkpoint row."""
    if not hasattr(dataset, "skip"):
        raise TypeError("streaming dataset must implement skip(row_count)")
    return dataset.skip(offset.row_index)
