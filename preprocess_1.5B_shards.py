#!/usr/bin/env python3
"""
Preprocess the 1.5B token dataset into sharded, pre-tokenized files.

This solves the CPU bottleneck by doing tokenization ONCE in parallel,
then training reads pre-tokenized tensors from disk (fast IO).

Usage:
    python preprocess_1.5B_shards.py \
        --dataset david-thrower/HelixLM-medium-1500.0Mt-2988750pt-20260528 \
        --output_dir ./preprocessed_1.5B \
        --seq_len 512 \
        --batch_size 16 \
        --num_proc 8

Then in training:
    from helix_lm.dataset import create_helix_prechunked_loader
    train_loader = create_helix_prechunked_loader(
        "./preprocessed_1.5B/train",
        tokenizer,
        seq_len=512,
        batch_size=16,
        shuffle=True,
    )
"""
import os
import sys
import argparse
import math
from pathlib import Path

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm

sys.path.insert(0, "/home/ubuntu/streaming-data-tests-copy/HelixLM")
from helix_lm.dataset import HelixPrechunkedDataset


def preprocess_split(
    hf_dataset_split,
    tokenizer,
    output_dir: str,
    seq_len: int = 512,
    stride: int = None,
    min_tail_len: int = 1,
    num_proc: int = 4,
    batch_size: int = 1000,
    max_shard_size: str = "500MB",
):
    """
    Preprocess a dataset split and save to disk as sharded pre-tokenized data.
    
    Args:
        hf_dataset_split: HF Dataset (e.g., ds['train'])
        tokenizer: Tokenizer instance
        output_dir: Directory to save preprocessed shards
        seq_len: Sequence length
        stride: Stride for chunking (default: seq_len = no overlap)
        min_tail_len: Minimum document length to keep
        num_proc: Number of parallel processes for tokenization
        batch_size: Batch size for Dataset.map()
        max_shard_size: Max shard file size
    """
    print(f"Preprocessing {len(hf_dataset_split):,} documents to {output_dir}")
    print(f"  seq_len={seq_len}, stride={stride or seq_len}, num_proc={num_proc}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use HelixPrechunkedDataset.preprocess with output_dir to save
    dataset = HelixPrechunkedDataset.preprocess(
        hf_dataset=hf_dataset_split,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_column="text",
        stride=stride,
        min_tail_len=min_tail_len,
        add_eos=True,
        output_dir=output_dir,
        num_proc=num_proc,
        batch_size=batch_size,
    )
    
    total_chunks = len(dataset)
    print(f"\nSaved {total_chunks:,} chunks to {output_dir}")
    
    # Calculate expected batches (for drop_last=True)
    # This is approximate - actual batches depend on batch_size in DataLoader
    print(f"  Approximate batches at batch_size=X: {total_chunks} // X")
    
    return total_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset into sharded pre-tokenized files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="david-thrower/HelixLM-medium-1500.0Mt-2988750pt-20260528",
        help="Dataset name on HuggingFace Hub"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./preprocessed_1.5B",
        help="Directory to save preprocessed shards"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Sequence length for chunks"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for chunking (default: seq_len = no overlap)"
    )
    parser.add_argument(
        "--min_tail_len",
        type=int,
        default=1,
        help="Minimum document length to keep"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Number of parallel processes for tokenization"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for Dataset.map()"
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="500MB",
        help="Max shard file size (e.g., '500MB', '1GB')"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer name or path"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HelixLM Dataset Preprocessing")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Seq len: {args.seq_len}")
    print(f"Stride: {args.stride or args.seq_len}")
    print(f"Parallel processes: {args.num_proc}")
    print("=" * 70)
    
    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {len(tokenizer):,}")
    
    # Load dataset (non-streaming for parallel preprocessing)
    print("\n[2/4] Loading dataset (this may take a while for large datasets)...")
    ds = load_dataset(args.dataset, streaming=False)
    print(f"  Available splits: {list(ds.keys())}")
    
    # Process train split
    if "pretrain_train" in ds:
        train_split = "pretrain_train"
    elif "train" in ds:
        train_split = "train"
    else:
        train_split = list(ds.keys())[0]
    
    print(f"\n[3/4] Processing train split: {train_split}")
    train_chunks = preprocess_split(
        ds[train_split],
        tokenizer,
        os.path.join(args.output_dir, "train"),
        seq_len=args.seq_len,
        stride=args.stride,
        min_tail_len=args.min_tail_len,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        max_shard_size=args.max_shard_size,
    )
    
    # Process validation split
    val_split = None
    if "pretrain_val" in ds:
        val_split = "pretrain_val"
    elif "validation" in ds:
        val_split = "validation"
    elif "val" in ds:
        val_split = "val"
    
    if val_split:
        print(f"\n[4/4] Processing validation split: {val_split}")
        val_chunks = preprocess_split(
            ds[val_split],
            tokenizer,
            os.path.join(args.output_dir, "val"),
            seq_len=args.seq_len,
            stride=args.stride,
            min_tail_len=args.min_tail_len,
            num_proc=max(1, args.num_proc // 2),
            batch_size=args.batch_size,
            max_shard_size=args.max_shard_size,
        )
    else:
        val_chunks = 0
    
    # Write metadata
    print("\n" + "=" * 70)
    print("Preprocessing Complete!")
    print("=" * 70)
    print(f"Train chunks: {train_chunks:,}")
    if val_chunks:
        print(f"Val chunks: {val_chunks:,}")
    print(f"\nOutput directory: {args.output_dir}")
    
    # Generate example code for training script
    print("\n" + "-" * 70)
    print("Example usage in training script:")
    print("-" * 70)
    print(f'''
from helix_lm.dataset import create_helix_prechunked_loader

# This is now FAST - just reads pre-tokenized tensors from disk
train_loader = create_helix_prechunked_loader(
    "{args.output_dir}/train",
    tokenizer,
    seq_len={args.seq_len},
    batch_size=YOUR_BATCH_SIZE,
    shuffle=True,
)
val_loader = create_helix_prechunked_loader(
    "{args.output_dir}/val",
    tokenizer,
    seq_len={args.seq_len},
    batch_size=YOUR_BATCH_SIZE,
    shuffle=False,
)

# Exact counts are known:
print(f"Train batches: {{len(train_loader)}}")
''')
    print("-" * 70)
    
    # Print batch sizes table
    print("\nExpected batch counts at different batch sizes:")
    print(f"{'Batch Size':<12} {'Train Batches':>15} {'Val Batches':>15}")
    print("-" * 45)
    for bs in [8, 16, 32, 64]:
        train_batches = train_chunks // bs
        val_batches = val_chunks // bs if val_chunks > 0 else 0
        marker = "*" if bs == 16 else " "
        print(f"{marker}{bs:<11} {train_batches:>15,} {val_batches:>15,}")
    print("\n* = default in your ablations config")


if __name__ == "__main__":
    main()
