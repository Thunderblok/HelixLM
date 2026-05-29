#!/usr/bin/env python3
"""
Debug script to compare chunking between DocumentAwareDataset and HelixIterableDataset
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helix_lm import HelixTokenizer
from helix_lm.dataset import DocumentAwareDataset, HelixIterableDataset

# Test parameters
SEQ_LEN = 32
STRIDE = 32  # Same as SEQ_LEN
MIN_TAIL_LEN = 1

# Create sample documents with known token counts
test_texts = [
    "Once upon a time there was a little cat who liked to play in the garden every day.",
    "The quick brown fox jumps over the lazy dog. This is a test sentence for chunking analysis.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Short.",
]

# Add more varied length texts
import random
random.seed(42)

# Generate random texts of different lengths
long_text = " ".join([random.choice(["quick", "brown", "fox", "jumps", "over", "lazy", "dog", "cat", "dogs", "cats"]) for _ in range(200)])
medium_text = " ".join([random.choice(["quick", "brown", "fox", "jumps"]) for _ in range(50)])
short_text = "Hello world this is a test"

test_texts = [long_text, medium_text, short_text] + test_texts

print("=" * 70)
print("Chunking Debug Analysis")
print("=" * 70)

tokenizer = HelixTokenizer("gpt2")

# Analyze each text
for i, text in enumerate(test_texts):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_count = len(tokens)
    print(f"\nText {i}: {token_count} tokens")
    
    # Calculate how many chunks would be produced
    if token_count >= SEQ_LEN:
        starts = list(range(0, token_count - SEQ_LEN + 1, STRIDE))
        num_chunks = len(starts)
        last_covered_end = starts[-1] + SEQ_LEN if starts else 0
        remainder_len = token_count - last_covered_end
        
        print(f"  starts: {starts[:5]}{'...' if len(starts) > 5 else ''} (count: {len(starts)})")
        print(f"  last_covered_end: {last_covered_end}")
        print(f"  remainder_len: {remainder_len}")
        
        if remainder_len >= MIN_TAIL_LEN:
            num_chunks += 1
            print(f"  +1 tail chunk")
        
        print(f"  Expected total chunks: {num_chunks}")
    else:
        print(f"  Short document -> 1 padded chunk")

print("\n" + "=" * 70)
print("Testing DocumentAwareDataset")
print("=" * 70)

doc_ds = DocumentAwareDataset(
    test_texts, tokenizer, SEQ_LEN, min_tail_len=MIN_TAIL_LEN, add_eos=True, lazy=True, stride=STRIDE
)

print(f"Total samples from DocumentAwareDataset: {len(doc_ds)}")

print("\n" + "=" * 70)
print("Testing HelixIterableDataset")
print("=" * 70)

class ListIterableDataset:
    def __init__(self, texts, epoch=0):
        self.texts = texts
        self._epoch = epoch
        
    def __iter__(self):
        for text in self.texts:
            yield {"text": text}
            
    def set_epoch(self, epoch):
        self._epoch = epoch

list_iterable = ListIterableDataset(test_texts)
stream_ds = HelixIterableDataset(
    list_iterable, tokenizer, SEQ_LEN, stride=STRIDE, 
    min_tail_len=MIN_TAIL_LEN, add_eos=True, shuffle_buffer_size=0
)

stream_samples = list(stream_ds)
print(f"Total samples from HelixIterableDataset: {len(stream_samples)}")

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"DocumentAwareDataset: {len(doc_ds)} samples")
print(f"HelixIterableDataset: {len(stream_samples)} samples")
print(f"Match: {len(doc_ds) == len(stream_samples)}")

if len(doc_ds) != len(stream_samples):
    print(f"\nDISCREPANCY FOUND!")
    print(f"Difference: {len(stream_samples) - len(doc_ds)} samples")
