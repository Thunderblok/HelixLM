"""
HelixLM Trainer with gradient accumulation, configurable AMP, and progress bars.

Supports BOTH map-style and iterable datasets:
  - Map-style (DocumentAwareDataset, HelixDataset): index-based, supports len(),
    shuffle, random sampler. Used for datasets that fit in memory.
  - Iterable (HelixIterableDataset): streaming, no len(), iterates directly.
    Used for 2B+ token corpora. Trainer auto-detects and skips len()-based setup.

Key features:
  - Gradient accumulation for effective larger batch sizes
  - Configurable AMP (default: off for stability on small models)
  - NaN/Inf detection and batch skipping
  - Scheduler steps count optimizer steps, not raw batches
  - Uses DocumentAwareDataset (no cross-document boundary crossings)
  - Modern torch.amp API (not deprecated torch.cuda.amp)
  - Live tqdm progress bars with loss, PPL, LR, and throughput metrics
  - Optional train/val DataLoader injection for custom dataset pipelines
  - warmup_ratio support for iterable datasets (avoids needing exact step count)
"""
import os
import math
import time
import warnings
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import HelixConfig
from .hf_model import HelixForCausalLM
from .dataset import create_document_loader


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.1,
):
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (
            1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        )
        return min_lr_ratio + (1.0 - min_lr_ratio) * max(0.0, cosine)

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup_ratio(
    optimizer,
    num_warmup_ratio: float,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.1,
):
    """
    Cosine LR schedule with warmup specified as a ratio of total steps.
    Used when exact step count is known but warmup is easier to specify as ratio.
    """
    num_warmup_steps = max(1, int(num_training_steps * num_warmup_ratio))
    return get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, num_cycles, min_lr_ratio
    )


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss, capping at exp(20) to avoid overflow."""
    return math.exp(min(loss, 20))


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def _is_iterable_dataset(dataset):
    """Check if a dataset is iterable (no random access / len())."""
    # Check for HF IterableDataset
    if hasattr(dataset, "__iter__") and not hasattr(dataset, "__getitem__"):
        return True
    # Check for our HelixIterableDataset (has __getitem__ but raises TypeError on __len__)
    if hasattr(dataset, "__len__"):
        try:
            len(dataset)
        except (TypeError, NotImplementedError):
            return True
    return False


class Trainer:
    """Trainer for HelixLM with gradient accumulation, AMP, and progress bars.

    Supports both map-style and iterable datasets:
      - Map-style: uses len() for scheduler setup, supports shuffle
      - Iterable: auto-detected, iterates directly, uses automatic silent count
                   or warmup_ratio for scheduler setup
    """

    def __init__(
        self,
        model: HelixForCausalLM,
        cfg: HelixConfig,
        train_texts: Optional[List[str]] = None,
        val_texts: Optional[List[str]] = None,
        tokenizer=None,
        output_dir: str = "./checkpoints",
        example_prompts: Optional[List[str]] = None,
        generated_example_length: int = 15,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
        amp_dtype: Optional[str] = None,
        min_tail_len: Optional[int] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True,
        warmup_ratio: Optional[float] = None,
    ):
        """
        Initialize Trainer.

        Args:
            model: HelixForCausalLM instance.
            cfg: HelixConfig with training hyperparameters.
            train_texts: List of training document texts (used if train_loader not provided).
            val_texts: List of validation document texts (used if val_loader not provided).
            tokenizer: Tokenizer instance.
            output_dir: Directory to save checkpoints.
            example_prompts: Prompts for generation samples during training.
            generated_example_length: Number of tokens to generate for samples.
            grad_accum_steps: Gradient accumulation steps (default: 1).
            use_amp: Whether to use torch.amp automatic mixed precision.
            amp_dtype: AMP autocast dtype: "float16" or "bfloat16" (default: "float16").
            min_tail_len: Minimum tail length for DocumentAwareDataset.
            train_loader: Optional custom DataLoader to override built-in dataset creation.
            val_loader: Optional custom DataLoader to override built-in dataset creation.
            verbose: Whether to show tqdm progress bars and print logs.
            warmup_ratio: If provided, overrides cfg.warmup_steps as a ratio of total steps.
                          Useful for iterable datasets where exact step count is unknown.
                          E.g., warmup_ratio=0.1 means 10% of total steps are warmup.
        """
        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.use_amp = use_amp and torch.cuda.is_available()
        _amp_dtype = amp_dtype if amp_dtype is not None else getattr(cfg, "amp_dtype", "float16")
        self.amp_dtype = getattr(torch, _amp_dtype) if isinstance(_amp_dtype, str) else _amp_dtype
        self.verbose = verbose
        self.warmup_ratio = warmup_ratio
        
        # Cache for iterable dataset count
        self._cached_dataset_length: Optional[int] = None

        if example_prompts:
            self.example_prompts = example_prompts
        else:
            self.example_prompts = [
                "In the beginning",
                "And God said",
                "The sky was",
            ]
        self.generated_example_length = generated_example_length

        self.device = self._get_device()
        self.model = self.model.to(self.device)

        # Validate config
        self.validate_config()

        # Data loaders: use injected loaders if provided, otherwise build from texts
        if train_loader is not None:
            self.train_loader = train_loader
        else:
            if train_texts is None:
                raise ValueError("Either train_loader or train_texts must be provided.")
            self.train_loader = create_document_loader(
                train_texts,
                tokenizer,
                cfg.seq_len,
                cfg.batch_size,
                shuffle=True,
                min_tail_len=min_tail_len,
                lazy=True,
            )

        self.val_loader = None
        if val_loader is not None:
            self.val_loader = val_loader
        elif val_texts is not None:
            self.val_loader = create_document_loader(
                val_texts,
                tokenizer,
                cfg.seq_len,
                cfg.batch_size,
                shuffle=False,
                drop_last=False,
                min_tail_len=min_tail_len,
                lazy=True,
            )

        # Detect if train_loader uses an iterable dataset
        inner_ds = getattr(self.train_loader, "dataset", None)
        self._is_iterable = _is_iterable_dataset(inner_ds) if inner_ds is not None else False

        if self._is_iterable and self.verbose:
            print("  [Trainer] Iterable dataset detected — skipping len()-based setup")

        # AdamW with standard betas (0.9, 0.999)
        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler setup
        self._scheduler_warmup = max(1, cfg.warmup_steps // self.grad_accum_steps)
        self._scheduler_cycles = 0.5
        self._scheduler_min_lr = 0.1
        self.scheduler = None

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "perplexity": []}

        # GradScaler for AMP (only if use_amp=True and CUDA available and dtype is float16)
        # BFloat16 does not need/ support GradScaler — it has sufficient range natively.
        self.scaler = None
        if self.use_amp and self.amp_dtype == torch.float16:
            try:
                from torch.amp import GradScaler
                self.scaler = GradScaler("cuda")
            except Exception:
                pass  # scaler stays None, AMP still works without scaling

    def _count_iterable_dataset(self, iterable_ds) -> int:
        """
        Efficiently count batches in an iterable dataset without materializing.
        Uses a counting pass that iterates through the DataLoader to get actual batch count.
        
        Args:
            iterable_ds: DataLoader wrapping an iterable dataset
        
        Returns:
            Total batch count
        """
        if self.verbose:
            print("  [Trainer] Counting iterable dataset batches...")
        
        count = 0
        start_time = time.time()
        
        # Iterate through DataLoader to count actual batches
        # This handles drop_last=True correctly
        for _ in iterable_ds:
            count += 1
        
        elapsed = time.time() - start_time
        if self.verbose:
            print(f"  [Trainer] Counted {count:,} batches in {format_time(elapsed)}")
        
        return count

    def _compute_total_steps(self) -> int:
        """
        Compute total optimizer steps for scheduler setup.
        
        For map-style datasets: uses len(train_loader).
        For iterable datasets: performs a counting pass and caches the result.
        """
        if self._is_iterable:
            # Iterable dataset: count batches once and cache
            if self._cached_dataset_length is None:
                self._cached_dataset_length = self._count_iterable_dataset(self.train_loader)
            
            total_batches = self._cached_dataset_length
            total_steps = max(1, total_batches // self.grad_accum_steps)
            
            # Adjust for multi-epoch training
            total_steps *= self.cfg.epochs
            
            return total_steps
        else:
            # Map-style dataset: len() is cheap and accurate
            steps_per_epoch = math.ceil(len(self.train_loader) / self.grad_accum_steps)
            return steps_per_epoch * self.cfg.epochs

    def _reset_iterable_dataset(self):
        """Reset iterable dataset to prepare for actual training after counting pass."""
        inner_ds = getattr(self.train_loader, "dataset", None)
        if inner_ds is not None and hasattr(inner_ds, "set_epoch"):
            # Reset to epoch 0 for start of training
            inner_ds.set_epoch(0)

    def _get_device(self) -> torch.device:
        """Get device from config."""
        if self.cfg.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.cfg.device)

    def validate_config(self) -> None:
        """Validate training config and emit warnings for suboptimal settings."""
        total_params = getattr(self.model, "count_parameters", lambda: {"total": 0})()["total"]
        use_titans = getattr(self.cfg, "use_titans_memory", False)
        seq_len = getattr(self.cfg, "seq_len", 2048)

        if use_titans and total_params < 50_000_000 and seq_len < 512:
            warnings.warn(
                f"use_titans_memory=True on a small model ({total_params:,} params) "
                f"with seq_len={seq_len} may not provide substantial benefit, "
                f"as Titans state resets per batch at this scale. "
                f"Consider disabling Titans for faster training or increasing seq_len.",
                UserWarning,
                stacklevel=2,
            )

    def _init_scheduler(self, epoch: int):
        """
        Initialize scheduler. Called lazily on first train_epoch() call.

        For map-style datasets: uses len(train_loader) to compute steps.
        For iterable datasets: performs counting pass for accurate step count.
        """
        if self.scheduler is not None:
            return

        # Compute total steps (handles both map-style and iterable datasets)
        total_steps = self._compute_total_steps()
        
        if self._is_iterable:
            # Reset the iterable dataset after counting pass
            self._reset_iterable_dataset()
            
            if self.warmup_ratio is not None:
                # Warmup ratio mode: warmup is % of total steps
                if self.verbose:
                    print(
                        f"  [Trainer] Scheduler: {total_steps:,} total steps, "
                        f"warmup_ratio={self.warmup_ratio}"
                    )
                self.scheduler = get_cosine_schedule_with_warmup_ratio(
                    self.optimizer,
                    num_warmup_ratio=self.warmup_ratio,
                    num_training_steps=total_steps,
                    num_cycles=self._scheduler_cycles,
                    min_lr_ratio=self._scheduler_min_lr,
                )
            else:
                # Step-based warmup with accurate count
                if self.verbose:
                    print(
                        f"  [Trainer] Scheduler: {total_steps:,} total steps, "
                        f"warmup_steps={self._scheduler_warmup}"
                    )
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=self._scheduler_warmup,
                    num_training_steps=total_steps,
                    num_cycles=self._scheduler_cycles,
                    min_lr_ratio=self._scheduler_min_lr,
                )
        else:
            # Map-style dataset
            if self.verbose:
                print(f"  [Trainer] Scheduler: {total_steps:,} total steps")
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self._scheduler_warmup,
                num_training_steps=total_steps,
                num_cycles=self._scheduler_cycles,
                min_lr_ratio=self._scheduler_min_lr,
            )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation and progress bar."""
        self.model.train()
        total_loss = 0.0
        raw_count = 0
        accum_count = 0
        skipped_batches = 0
        epoch_start = time.time()
        tokens_seen = 0

        self.optimizer.zero_grad()

        # Lazily initialize scheduler now that we're about to train.
        self._init_scheduler(epoch)

        # For iterable datasets, set epoch for reshuffling
        inner_ds = getattr(self.train_loader, "dataset", None)
        if inner_ds is not None and hasattr(inner_ds, "set_epoch"):
            inner_ds.set_epoch(epoch)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            unit="batch",
            disable=not self.verbose,
        )

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            tokens_seen += input_ids.numel()

            # Get attention_mask from batch
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Build cca_step from global optimizer step (not batch index)
            cca_step = None
            if getattr(self.cfg, "use_cca", False):
                cca_step = self.global_step

            # Forward pass — autocast whenever AMP is enabled (independent of scaler)
            if self.use_amp:
                with torch.amp.autocast(
                    device_type="cuda", dtype=self.amp_dtype
                ):
                    outputs = self.model(
                        input_ids, labels=labels,
                        attention_mask=attention_mask,
                        cca_step=cca_step,
                    )
                    loss = outputs["loss"]
            else:
                outputs = self.model(
                    input_ids, labels=labels,
                    attention_mask=attention_mask,
                    cca_step=cca_step,
                )
                loss = outputs["loss"]

            # Skip NaN/Inf losses (numerical instability)
            if torch.isnan(loss) or torch.isinf(loss):
                skipped_batches += 1
                if skipped_batches <= 5 and self.verbose:
                    print(
                        f"  WARNING: NaN/Inf loss at batch {batch_idx}. "
                        f"Skipping. (Try disabling AMP: use_amp=False)"
                    )
                continue

            # Scale loss for gradient accumulation
            divisor = 1
            if self.grad_accum_steps > 1:
                # For iterable datasets, we may not know total batches
                # Check if this might be the last batch by testing if the iterator is exhausted
                is_last = False
                if not self._is_iterable:
                    is_last = (batch_idx + 1) == len(self.train_loader)
                if is_last and accum_count < self.grad_accum_steps - 1:
                    divisor = accum_count + 1
                else:
                    divisor = self.grad_accum_steps
                loss = loss / divisor

            # Backward pass — scale only if scaler exists
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1
            total_loss += loss.item() * divisor
            raw_count += 1

            # Optimizer step after accumulation
            is_last = False
            if not self._is_iterable:
                is_last = (batch_idx + 1) == len(self.train_loader)
            # For iterable datasets, step every grad_accum_steps unconditionally
            if accum_count >= self.grad_accum_steps or is_last:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                accum_count = 0
                self.global_step += 1

            # Live progress bar update
            avg = total_loss / max(raw_count, 1)
            lr = self.scheduler.get_last_lr()[0]
            elapsed = time.time() - epoch_start
            tok_per_sec = tokens_seen / max(elapsed, 1e-6)
            pbar.set_postfix({
                "loss": f"{avg:.4f}",
                "ppl": f"{compute_perplexity(avg):.2f}",
                "lr": f"{lr:.2e}",
                "tok/s": f"{tok_per_sec:,.0f}",
            })

        avg_loss = total_loss / max(raw_count, 1)
        return {
            "loss": avg_loss,
            "perplexity": compute_perplexity(avg_loss),
            "time": time.time() - epoch_start,
            "skipped_batches": skipped_batches,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set with progress bar."""
        if self.val_loader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Check if val_loader uses iterable dataset (no len())
        val_is_iterable = _is_iterable_dataset(getattr(self.val_loader, "dataset", None))
        
        # Use total=None for iterable datasets to avoid len() call
        pbar = tqdm(
            self.val_loader,
            desc="Validation",
            unit="batch",
            disable=not self.verbose,
            total=None if val_is_iterable else (len(self.val_loader) if hasattr(self.val_loader, "__len__") else None),
        )
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            if self.use_amp:
                with torch.amp.autocast(
                    device_type="cuda", dtype=self.amp_dtype
                ):
                    outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)
            else:
                outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)

            loss = outputs["loss"]
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1
                avg = total_loss / max(num_batches, 1)
                pbar.set_postfix({
                    "loss": f"{avg:.4f}",
                    "ppl": f"{compute_perplexity(avg):.2f}",
                })

        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss, "perplexity": compute_perplexity(avg_loss)}

    @torch.no_grad()
    def generate_sample(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> str:
        """Generate text from a prompt."""
        if self.tokenizer is None:
            return ""
        self.model.eval()
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt)], dtype=torch.long
        ).to(self.device)
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        generated = self.model.generate_ext(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
        )
        new_tokens = generated[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def save_checkpoint(self, epoch: int, filename: Optional[str] = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"helixlm_epoch_{epoch}.pt"
        path = os.path.join(self.output_dir, filename)
        self.model.save_pretrained(path)
        if self.verbose:
            print(f"Checkpoint saved to {path}")

    def train(
        self, num_epochs: Optional[int] = None, eval_every: int = 1
    ) -> Dict[str, Any]:
        """Train for specified number of epochs."""
        epochs = num_epochs or self.cfg.epochs
        effective_batch = self.cfg.batch_size * self.grad_accum_steps

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training HelixLM on {self.device}")
            print(f"Parameters: {self.model.count_parameters()['total']:,}")
            print(
                f"Epochs: {epochs} | Batch: {self.cfg.batch_size} | "
                f"Accum: {self.grad_accum_steps} | Effective: {effective_batch}"
            )
            print(f"LR: {self.cfg.lr} | AMP: {self.use_amp}")
            if self._is_iterable:
                print(f"Dataset: iterable (streaming)")
                if self.warmup_ratio is not None:
                    print(f"Warmup: {self.warmup_ratio} (ratio mode)")
                if self._cached_dataset_length is not None:
                    print(f"Counted batches: {self._cached_dataset_length:,}")
            print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            if self.verbose:
                print(f"\nEpoch {epoch}/{epochs}")
                print("-" * 40)

            train_metrics = self.train_epoch(epoch)
            skip_info = ""
            if train_metrics.get("skipped_batches", 0) > 0:
                skip_info = f" | Skipped: {train_metrics['skipped_batches']}"
            if self.verbose:
                print(
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"PPL: {train_metrics['perplexity']:.2f} | "
                    f"Time: {format_time(train_metrics['time'])}"
                    f"{skip_info}"
                )
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["perplexity"].append(train_metrics["perplexity"])

            val_loader_exists = self.val_loader is not None
            if val_loader_exists and epoch % eval_every == 0:
                val_metrics = self.evaluate()
                if self.verbose:
                    print(
                        f"Val Loss: {val_metrics['loss']:.4f} | "
                        f"Val PPL: {val_metrics['perplexity']:.2f}"
                    )
                self.history["val_loss"].append(val_metrics["loss"])
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint(epoch, "best_model")

            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

            if self.tokenizer and epoch % eval_every == 0 and self.verbose:
                print("\nGeneration samples:")
                for prompt in self.example_prompts:
                    if self.generated_example_length:
                        try:
                            generated = self.generate_sample(
                                prompt,
                                max_new_tokens=self.generated_example_length,
                            )
                            print(f"  '{prompt}' -> '{generated}'")
                        except Exception as e:
                            print(f"  '{prompt}' -> [Error: {e}]")
                    else:
                        print(
                            "Parameter 'generated_example_length' set to 0. "
                            "Skipping generation samples."
                        )
                print()

        self.save_checkpoint(epochs, "final_model")
        if self.verbose:
            print(f"\nTraining complete!")
        return self.history
