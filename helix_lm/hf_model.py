"""
HelixLM HuggingFace PreTrainedModel integration (V5-orthodox).

Fixes applied:
  1. Weight-tying: removed _tied_weights_keys={} override; uses HF's built-in mechanism.
  2. Generation: prepare_inputs_for_generation passes FULL sequence (no KV-cache).
  3. Standard generate() now works via GenerationMixin; generate_ext() is deprecated alias.
  4. Auto-registration: explicit, visible, no silent try/except.
  5. use_cache=False is hard-enforced — the recurrent graph has no KV state.

Provides full compatibility with the transformers ecosystem:
  - HelixForCausalLM: AutoModelForCausalLM registration
  - save_pretrained / from_pretrained / push_to_hub
  - Standard model.generate() with StoppingCriteria, logits processors
  - Batched generation with stop token / stop string detection
  - Automatic device placement when config.device="auto"
"""
import math
from typing import Optional, List, Dict, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    GenerationMixin,
    AutoModelForCausalLM,
    AutoConfig,
    StoppingCriteria,
)

from .config import HelixConfig
from .model import HelixLMCore
from .tokenizer import HelixTokenizer


# ---------------------------------------------------------------------------
# Stop string detection
# ---------------------------------------------------------------------------
class StopStringCriteria(StoppingCriteria):
    """Stops generation when any of the given strings is produced."""
    def __init__(self, tokenizer: HelixTokenizer, stop_strings: List[str], batch_size: int = 1):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.batch_size = batch_size
        self._decoded = [""] * batch_size

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        should_stop = False
        for b in range(input_ids.shape[0]):
            text = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)
            for stop_str in self.stop_strings:
                if stop_str in text:
                    should_stop = True
                    break
        return should_stop


# ---------------------------------------------------------------------------
# HelixForCausalLM: HF-compatible model
# ---------------------------------------------------------------------------
class HelixPreTrainedModel(PreTrainedModel):
    """Base class for HelixLM models with HF integration."""
    config_class = HelixConfig
    base_model_prefix = "helix"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HelixRecurrentBlock"]
    # REMOVED: _tied_weights_keys = {}  — was killing HF weight-tying mechanism
    # REMOVED: get_expanded_tied_weights_keys override

    def _init_weights(self, module):
        """Initialize weights the same way as the core model."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def to_device(self, device: Optional[Union[str, torch.device]] = None) -> "HelixPreTrainedModel":
        """Move model to the specified device, or auto-detect if None."""
        if device is None:
            device = self._resolve_device()
        return self.to(device)

    def _resolve_device(self) -> torch.device:
        """Resolve device from config or auto-detect."""
        cfg_device = getattr(self.config, "device", "auto")
        if cfg_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(cfg_device)


class HelixForCausalLM(HelixPreTrainedModel, GenerationMixin):
    """
    HelixLM causal language model with full HuggingFace compatibility.

    Weight-tying behavior:
      - If config.tie_word_embeddings=True, lm_head.weight shares storage with
        model.embed.weight (standard HF pattern).
      - This is handled by calling tie_weights() in __init__ and letting HF's
        built-in _tie_or_clone_weights() do the work.

    Generation behavior:
      - NO KV-cache: the recurrent graph re-initializes node_states on every
        forward.  Generation must pass the full sequence on every step.
      - prepare_inputs_for_generation honors this by passing the full input_ids
        (or last seq_len window) and ignoring past_key_values.
    """
    _tied_weights_keys = {"lm_head.weight": "lm_head.weight"}

    def __init__(self, config: HelixConfig):
        super().__init__(config)
        self.config = config

        # Hard-enforce: this model has no KV-cache
        self.config.use_cache = False

        # Core model without output head (HelixForCausalLM owns lm_head)
        self.model = HelixLMCore(config, tie_weights=False, create_output_head=False)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()

        # Tie weights if requested (HF standard mechanism)
        if config.tie_word_embeddings:
            self.tie_weights()

        # Auto device placement
        target_device = self._resolve_device()
        if target_device.type == "cpu" and torch.cuda.is_available():
            import warnings
            warnings.warn(
                f"HelixConfig.device='{getattr(config, 'device', 'auto')}' resolved to CPU, "
                f"but CUDA is available. Model will run on CPU. "
                f"Call model.to('cuda') or set config.device='cuda' to use GPU.",
                UserWarning,
                stacklevel=2,
            )
        self.to(target_device)

    def tie_weights(self, recompute_mapping=True):
        """Tie lm_head.weight to model.embed.weight using HF's built-in mechanism."""
        if self.config.tie_word_embeddings:
            # Direct weight sharing (same tensor object)
            self.lm_head.weight = self.model.embed.weight
        return super().tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Dict]:
        """
        Forward pass compatible with HF transformers.
        """
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)
        # IGNORE use_cache — this model has no KV-cache
        _ = use_cache  # noqa: F841

        if inputs_embeds is not None:
            e = inputs_embeds
        else:
            e = self.model.embed(input_ids)

        # Run through recurrent core
        h = self.model.recurrent(e, e)

        # Output
        h = self.model.out_norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            if getattr(self.config, "memory_efficient_forward", False):
                del logits
                logits = None
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": None,
            "hidden_states": h if output_hidden_states else None,
            "attentions": None,
        }

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for the .generate() method.

        CRITICAL: This model has NO KV-cache.  The recurrent graph re-initializes
        node_states on every forward.  Therefore we must pass the FULL sequence
        (or the last seq_len window) on every generation step — NEVER just
        input_ids[:, -1:].

        past_key_values is explicitly ignored.
        """
        # Ignore past_key_values — the recurrent graph doesn't use them
        if past_key_values is not None:
            pass  # explicitly no-op

        # Pass the full sequence, but cap at seq_len to avoid OOM on long gens
        seq_len = input_ids.shape[1]
        if seq_len > self.config.seq_len:
            input_ids = input_ids[:, -self.config.seq_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.config.seq_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # Do NOT pass past_key_values or use_cache — this model is stateless
        }

    def _reorder_cache(self, past_key_values: Any, beam_idx: torch.Tensor) -> Any:
        """Reorder cache for beam search."""
        # For stateless architectures, nothing to reorder
        return past_key_values

    @torch.no_grad()
    def generate_ext(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        stop_strings: Optional[List[str]] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        return_full_text: bool = True,
    ) -> torch.Tensor:
        """
        DEPRECATED: Use model.generate() instead.

        Kept as a thin alias for backward compatibility.
        """
        import warnings
        warnings.warn(
            "generate_ext() is deprecated. Use model.generate() with standard "
            "HF GenerationMixin arguments (temperature, top_k, top_p, "
            "stopping_criteria, etc.)",
            DeprecationWarning,
            stacklevel=2,
        )
        # Delegate to the standard GenerationMixin.generate()
        return self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# Explicit Auto-registration (no silent try/except)
# ---------------------------------------------------------------------------
def register_helix_for_auto_classes():
    """Register HelixLM config and model classes with HF AutoClasses."""
    try:
        AutoConfig.register("helix", HelixConfig)
    except Exception as e:
        import warnings
        warnings.warn(f"AutoConfig.register('helix') failed: {e}. "
                      "You may need to use trust_remote_code=True when loading.",
                      RuntimeWarning)
        return

    try:
        AutoModelForCausalLM.register(HelixConfig, HelixForCausalLM)
    except Exception as e:
        import warnings
        warnings.warn(f"AutoModelForCausalLM.register failed: {e}. "
                      "You may need to use trust_remote_code=True when loading.",
                      RuntimeWarning)
        return

    # Register for auto_class so push_to_hub writes auto_map in config.json
    try:
        HelixConfig.register_for_auto_class()
        HelixForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    except Exception as e:
        import warnings
        warnings.warn(f"register_for_auto_class failed: {e}. "
                      "Hub push/pull may require manual trust_remote_code=True.",
                      RuntimeWarning)

# Register on import
register_helix_for_auto_classes()
