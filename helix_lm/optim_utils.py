"""
Optimizer utilities for HelixLM.

Provides helper functions for building hybrid optimizers (Muon + AdamW)
and parameter splitting by dimensionality. Use these instead of
monkey-patching the Trainer.

Example:
    from helix_lm.optim_utils import build_hybrid_muon_adamw

    optimizer = build_hybrid_muon_adamw(
        model, lr=1e-3, weight_decay=0.01,
        muon_lr=0.02, muon_ns_steps=5,
    )
    trainer = Trainer(model=model, cfg=cfg, ..., optimizer=optimizer)
"""
import torch
from torch.optim import AdamW

from .muon import Muon


def split_params_by_dimension(model):
    """Split model parameters into Muon-eligible (2D) and AdamW (non-2D) groups.

    Args:
        model: nn.Module with parameters to split

    Returns:
        tuple: (muon_params_list, adamw_params_list)
    """
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if (
            param.ndim == 2
            and "embed" not in name
            and "norm" not in name
            and param.requires_grad
        ):
            muon_params.append(param)
        elif param.requires_grad:
            adamw_params.append(param)
    return muon_params, adamw_params


def build_hybrid_muon_adamw(
    model,
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    muon_lr=0.02,
    muon_momentum=0.95,
    muon_nesterov=True,
    muon_ns_steps=5,
    muon_weight_decay=0.0,
):
    """Build a hybrid Muon + AdamW optimizer list for Trainer.

    Muon handles 2D weight matrices (projections, FFN weights).
    AdamW handles non-2D params (embeddings, norms, biases).

    Pass the returned list to Trainer as optimizer=[muon, adamw].
    The Trainer handles stepping, zero_grad, and scheduler binding
    automatically via its multi-optimizer support.

    Args:
        model: HelixForCausalLM model instance
        lr: Learning rate for AdamW params (default: 1e-3)
        weight_decay: Weight decay for AdamW (default: 0.01)
        betas: AdamW beta values (default: (0.9, 0.999))
        muon_lr: Muon learning rate (default: 0.02)
        muon_momentum: Muon momentum (default: 0.95)
        muon_nesterov: Use Nesterov momentum (default: True)
        muon_ns_steps: Newton-Schulz iteration steps (default: 5)
        muon_weight_decay: Weight decay for Muon (default: 0.0)

    Returns:
        list: [Muon optimizer, AdamW optimizer] for Trainer
    """
    muon_params, adamw_params = split_params_by_dimension(model)

    muon_p_n = sum(p.numel() for p in muon_params)
    adam_p_n = sum(p.numel() for p in adamw_params)

    muon = Muon(
        muon_params,
        lr=muon_lr,
        momentum=muon_momentum,
        nesterov=muon_nesterov,
        ns_steps=muon_ns_steps,
        weight_decay=muon_weight_decay,
    )
    adamw = AdamW(
        adamw_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )

    return [muon, adamw]


def count_optimizer_params(optimizers):
    """Count total parameters managed by optimizer(s).

    Args:
        optimizers: Single optimizer or list of optimizers

    Returns:
        dict: {'total': N, 'by_group': [...]}
    """
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]

    by_group = []
    total = 0
    for i, opt in enumerate(optimizers):
        n = sum(
            p.numel() for g in opt.param_groups for p in g["params"]
        )
        by_group.append({"optimizer": i, "params": n})
        total += n

    return {"total": total, "by_group": by_group}
