"""
HybridOptimizer: Muon for 2D weight matrices, AdamW for everything else.
Presents a single torch.optim.Optimizer interface for Trainer/GradScaler/Scheduler.
"""

import torch
from torch.optim import Optimizer, AdamW
from .muon import Muon


class HybridOptimizer(Optimizer):
    """
    Wraps Muon + AdamW as one Optimizer.
    Scheduler and GradScaler see unified param_groups.
    """

    def __init__(self, muon_optimizer: Muon, adamw_optimizer: AdamW):
        # Guard 1: exhaustive overlap / omission check
        muon_ids = {id(p) for g in muon_optimizer.param_groups for p in g["params"]}
        adamw_ids = {id(p) for g in adamw_optimizer.param_groups for p in g["params"]}

        if muon_ids & adamw_ids:
            raise ValueError(
                f"HybridOptimizer: {len(muon_ids & adamw_ids)} parameters assigned to BOTH "
                "Muon and AdamW. Partition must be mutually exclusive."
            )
        if not (muon_ids | adamw_ids):
            raise ValueError("HybridOptimizer: no trainable parameters assigned.")

        # Pass a param_group dict with EMPTY params list.
        # PyTorch checks len(param_groups) > 0, not len(params) > 0.
        # This avoids both "empty parameter list" and add_param_group() issues.
        super().__init__([{"params": []}], dict(lr=0.0))

        self.muon = muon_optimizer
        self.adamw = adamw_optimizer

        # Replace with live references so Scheduler LR mutations propagate in-place
        self.param_groups = []
        for g in self.muon.param_groups:
            self.param_groups.append(g)
        for g in self.adamw.param_groups:
            self.param_groups.append(g)

    def step(self, closure=None):
        loss_muon = self.muon.step(closure)
        loss_adamw = self.adamw.step(closure)
        return loss_muon if loss_muon is not None else loss_adamw

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])

    def add_param_group(self, param_group):
        raise NotImplementedError("HybridOptimizer does not support add_param_group")


def build_hybrid_optimizer(model, cfg):
    """
    Build HybridOptimizer from HelixConfig.
    Muon: 2D params excluding embeddings / output head.
    AdamW: embeddings, LM head, biases, norms, and all non-2D params.
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in name and "head" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    # Redundant safety: verify exhaustiveness
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(muon_params) + len(adamw_params) == len(all_trainable), (
        f"Partition mismatch: Muon({len(muon_params)}) + AdamW({len(adamw_params)}) "
        f"!= Total({len(all_trainable)})"
    )

    muon = Muon(
        muon_params,
        lr=cfg.lr * getattr(cfg, "muon_lr_factor", 1.0),
        momentum=getattr(cfg, "muon_momentum", 0.95),
        nesterov=True,
        ns_steps=getattr(cfg, "muon_ns_steps", 5),
        weight_decay=cfg.weight_decay,
    )

    adamw = AdamW(
        adamw_params,
        lr=cfg.lr * getattr(cfg, "adamw_lr_factor", 0.1),
        weight_decay=cfg.weight_decay,
        betas=getattr(cfg, "adamw_betas", (0.9, 0.999)),
    )

    return HybridOptimizer(muon, adamw)
