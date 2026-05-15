
"""
Muon Optimizer (MomentUm Orthogonalized by Newton-schulz)
=========================================================

Implements Muon with Newton-Schulz iteration for orthogonalizing gradient updates.
Best for 2D weight matrices; combine with AdamW for non-2D parameters.

Based on https://github.com/KellerJordan/Muon
"""
import torch
from torch.optim.optimizer import Optimizer


class Muon(Optimizer):
    """
    Muon optimizer: orthogonalizes momentum updates via Newton-Schulz iteration.

    Designed for 2D parameter matrices. Use alongside AdamW for non-2D params.

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 0.02)
        momentum: momentum factor (default: 0.95)
        nesterov: use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        orthogonalize: whether to orthogonalize (default: True)
        weight_decay: weight decay (default: 0.0)
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, 
                 ns_steps=5, orthogonalize=True, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                       ns_steps=ns_steps, orthogonalize=orthogonalize,
                       weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            orthogonalize = group['orthogonalize']
            weight_decay = group['weight_decay']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Get or initialize momentum buffer
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(p)
                buf = param_state['momentum_buffer']

                # Update momentum
                buf.mul_(momentum).add_(grad)

                # Nesterov: use momentum buffer + gradient
                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf

                # Orthogonalize via Newton-Schulz iteration
                if orthogonalize and update.ndim == 2:
                    update = newton_schulz(update, steps=ns_steps)
                    # Scale by matrix norm for adaptive LR
                    scale = max(1, p.shape[0] / p.shape[1]) ** 0.5
                    update.mul_(scale)

                # Apply update
                p.add_(update, alpha=-lr)

        return loss


def newton_schulz(G, steps=5, eps=1e-7):
    """
    Orthogonalize a matrix using Newton-Schulz iteration.

    Finds the closest orthogonal matrix to G via iterative refinement.
    More stable than SVD for gradient updates.

    Args:
        G: gradient matrix (2D tensor)
        steps: number of iteration steps (default: 5)
        eps: small constant for numerical stability

    Returns:
        Orthogonalized matrix
    """
    if G.ndim != 2:
        return G

    # Normalize for stability
    a, b = G.shape
    # Use transpose for skinny matrices (more stable)
    transpose = a < b
    if transpose:
        G = G.T

    # Compute normalizer
    norm = G.norm()
    if norm < eps:
        return G if not transpose else G.T

    # Normalize
    X = G / norm

    # Newton-Schulz iteration: X_{k+1} = 1.5 * X_k - 0.5 * X_k * X_k^T * X_k
    for _ in range(steps):
        XTX = X.T @ X
        X = 1.5 * X - 0.5 * X @ XTX

    # Rescale
    X = X * norm

    if transpose:
        X = X.T

    return X
