"""
Heterogeneous neural nodes for HelixLM.

Each node type implements a different computational mechanism inspired by
diverse neural structures in biological brains.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        norm = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        weight = self.weight.to(x_f.dtype)
        return (x_f * norm * weight).to(dtype)


class HeteroNode(nn.Module):
    """Base class for all heterogeneous nodes."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError


class ErrorCorrectingMultiScaleAttnNode(HeteroNode):
    """
    Three-path attention with learned consensus and logical error correction.

    Architecture (per layer):
        1. Ensemble     : local windowed, coarse windowed, compressed global
        2. Consensus  : cosine-similarity soft voting across 3 views (outlier rejection)
        3. Correction : bottleneck FFN anomaly detector (2d -> d/2 -> d)
        4. Output FFN : standard post-attention reasoning layer (d -> 4d -> d)

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads. Must divide d_model.
    local_window : int, default 64
        Window size for fine-grained local attention.
    coarse_window : int, default 128
        Window size for mid-range coarse attention.
    compressed_windows : int, default 8
        Fixed number of compressed global tokens. Independent of sequence length.
        Increase sub-linearly if raising T significantly (e.g., 8 -> 16 -> 32).
    corrector_dim : int | None, default None
        Hidden dimension of the bottleneck corrector FFN.
        None defaults to d_model // 2.
    output_ffn_dim : int | None, default None
        Hidden dimension of the final output FFN.
        None defaults to 4 * d_model (standard Transformer).
    consensus_type : {"cosine", "mha"}, default "cosine"
        "cosine"  : cosine-similarity soft voting (~0 params, recommended).
        "mha"     : legacy multi-head self-attention over 3 views (~4.2M params).
    corrector_type : {"ffn", "attn"}, default "ffn"
        "ffn"  : bottleneck FFN on [consensus, original] (recommended).
        "attn" : experimental tiny self-attention + FFN on consensus.
    dropout : float, default 0.0
        Dropout rate for FFN and residual paths.
    attn_dropout : float, default 0.0
        Dropout rate inside attention softmax. If 0, falls back to `dropout`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        local_window: int = 64,
        coarse_window: int = 128,
        compressed_windows: int = 8,
        corrector_dim: int | None = None,
        output_ffn_dim: int | None = None,
        consensus_type: str = "cosine",
        corrector_type: str = "ffn",
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__(d_model)
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.local_window = local_window
        self.coarse_window = coarse_window
        self.compressed_windows = compressed_windows

        # ------------------------------------------------------------------
        # Layer 1: Ensemble (shared QKV + output projection)
        # ------------------------------------------------------------------
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Compressed global path
        self.compress_proj = nn.Linear(d_model, d_model)
        self.compress_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.expand_proj = nn.Linear(d_model, d_model)

        # ------------------------------------------------------------------
        # Layer 2: Consensus
        # ------------------------------------------------------------------
        self.consensus_type = consensus_type.lower()
        if self.consensus_type == "mha":
            self.cross_view_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=attn_dropout, batch_first=True
            )
            self.cross_view_norm = RMSNorm(d_model)
        elif self.consensus_type == "cosine":
            self.consensus_temp = nn.Parameter(torch.ones(1))
            self.consensus_norm = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown consensus_type: {consensus_type}")

        # ------------------------------------------------------------------
        # Layer 3: Correction
        # ------------------------------------------------------------------
        self.corrector_type = corrector_type.lower()
        corrector_hidden = corrector_dim if corrector_dim is not None else d_model // 2

        if self.corrector_type == "ffn":
            self.corrector = nn.Sequential(
                nn.Linear(d_model * 2, corrector_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(corrector_hidden, d_model),
            )
        elif self.corrector_type == "attn":
            # Experimental: tiny self-attention over consensus, then FFN
            self.corrector_attn = nn.MultiheadAttention(
                d_model, max(1, n_heads // 4), dropout=attn_dropout, batch_first=True
            )
            self.corrector_ffn = nn.Sequential(
                nn.Linear(d_model, corrector_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(corrector_hidden, d_model),
            )
            self.corrector_norm1 = RMSNorm(d_model)
            self.corrector_norm2 = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown corrector_type: {corrector_type}")

        self.corrector_gate = nn.Linear(d_model * 2, 1)

        # ------------------------------------------------------------------
        # Layer 4: Output FFN
        # ------------------------------------------------------------------
        out_ffn_hidden = output_ffn_dim if output_ffn_dim is not None else 4 * d_model
        self.output_ffn = nn.Sequential(
            nn.Linear(d_model, out_ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_ffn_hidden, d_model),
        )
        self.output_norm = RMSNorm(d_model)

        # Final residual gate
        self.out_gate = nn.Linear(d_model * 2, 1)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout if attn_dropout > 0 else dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _pad_to_window(self, x: torch.Tensor, win: int):
        """Zero-pad sequence length to nearest multiple of win."""
        B, T, D = x.shape
        pad_len = (win - T % win) % win
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        return x, pad_len

    def _windowed_attn(self, x: torch.Tensor, win: int):
        B, T, D = x.shape
        x_pad, pad_len = self._pad_to_window(x, win)
        T_pad = T + pad_len
        nwin = T_pad // win

        xw = x_pad.reshape(B, nwin, win, D).view(B * nwin, win, D)

        qkv = self.qkv_proj(xw).view(B * nwin, win, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal = torch.triu(torch.ones(win, win, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B * nwin, win, D)
        out = self.out_proj(out)
        out = out.view(B, nwin, win, D).reshape(B, T_pad, D)

        if pad_len:
            out = out[:, :T, :]
        return out

    def _compressed_attn(self, x: torch.Tensor):
        B, T, D = x.shape
        z = self.compress_proj(x)                       # (B, T, D)
        z = F.adaptive_avg_pool1d(z.transpose(1, 2), self.compressed_windows).transpose(1, 2)
        z, _ = self.compress_attn(z, z, z, need_weights=False)
        z = z.transpose(1, 2)                           # (B, D, W)
        z = F.interpolate(z, size=T, mode="nearest")    # (B, D, T)
        z = z.transpose(1, 2)                           # (B, T, D)
        return self.expand_proj(z)

    def forward(self, x, state=None, cache=None, attention_mask=None, **kwargs):
        # Assume HeteroNode provides self.norm
        x = self.norm(x)

        # ===== Layer 1: Ensemble =====
        local = self._windowed_attn(x, self.local_window)
        coarse = self._windowed_attn(x, self.coarse_window)
        comp = self._compressed_attn(x)

        # ===== Layer 2: Consensus =====
        stacked = torch.stack([local, coarse, comp], dim=2)  # [B, T, 3, D]

        if self.consensus_type == "mha":
            B, T, V, D = stacked.shape
            flat = stacked.reshape(B * T, V, D)
            cv_out, attn_weights = self.cross_view_attn(flat, flat, flat)
            cv_out = self.cross_view_norm(flat + cv_out)
            self_confidence = attn_weights.diagonal(dim1=1, dim2=2)  # [B*T, 3]
            vote_weights = F.softmax(self_confidence, dim=-1).unsqueeze(-1)
            consensus = (cv_out * vote_weights).sum(dim=1).reshape(B, T, D)
        else:  # cosine (default)
            # L2-normalize for scale-invariant cosine similarity
            stacked_norm = F.normalize(stacked, dim=-1, p=2)
            # Pairwise cosine: [B, T, 3, 3]
            sim = torch.matmul(stacked_norm, stacked_norm.transpose(-2, -1))
            # Mean agreement per view (self-sim is always 1.0)
            confidence = sim.mean(dim=-1)  # [B, T, 3]
            vote_weights = F.softmax(confidence * self.consensus_temp, dim=-1).unsqueeze(-1)  # [B, T, 3, 1]
            consensus = (stacked * vote_weights).sum(dim=2)  # [B, T, D]
            consensus = self.consensus_norm(consensus)

        # ===== Layer 3: Correction =====
        corrector_input = torch.cat([consensus, x], dim=-1)  # [B, T, 2D]

        if self.corrector_type == "ffn":
            delta = self.corrector(corrector_input)
        else:  # "attn" — experimental
            c_att, _ = self.corrector_attn(consensus, consensus, consensus)
            c_att = self.corrector_norm1(consensus + c_att)
            delta = self.corrector_ffn(c_att)
            delta = self.corrector_norm2(c_att + delta)

        fix_gate = torch.sigmoid(self.corrector_gate(corrector_input))  # [B, T, 1]
        corrected = consensus + fix_gate * delta

        # Gated residual merge
        out_gate = torch.sigmoid(self.out_gate(torch.cat([corrected, x], dim=-1)))  # [B, T, 1]
        out = x + out_gate * self.dropout(corrected)

        # ===== Layer 4: Output FFN =====
        out = out + self.dropout(self.output_ffn(self.output_norm(out)))

        return out, None


class LinearAttnNode(HeteroNode):
    """
    Causal linear attention node using feature maps.
    O(n) in sequence length during training via prefix sums.
    """
    def __init__(self, d_model: int, n_heads: int = 4, feature_dim: int = 64, dropout: float = 0.0,
                 attn_dropout: float = 0.0):
        super().__init__(d_model)
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.feature_dim = feature_dim

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.q_feat = nn.Linear(self.head_dim, feature_dim, bias=False)
        self.k_feat = nn.Linear(self.head_dim, feature_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout if attn_dropout > 0 else dropout)

        self._reset_parameters()

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        B, T, D = x.shape
        x = self.norm(x)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # AMP-safe: compute feature maps and cumulatives in fp32 to prevent
        # float16 overflow. The einsum over feature_dim can exceed 65504.
        # Cast weights to fp32 so matmul dtype matches even when weights are bf16.
        q_fp32 = self._feature_map(F.linear(q.float(), self.q_feat.weight.float(),
                                            self.q_feat.bias.float() if self.q_feat.bias is not None else None))
        k_fp32 = self._feature_map(F.linear(k.float(), self.k_feat.weight.float(),
                                            self.k_feat.bias.float() if self.k_feat.bias is not None else None))
        v_fp32 = v.float()

        # Apply attention mask: zero out k,v contributions from pad positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, T, 1)
            k_fp32 = k_fp32 * mask
            v_fp32 = v_fp32 * mask

        kv = torch.einsum('bhTf,bhTd->bhTfd', k_fp32, v_fp32)

        # Move T to the LAST dimension so cumsum scans contiguous memory (stride = 1)
        kv = kv.permute(0, 1, 3, 4, 2).contiguous()   # [B, H, F, D, T]
        kv_cum = torch.cumsum(kv, dim=-1)              # one contiguous scan per row
        kv_cum = kv_cum.permute(0, 1, 4, 2, 3)        # [B, H, T, F, D]

        z = torch.cumsum(k_fp32, dim=2).sum(dim=-1, keepdim=True).clamp(min=1e-6)

        out = torch.einsum('bhTf,bhTfd->bhTd', q_fp32, kv_cum) / z
        out = out.to(x.dtype)  # cast back to fp16/bf16
        # --------------------------------------------------------------

        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.attn_dropout(self.out_proj(out))
        return out, None


class FullAttnNode(HeteroNode):
    """Standard causal softmax attention with multi-head support and optional RoPE."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0, use_rope: bool = True,
                 attn_dropout: float = 0.0):
        super().__init__(d_model)
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout if attn_dropout > 0 else dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        B, T, D = x.shape
        x = self.norm(x)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        # Padding mask: prevent attending to padded positions
        if attention_mask is not None:
            # attention_mask: (B, T) with 0 for pad, 1 for real
            # Expand to (B, n_heads, T, T) to mask key positions for all queries
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask.expand(-1, self.n_heads, T, -1), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.resid_dropout(self.out_proj(out))
        return out, None


class DenseNode(HeteroNode):
    """Dense processing node with GELU activation."""
    def __init__(self, d_model: int, expansion: float = 2.0, dropout: float = 0.0):
        super().__init__(d_model)
        h = int(d_model * expansion)
        self.w1 = nn.Linear(d_model, h)
        self.w2 = nn.Linear(h, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        x = self.norm(x)
        h = F.gelu(self.w1(x))
        out = self.w2(h)
        return self.dropout(out), None


class SwiGLUNode(HeteroNode):
    """SwiGLU activation node as used in modern LLMs."""
    def __init__(self, d_model: int, expansion: float = 2.0, dropout: float = 0.0):
        super().__init__(d_model)
        h = int(d_model * expansion)
        self.gate = nn.Linear(d_model, h, bias=False)
        self.up = nn.Linear(d_model, h, bias=False)
        self.down = nn.Linear(h, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.xavier_uniform_(self.down.weight)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        x = self.norm(x)
        h = F.silu(self.gate(x)) * self.up(x)
        out = self.down(h)
        return self.dropout(out), None


class SSMNode(HeteroNode):
    """
    Simplified SSM node (Mamba-style) with efficient batched sequential scan.
    For production, replace with Mamba2SSD from mamba2.py.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__(d_model)
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=False
        )
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log.data = torch.log(A)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        B, T, D = x.shape
        x = self.norm(x)

        x_and_gate = self.in_proj(x)
        x_inner, gate = x_and_gate.chunk(2, dim=-1)
        x_inner = x_inner * F.silu(gate)

        x_conv = self.conv(x_inner.transpose(1, 2))[:, :, :T].transpose(1, 2)

        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log.float())
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        dt = dt.float()
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(2)

        if state is None:
            h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        else:
            h = state

        ys = []
        for t in range(T):
            h = A_bar[:, t] * h + B_bar[:, t] * x_conv[:, t].unsqueeze(-1)
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            y = y + self.D * x_conv[:, t]
            ys.append(y)

        out = torch.stack(ys, dim=1)
        out = self.dropout(self.out_proj(out.to(x.dtype)))
        return out, h.detach()


class Mamba2Node(HeteroNode):
    """Mamba-2 SSD node wrapper."""
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2,
                 dt_rank: str = "auto", conv_bias: bool = True, bias: bool = False, dropout: float = 0.0):
        super().__init__(d_model)
        from .mamba2 import Mamba2SSD
        self.mamba = Mamba2SSD(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, dt_rank=dt_rank, conv_bias=conv_bias, bias=bias,
            use_fast_path=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        x = self.norm(x)
        out, new_state = self.mamba(x, state=state)
        return self.dropout(out), new_state


class GateNode(HeteroNode):
    """Aggregation node with learned softmax weighted sum."""
    def __init__(self, d_model: int, n_preds: int = 2, dropout: float = 0.0):
        super().__init__(d_model)
        self.n_preds = n_preds
        self.weights = nn.Parameter(torch.ones(n_preds) / n_preds)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x_list, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        if not isinstance(x_list, list):
            raise TypeError("GateNode expects a list of predecessor tensors")
        n = len(x_list)
        if n == 0:
            raise ValueError("GateNode received no predecessors")
        weights = F.softmax(self.weights[:n], dim=0)
        out = sum(w * x for w, x in zip(weights, x_list))
        out = self.out_proj(out)
        return self.dropout(out), None



class TitansMemoryNode(HeteroNode):
    """
    Titans-style neural long-term memory node for HelixLM.

    Maintains persistent memory across forward passes using a surprise-gated
    delta rule. Compatible with the existing heterogeneous graph interface.

    Architecture (based on Behrouz et al. 2025 "Titans: Learning to Memorize
    at Test Time" MAC variant):
        - Keys and values are projected from the input hidden states.
        - A persistent memory tensor M (batch, feature_dim, d_model) stores
          the long-term key->value mapping via outer-product updates.
        - Surprise metric = ||v_pred - v_true|| drives update magnitude.
        - Retrieval uses query projection + ELU feature map.

    The memory tensor is returned as ``state`` and can be persisted across
    chunks by the caller (HelixRecurrentBlock / HelixLMCore).
    """
    def __init__(
        self,
        d_model: int,
        feature_dim: int = 64,
        eta_init: float = 0.01,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__(d_model)
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Projections for memory keys, values, and retrieval queries
        self.k_proj = nn.Linear(d_model, feature_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, feature_dim, bias=False)

        # Learnable per-coordinate learning rate for memory updates
        self.eta = nn.Parameter(torch.ones(feature_dim) * eta_init)

        # Feature map: ELU + 1 (standard in Titans / linear attention literature)
        self.phi = lambda x: F.elu(x, alpha=1.0) + 1.0

        # Output projection and dropout
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _init_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialize a zero memory tensor for a new batch."""
        return torch.zeros(batch_size, self.feature_dim, self.d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: Any = None,
        cache: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            x: (B, T, D) input hidden states.
            state: Previous persistent memory tensor (B, feature_dim, D) or None.
            cache: Unused (for API compatibility).

        Returns:
            (output, updated_memory) where updated_memory has shape
            (B, feature_dim, D) and should be passed back on the next chunk.
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        # 1. Retrieve or initialize persistent memory
        if state is not None:
            M = state  # (B, feature_dim, D)
        else:
            M = self._init_memory(B, device, dtype)

        # 2. Pre-norm and project to keys / values
        x_norm = self.norm(x)

        k = self.phi(self.k_proj(x_norm))  # (B, T, feature_dim)
        v = self.v_proj(x_norm)             # (B, T, d_model)

        # 3. Memory update loop (test-time learning)
        eta = self.eta.abs().clamp(min=1e-6)  # (feature_dim,)

        for t in range(T):
            k_t = k[:, t, :]          # (B, feature_dim)
            v_t = v[:, t, :]          # (B, d_model)

            # Surprise metric: deviation of memory prediction from true value
            v_pred = torch.einsum('bf,bfd->bd', k_t, M)
            surprise = torch.norm(v_t - v_pred, dim=-1, keepdim=True)  # (B, 1)

            # Delta rule update: outer product of key and value
            delta = torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(1))

            # Surprise-gated update with learnable per-coordinate LR
            M = M + eta.view(1, -1, 1) * surprise.unsqueeze(-1) * delta

            # Layer norm over the feature_dim dimension to prevent explosion
            M = F.layer_norm(M, M.shape[-2:])

        # 4. Memory retrieval for output
        q = self.phi(self.q_proj(x_norm))  # (B, T, feature_dim)
        mem_out = torch.einsum('btf,bfd->btd', q, M)  # (B, T, d_model)

        # 5. Output projection + residual
        out = self.dropout(self.out_proj(mem_out))
        output = x + out

        return output, M
