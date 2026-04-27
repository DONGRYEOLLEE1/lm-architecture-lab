"""Pure-PyTorch Mamba-2 (State-Space Duality).

Reference: Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient
Algorithms Through Structured State-Space Duality" (ICML 2024).

What changes vs Mamba-1:
  - A is a **scalar per head** instead of a (d_inner, d_state) matrix. This
    is the SSD restriction that makes the recurrence identical to a
    structured causal attention.
  - B and C are shared across heads inside a "group" — n_groups = 1 means
    all heads share the same B/C, exactly matching the SSD masked-attention
    interpretation.
  - dt becomes per-head (broadcast across head_dim) instead of per-channel.
  - A single big `in_proj` produces [z, x, B, C, dt] all at once, so the
    block has fewer linear layers than Mamba-1.
  - A mid-block normalization sits between the SSM output and the gate, so
    the output projection sees a normalized signal.

We keep a sequential reference for the scan rather than the chunked SSD form
so the file maps cleanly to Algorithm 2 in the paper. The chunk-parallel
form is what the official CUDA path uses, but it does not change what the
recurrence computes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Mamba2Config:
    d_model: int = 256
    n_layer: int = 4
    vocab_size: int = 256
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 32
    n_groups: int = 1
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_init_floor: float = 1e-4
    A_init_range: tuple[float, float] = (1.0, 16.0)
    norm_eps: float = 1e-5
    bias: bool = False
    conv_bias: bool = True
    pad_vocab_size_multiple: int = 8

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_state <= 0:
            raise ValueError("d_state must be positive")
        if self.d_conv <= 0:
            raise ValueError("d_conv must be positive")
        if self.expand <= 0:
            raise ValueError("expand must be positive")
        if self.headdim <= 0:
            raise ValueError("headdim must be positive")
        if self.n_groups <= 0:
            raise ValueError("n_groups must be positive")
        if self.pad_vocab_size_multiple <= 0:
            raise ValueError("pad_vocab_size_multiple must be positive")
        self.d_inner = self.expand * self.d_model
        if self.d_inner % self.headdim != 0:
            raise ValueError("expand * d_model must be divisible by headdim")
        self.nheads = self.d_inner // self.headdim
        if self.nheads % self.n_groups != 0:
            raise ValueError("nheads must be divisible by n_groups")
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += self.pad_vocab_size_multiple - (
                self.vocab_size % self.pad_vocab_size_multiple
            )


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def ssd_sequential_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Sequential reference for the SSD recurrence.

    Shapes:
        x  : (b, l, h, d)   — h heads of width d
        dt : (b, l, h)       — per-head time-step delta
        A  : (h,)            — scalar per head (SSD restriction)
        B  : (b, l, g, n)    — g groups, n state dim (head-shared)
        C  : (b, l, g, n)
    Returns:
        y  : (b, l, h, d)

    The state per head is a (d, n) matrix that gets discounted by exp(dt*A)
    each step and accumulates the rank-1 update (dt * x) outer-product B.
    """
    b, l, h, d = x.shape
    n = B.shape[-1]
    g = B.shape[-2]
    if h % g != 0:
        raise ValueError("nheads must be divisible by n_groups")
    rep = h // g
    if rep > 1:
        B = B.repeat_interleave(rep, dim=-2)  # (b, l, h, n)
        C = C.repeat_interleave(rep, dim=-2)

    dA = torch.exp(dt * A)  # (b, l, h)

    state = x.new_zeros(b, h, d, n)
    ys = []
    for t in range(l):
        x_t = x[:, t]                    # (b, h, d)
        dt_t = dt[:, t].unsqueeze(-1)    # (b, h, 1)
        dA_t = dA[:, t]                  # (b, h)
        B_t = B[:, t]                    # (b, h, n)
        C_t = C[:, t]                    # (b, h, n)

        update = (dt_t * x_t).unsqueeze(-1) * B_t.unsqueeze(-2)  # (b, h, d, n)
        state = dA_t.unsqueeze(-1).unsqueeze(-1) * state + update
        y_t = (state * C_t.unsqueeze(-2)).sum(dim=-1)  # (b, h, d)
        ys.append(y_t)

    return torch.stack(ys, dim=1)  # (b, l, h, d)


class Mamba2Block(nn.Module):
    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.d_inner
        h = cfg.nheads
        g = cfg.n_groups
        n = cfg.d_state

        # Single fused projection: [z, x, B, C, dt]
        d_in_proj = 2 * d_in + 2 * g * n + h
        self.in_proj = nn.Linear(cfg.d_model, d_in_proj, bias=cfg.bias)

        conv_dim = d_in + 2 * g * n
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=cfg.d_conv,
            groups=conv_dim,
            padding=cfg.d_conv - 1,
            bias=cfg.conv_bias,
        )

        # dt bias initialized so softplus(dt_bias) ≈ U[dt_min, dt_max].
        dt = torch.exp(
            torch.rand(h) * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
            + math.log(cfg.dt_min)
        ).clamp(min=cfg.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_reinit = True  # type: ignore[attr-defined]

        # A_log: scalar per head, sampled in [A_min, A_max] then logged.
        a_min, a_max = cfg.A_init_range
        A = torch.empty(h).uniform_(a_min, a_max)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_reinit = True  # type: ignore[attr-defined]
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # D: per-head skip scalar, like the Mamba-1 D but one value per head.
        self.D = nn.Parameter(torch.ones(h))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        self.norm = RMSNorm(d_in, eps=cfg.norm_eps)
        self.out_proj = nn.Linear(d_in, cfg.d_model, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        cfg = self.cfg

        zxBCdt = self.in_proj(x)  # (b, l, 2*d_in + 2*g*n + h)
        z, xBC, dt = torch.split(
            zxBCdt,
            [cfg.d_inner, cfg.d_inner + 2 * cfg.n_groups * cfg.d_state, cfg.nheads],
            dim=-1,
        )

        xBC = xBC.transpose(1, 2)
        xBC = self.conv1d(xBC)[:, :, :l]
        xBC = xBC.transpose(1, 2)
        xBC = F.silu(xBC)

        x_, B, C = torch.split(
            xBC,
            [cfg.d_inner, cfg.n_groups * cfg.d_state, cfg.n_groups * cfg.d_state],
            dim=-1,
        )

        x_ = x_.view(b, l, cfg.nheads, cfg.headdim)
        B = B.view(b, l, cfg.n_groups, cfg.d_state)
        C = C.view(b, l, cfg.n_groups, cfg.d_state)

        # Per-head softplus dt with learnable bias.
        dt = F.softplus(dt + self.dt_bias)

        A = -torch.exp(self.A_log.float())  # negative real spectrum, (h,)

        y = ssd_sequential_scan(x_, dt, A, B, C)  # (b, l, h, d)
        y = y + x_ * self.D[None, None, :, None]

        y = y.reshape(b, l, cfg.d_inner)
        y = self.norm(y)
        y = y * F.silu(z)

        return self.out_proj(y)


class ResidualBlock(nn.Module):
    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mixer = Mamba2Block(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.norm(x)) + x


def _init_weights(module: nn.Module, initializer_range: float, n_layer: int) -> None:
    if isinstance(module, nn.Linear):
        if not getattr(module.weight, "_no_reinit", False):
            nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if isinstance(module, Mamba2Block):
        with torch.no_grad():
            module.out_proj.weight.normal_(std=initializer_range / math.sqrt(2 * n_layer))


class Mamba2(nn.Module):
    def __init__(self, cfg: Mamba2Config, initializer_range: float = 0.02):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([ResidualBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying
        self.apply(lambda m: _init_weights(m, initializer_range, cfg.n_layer))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(input_ids)[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
