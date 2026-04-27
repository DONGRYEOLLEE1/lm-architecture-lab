"""Pure-PyTorch Mamba (Selective State Space Model).

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
State Spaces" (2023). Architecture mirrors the official `mamba_simple.py`
but uses a sequential selective_scan so no CUDA kernels are required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class ModelArgs:
    d_model: int = 256
    n_layer: int = 4
    vocab_size: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | str = "auto"
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
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
        if self.pad_vocab_size_multiple <= 0:
            raise ValueError("pad_vocab_size_multiple must be positive")
        self.d_inner = self.expand * self.d_model
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        elif not isinstance(self.dt_rank, int) or self.dt_rank <= 0:
            raise ValueError("dt_rank must be a positive integer or 'auto'")
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += self.pad_vocab_size_multiple - (
                self.vocab_size % self.pad_vocab_size_multiple
            )


def _init_weights(module: nn.Module, initializer_range: float, n_layer: int) -> None:
    """GPT-NeoX-style init used by the official Mamba repo."""
    if isinstance(module, nn.Linear):
        if not getattr(module.weight, "_no_reinit", False):
            nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    # Rescale residual-out projection by 1/sqrt(2 * n_layer) — GPT-2 trick.
    if isinstance(module, MambaBlock):
        with torch.no_grad():
            module.out_proj.weight.normal_(std=initializer_range / math.sqrt(2 * n_layer))


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


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Sequential selective scan (Algorithm 2 in the Mamba paper).

    Shapes:
        u     : (b, l, d_in)
        delta : (b, l, d_in)
        A     : (d_in, n)
        B, C  : (b, l, n)
        D     : (d_in,)
    """
    b, l, d_in = u.shape
    n = A.shape[1]

    # Discretize continuous parameters (zero-order hold).
    deltaA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
    deltaB_u = torch.einsum("bld,bln,bld->bldn", delta, B, u)

    x = u.new_zeros((b, d_in, n))
    ys = []
    for t in range(l):
        x = deltaA[:, t] * x + deltaB_u[:, t]
        y = torch.einsum("bdn,bn->bd", x, C[:, t])
        ys.append(y)
    y = torch.stack(ys, dim=1)  # (b, l, d_in)
    return y + u * D


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        d_in = args.d_inner

        self.in_proj = nn.Linear(args.d_model, d_in * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=d_in,
            out_channels=d_in,
            kernel_size=args.d_conv,
            groups=d_in,
            padding=args.d_conv - 1,
            bias=args.conv_bias,
        )

        # x -> (dt, B, C)
        self.x_proj = nn.Linear(d_in, args.dt_rank + args.d_state * 2, bias=False)
        # low-rank dt projection
        self.dt_proj = nn.Linear(args.dt_rank, d_in, bias=True)

        # dt bias initialization so that softplus(bias) ~ U[dt_min, dt_max].
        dt_init_std = args.dt_rank**-0.5 * args.dt_scale
        if args.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif args.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(args.dt_init)
        self.dt_proj.weight._no_reinit = True  # type: ignore[attr-defined]

        dt = torch.exp(
            torch.rand(d_in) * (math.log(args.dt_max) - math.log(args.dt_min))
            + math.log(args.dt_min)
        ).clamp(min=args.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

        # S4D-real init for A: A_log = log(n), broadcast over d_in.
        A = repeat(torch.arange(1, args.d_state + 1, dtype=torch.float32), "n -> d n", d=d_in)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        self.D = nn.Parameter(torch.ones(d_in))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        self.out_proj = nn.Linear(d_in, args.d_model, bias=args.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape

        x_and_res = self.in_proj(x)  # (b, l, 2*d_in)
        x, res = x_and_res.chunk(2, dim=-1)

        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        A = -torch.exp(self.A_log.float())  # (d_in, n) — negative real spectrum
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2n)
        delta, B, C = x_dbl.split(
            [self.args.dt_rank, self.args.d_state, self.args.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        return selective_scan(x, delta, A, B, C, D)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm = RMSNorm(args.d_model)
        self.mixer = MambaBlock(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.norm(x)) + x


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs, initializer_range: float = 0.02):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying
        self.apply(lambda m: _init_weights(m, initializer_range, args.n_layer))

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
