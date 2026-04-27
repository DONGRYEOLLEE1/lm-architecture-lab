"""Pure-PyTorch xLSTM (Extended LSTM).

Reference: Beck, Pöppel, Spanring, Auer, Prudnikova, Kopp, Klambauer,
Brandstetter, Hochreiter, "xLSTM: Extended Long Short-Term Memory" (2024).

Two cells stacked into one residual backbone:

  - mLSTM (matrix LSTM): the recurrence carries a (head_dim × head_dim)
    matrix memory C and a head_dim normalizer n. All gates depend only on
    x_t (no h_{t-1} term), so the recurrence is parallelizable in the same
    way as linear attention. We keep a sequential reference here.

  - sLSTM (scalar LSTM): the recurrence carries a per-channel scalar memory
    c, plus a normalizer n. Gates have a recurrent dependency on h_{t-1},
    so this cell is fundamentally sequential — closer to the original LSTM
    than mLSTM is.

Both cells share the **exponential gating + log-space stabilizer** trick:
i_t and f_t are interpreted as log-gates, and the running max state m_t
keeps `exp(log f_t + m_{t-1})` and `exp(log i_t)` in a representable range.
Without that, exponential gates blow up after a few hundred steps.

What we simplify vs the official `xlstm` package:

  - Sequential reference scans (no parallel chunked form for mLSTM).
  - Block-diagonal recurrent matrices in sLSTM, but no "new memory mixing"
    across heads — that mixing is a follow-up trick and the cell is
    interpretable without it.
  - One conv1d kernel size, no separate q/k conv heads.

The point is that the two cells should behave like the paper says they do,
and you can read this file alongside Algorithm 1/2 in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class xLSTMConfig:
    vocab_size: int = 128
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    proj_factor: float = 1.5  # mLSTM up-projection factor
    sLSTM_ffn_factor: float = 4 / 3  # sLSTM post-cell FFN expansion
    conv1d_kernel: int = 4
    layer_pattern: tuple[str, ...] = field(
        default_factory=lambda: ("m", "s", "m", "s")
    )
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    pad_vocab_size_multiple: int = 8
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.proj_factor <= 0:
            raise ValueError("proj_factor must be positive")
        if self.conv1d_kernel <= 0:
            raise ValueError("conv1d_kernel must be positive")
        if len(self.layer_pattern) != self.num_layers:
            raise ValueError(
                f"layer_pattern length ({len(self.layer_pattern)}) must match num_layers ({self.num_layers})"
            )
        if any(p not in ("m", "s") for p in self.layer_pattern):
            raise ValueError("layer_pattern entries must be 'm' (mLSTM) or 's' (sLSTM)")
        if self.pad_vocab_size_multiple > 0 and (
            self.vocab_size % self.pad_vocab_size_multiple != 0
        ):
            self.vocab_size += self.pad_vocab_size_multiple - (
                self.vocab_size % self.pad_vocab_size_multiple
            )

        proj_d = int(self.hidden_size * self.proj_factor)
        if proj_d % self.num_heads != 0:
            # Round up to the nearest multiple of num_heads to keep heads even.
            proj_d = ((proj_d + self.num_heads - 1) // self.num_heads) * self.num_heads
        self.proj_dim = proj_d


def _causal_depthwise_conv(channels: int, kernel: int, bias: bool = True) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel,
        groups=channels,
        padding=kernel - 1,
        bias=bias,
    )


def mlstm_sequential_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i_pre: torch.Tensor,
    f_pre: torch.Tensor,
    o_pre: torch.Tensor,
) -> torch.Tensor:
    """Sequential reference for the mLSTM recurrence (Algorithm 2).

    Shapes:
        q, k, v        : (b, l, h, dh)
        i_pre, f_pre   : (b, l, h)        — pre-activations interpreted as log-gates
        o_pre          : (b, l, h)        — pre-activation of sigmoid output gate

    State carried per head:
        C : (b, h, dh, dh)   — outer-product memory
        n : (b, h, dh)        — normalizer
        m : (b, h)            — log-space stabilizer

    Returns:
        h : (b, l, h, dh)
    """
    b, l, h, dh = q.shape

    C = q.new_zeros(b, h, dh, dh)
    n = q.new_zeros(b, h, dh)
    m = q.new_full((b, h), -1e9)  # start very negative so first step is unclipped

    out = []
    for t in range(l):
        log_i = i_pre[:, t]                      # (b, h)
        log_f = f_pre[:, t]                      # (b, h)
        m_new = torch.maximum(log_f + m, log_i)  # (b, h)
        i_eff = torch.exp(log_i - m_new)
        f_eff = torch.exp(log_f + m - m_new)

        v_t = v[:, t]                            # (b, h, dh)
        k_t = k[:, t]
        q_t = q[:, t]

        # C update: f * C + i * v ⊗ k
        update = (
            (i_eff[:, :, None] * v_t)[:, :, :, None]
            * k_t[:, :, None, :]
        )                                         # (b, h, dh, dh)
        C = f_eff[:, :, None, None] * C + update
        n = f_eff[:, :, None] * n + i_eff[:, :, None] * k_t  # (b, h, dh)

        h_num = (C * q_t[:, :, None, :]).sum(dim=-1)        # (b, h, dh)
        denom = (n * q_t).sum(dim=-1).abs().clamp(min=1.0)   # (b, h)
        o_act = torch.sigmoid(o_pre[:, t])                   # (b, h)
        h_t = o_act[:, :, None] * h_num / denom[:, :, None]
        out.append(h_t)
        m = m_new

    return torch.stack(out, dim=1)


def slstm_sequential_scan(
    x_i: torch.Tensor,
    x_f: torch.Tensor,
    x_z: torch.Tensor,
    x_o: torch.Tensor,
    r_i: torch.Tensor,
    r_f: torch.Tensor,
    r_z: torch.Tensor,
    r_o: torch.Tensor,
) -> torch.Tensor:
    """Sequential reference for the sLSTM recurrence (Algorithm 1).

    Inputs:
        x_*  : (b, l, h, dh)   — gate pre-activations contributed by x_t
        r_*  : (h, dh, dh)      — block-diagonal recurrent matrices

    State per head, per channel:
        c : (b, h, dh)   cell scalar memory
        n : (b, h, dh)   normalizer
        m : (b, h, dh)   stabilizer

    Returns:
        h : (b, l, h, dh)
    """
    b, l, h, dh = x_i.shape

    c = x_i.new_zeros(b, h, dh)
    n = x_i.new_zeros(b, h, dh)
    m = x_i.new_full((b, h, dh), -1e9)
    h_prev = x_i.new_zeros(b, h, dh)

    out = []
    for t in range(l):
        # einsum('bhd,hed->bhe', h_prev, R)
        ri_t = torch.einsum("bhd,hed->bhe", h_prev, r_i)
        rf_t = torch.einsum("bhd,hed->bhe", h_prev, r_f)
        rz_t = torch.einsum("bhd,hed->bhe", h_prev, r_z)
        ro_t = torch.einsum("bhd,hed->bhe", h_prev, r_o)

        log_i = x_i[:, t] + ri_t
        log_f = x_f[:, t] + rf_t
        z_t = torch.tanh(x_z[:, t] + rz_t)
        o_pre = x_o[:, t] + ro_t

        m_new = torch.maximum(log_f + m, log_i)
        i_eff = torch.exp(log_i - m_new)
        f_eff = torch.exp(log_f + m - m_new)
        o_eff = torch.sigmoid(o_pre)

        c = f_eff * c + i_eff * z_t
        n = f_eff * n + i_eff
        h_t = o_eff * c / n.clamp(min=1e-6)
        m = m_new

        h_prev = h_t
        out.append(h_t)

    return torch.stack(out, dim=1)


class mLSTMBlock(nn.Module):
    """Pre-norm residual block hosting an mLSTM cell.

    Layout: norm → up-projection (gate path z + main path) → causal conv on
    main → q/k/v projections + i/f/o gate projections → mLSTM scan →
    GroupNorm over heads → learnable skip + SiLU(z) gate → down-projection.
    """

    def __init__(self, cfg: xLSTMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        pd = cfg.proj_dim
        h = cfg.num_heads
        dh = pd // h
        self.h, self.dh, self.pd = h, dh, pd

        self.norm = nn.LayerNorm(d, eps=cfg.norm_eps)
        self.up_proj = nn.Linear(d, 2 * pd, bias=False)
        self.conv = _causal_depthwise_conv(pd, cfg.conv1d_kernel)

        self.q_proj = nn.Linear(pd, pd, bias=False)
        self.k_proj = nn.Linear(pd, pd, bias=False)
        self.v_proj = nn.Linear(pd, pd, bias=False)
        self.i_proj = nn.Linear(pd, h, bias=True)
        self.f_proj = nn.Linear(pd, h, bias=True)
        self.o_proj = nn.Linear(pd, h, bias=True)

        self.group_norm = nn.GroupNorm(h, pd, eps=cfg.norm_eps)
        self.learnable_skip = nn.Parameter(torch.ones(pd))
        self.down_proj = nn.Linear(pd, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        residual = x

        h = self.norm(x)
        zm = self.up_proj(h)                   # (b, l, 2*pd)
        z, main = zm.chunk(2, dim=-1)          # gate path z, main path

        m = self.conv(main.transpose(1, 2))[..., :l].transpose(1, 2)
        m = F.silu(m)

        q = self.q_proj(m).view(b, l, self.h, self.dh)
        k = self.k_proj(m).view(b, l, self.h, self.dh) / (self.dh ** 0.5)
        v = self.v_proj(m).view(b, l, self.h, self.dh)
        i_pre = self.i_proj(m)
        f_pre = self.f_proj(m)
        o_pre = self.o_proj(m)

        out = mlstm_sequential_scan(q, k, v, i_pre, f_pre, o_pre)  # (b, l, h, dh)
        out = out.reshape(b, l, self.pd)

        # GroupNorm expects (B, C, L)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)

        out = out + self.learnable_skip * m
        out = out * F.silu(z)

        return residual + self.down_proj(out)


class sLSTMBlock(nn.Module):
    """Pre-norm residual block hosting an sLSTM cell + a small post-FFN.

    Layout: norm → causal conv → linear(x → 4·hidden) for (i, f, z, o)
    contributions → block-diagonal recurrent matrices → sLSTM scan →
    GroupNorm → GeGLU FFN → residual.
    """

    def __init__(self, cfg: xLSTMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        h = cfg.num_heads
        dh = d // h
        self.h, self.dh = h, dh

        self.norm = nn.LayerNorm(d, eps=cfg.norm_eps)
        self.conv = _causal_depthwise_conv(d, cfg.conv1d_kernel)
        self.x_proj = nn.Linear(d, 4 * d, bias=True)

        # Block-diagonal recurrent matrices, one (dh, dh) per head per gate.
        std = cfg.initializer_range
        self.r_i = nn.Parameter(torch.randn(h, dh, dh) * std)
        self.r_f = nn.Parameter(torch.randn(h, dh, dh) * std)
        self.r_z = nn.Parameter(torch.randn(h, dh, dh) * std)
        self.r_o = nn.Parameter(torch.randn(h, dh, dh) * std)

        self.group_norm = nn.GroupNorm(h, d, eps=cfg.norm_eps)

        ffn_dim = max(int(d * cfg.sLSTM_ffn_factor), h)
        if ffn_dim % h != 0:
            ffn_dim = ((ffn_dim + h - 1) // h) * h
        self.ffn_up = nn.Linear(d, 2 * ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        residual = x

        h_in = self.norm(x)
        h_conv = self.conv(h_in.transpose(1, 2))[..., :l].transpose(1, 2)
        h_conv = F.silu(h_conv)

        gates = self.x_proj(h_conv)                # (b, l, 4*d)
        xi, xf, xz, xo = gates.chunk(4, dim=-1)

        xi = xi.view(b, l, self.h, self.dh)
        xf = xf.view(b, l, self.h, self.dh)
        xz = xz.view(b, l, self.h, self.dh)
        xo = xo.view(b, l, self.h, self.dh)

        out = slstm_sequential_scan(xi, xf, xz, xo, self.r_i, self.r_f, self.r_z, self.r_o)
        out = out.reshape(b, l, self.h * self.dh)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)

        ffn = self.ffn_up(out)
        a, gate = ffn.chunk(2, dim=-1)
        ffn_out = self.ffn_down(F.gelu(a, approximate="tanh") * gate)

        return residual + ffn_out


class xLSTM(nn.Module):
    def __init__(self, cfg: xLSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [
                mLSTMBlock(cfg) if kind == "m" else sLSTMBlock(cfg)
                for kind in cfg.layer_pattern
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=std)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
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
