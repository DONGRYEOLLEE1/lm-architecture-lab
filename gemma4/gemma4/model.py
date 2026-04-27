"""Pure-PyTorch Gemma 4 (speculative reconstruction).

Gemma 4 has not released a public technical report at the time of writing,
so this implementation synthesizes the well-documented pieces of the Gemma
lineage into a single coherent backbone:

- Per-Layer Embeddings (PLE) from Gemma 3n: each transformer layer owns a
  small embedding table that adds a token-specific signal at that depth.
- Sliding-window + global attention interleaving from Gemma 3 (5 local
  layers per 1 global by default).
- QK-RMSNorm and Gemma-style `(1 + weight)` RMSNorm from Gemma 2/3.
- Pre- and post-sublayer normalization ("sandwich norm") around both
  attention and FFN.
- GeGLU feed-forward (tanh-approx GELU gate) — Gemma uses GeGLU, not SwiGLU.
- Dual RoPE base (large theta for global layers, small theta for local).
- Embedding scaling by sqrt(hidden_size).

References:
- Gemma 3 technical report (DeepMind, 2025)
- Gemma 3n model card / developer guide (PLE, MatFormer)
- HuggingFace `transformers` `modeling_gemma3.py` / `modeling_gemma3n.py`
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Gemma4Config:
    vocab_size: int = 262_144
    hidden_size: int = 2048
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    intermediate_size: int = 16_384
    # RoPE — Gemma 3 uses two bases, large for global, small for local.
    rope_theta_global: float = 1_000_000.0
    rope_theta_local: float = 10_000.0
    max_position_embeddings: int = 32_768
    # Attention pattern.
    sliding_window: int = 1024
    global_attn_every_n: int = 6  # 1-indexed: every 6th layer is global.
    attn_bias: bool = False
    # MLP.
    mlp_bias: bool = False
    # Norm.
    norm_eps: float = 1e-6
    # Per-Layer Embeddings.
    per_layer_input_dim: int = 256
    # Init / output.
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    embed_scale: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        if self.global_attn_every_n <= 0:
            raise ValueError("global_attn_every_n must be positive")
        if self.per_layer_input_dim <= 0:
            raise ValueError("per_layer_input_dim must be positive")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

    def is_global_attn(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self.global_attn_every_n == 0


class Gemma4RMSNorm(nn.Module):
    """RMSNorm with (1 + weight) gain — Gemma-specific, zero-init stable."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight)).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float, max_seq_len: int):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.numel() and int(position_ids.max()) >= self.max_seq_len:
            raise ValueError(
                f"position_ids exceed max_seq_len={self.max_seq_len}"
            )
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out.to(q.dtype), k_out.to(k.dtype)


def _sliding_causal_mask(
    seq_len: int, window: int, device: torch.device
) -> torch.Tensor:
    """Bool mask: True where token i may attend to token j (causal + windowed)."""
    i = torch.arange(seq_len, device=device)[:, None]
    j = torch.arange(seq_len, device=device)[None, :]
    return (j <= i) & ((i - j) < window)


class Gemma4Attention(nn.Module):
    """GQA with QK-RMSNorm. Global layers see full context; local layers use a
    sliding window (Gemma 3 pattern)."""

    def __init__(self, cfg: Gemma4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_global = cfg.is_global_attn(layer_idx)
        self.sliding_window = None if self.is_global else cfg.sliding_window

        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.rep = self.n_heads // self.n_kv
        assert self.n_heads % self.n_kv == 0, "heads must be divisible by kv_heads"

        d = cfg.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=cfg.attn_bias)
        self.k_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=cfg.attn_bias)
        self.v_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=cfg.attn_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=cfg.attn_bias)

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=cfg.norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=cfg.norm_eps)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        b, l, _ = x.shape
        q = self.q_proj(x).view(b, l, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, l, self.n_kv, self.head_dim)
        v = self.v_proj(x).view(b, l, self.n_kv, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)

        if self.sliding_window is not None and l > 1:
            mask = _sliding_causal_mask(l, self.sliding_window, q.device)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(b, l, self.n_heads * self.head_dim)
        return self.o_proj(y)


class GeGLU(nn.Module):
    """Gemma's gated FFN: gate uses tanh-approx GELU (matches Gemma 2/3 ref)."""

    def __init__(self, cfg: Gemma4Config):
        super().__init__()
        d, h = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(d, h, bias=cfg.mlp_bias)
        self.up_proj = nn.Linear(d, h, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(h, d, bias=cfg.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


class PerLayerEmbeddings(nn.Module):
    """Per-Layer Embeddings (PLE) from Gemma 3n.

    Each transformer layer owns a dedicated embedding table of small dimension
    (`per_layer_input_dim`). At layer i, tokens look up into table i, pass
    through a per-layer projection, and are injected additively into the
    residual stream. The per-layer parameters are lookup-only (no matmul in
    the hot path), so in a production deployment they can sit in CPU/flash
    while only the shared hidden stream runs on the accelerator.
    """

    def __init__(self, cfg: Gemma4Config):
        super().__init__()
        self.cfg = cfg
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cfg.vocab_size, cfg.per_layer_input_dim)
                for _ in range(cfg.num_hidden_layers)
            ]
        )
        self.projections = nn.ModuleList(
            [
                nn.Linear(cfg.per_layer_input_dim, cfg.hidden_size, bias=False)
                for _ in range(cfg.num_hidden_layers)
            ]
        )

    def forward(self, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        e = self.embeddings[layer_idx](input_ids)
        return self.projections[layer_idx](F.gelu(e, approximate="tanh"))


class DecoderLayer(nn.Module):
    """Sandwich-norm block: pre+post RMSNorm around attention and FFN, then
    inject the per-layer embedding residual."""

    def __init__(self, cfg: Gemma4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_norm = Gemma4RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.post_attn_norm = Gemma4RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.pre_ffn_norm = Gemma4RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.post_ffn_norm = Gemma4RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.attn = Gemma4Attention(cfg, layer_idx)
        self.mlp = GeGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ple: torch.Tensor,
    ) -> torch.Tensor:
        h = self.attn(self.input_norm(x), cos, sin)
        x = x + self.post_attn_norm(h)
        h = self.mlp(self.pre_ffn_norm(x))
        x = x + self.post_ffn_norm(h)
        return x + ple


class Gemma4(nn.Module):
    def __init__(self, cfg: Gemma4Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.per_layer_embeddings = PerLayerEmbeddings(cfg)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.final_norm = Gemma4RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.rope_global = RotaryEmbedding(
            cfg.head_dim, cfg.rope_theta_global, cfg.max_position_embeddings
        )
        self.rope_local = RotaryEmbedding(
            cfg.head_dim, cfg.rope_theta_local, cfg.max_position_embeddings
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=std)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, l = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, -1)

        x = self.embed_tokens(input_ids)
        if self.cfg.embed_scale:
            x = x * (self.cfg.hidden_size**0.5)

        cos_g, sin_g = self.rope_global(position_ids)
        cos_l, sin_l = self.rope_local(position_ids)

        for i, layer in enumerate(self.layers):
            cos, sin = (cos_g, sin_g) if self.cfg.is_global_attn(i) else (cos_l, sin_l)
            ple = self.per_layer_embeddings(input_ids, i)
            x = layer(x, cos, sin, ple)

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
