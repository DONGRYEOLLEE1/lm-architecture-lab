"""Pure-PyTorch LFM2 (Liquid Foundation Models 2).

Hybrid backbone: 10 gated short-convolution blocks interleaved with 6 GQA
attention blocks. Each block pairs its mixer with a SwiGLU FFN, all in the
pre-norm / RMSNorm style. Config matches `LiquidAI/LFM2-350M`.

References:
- LFM2 Technical Report, arXiv:2511.23404
- HuggingFace transformers `modeling_lfm2.py`
- LiquidAI/LFM2-350M config: full_attn_idxs = [2, 5, 8, 10, 12, 14]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_FULL_ATTN_IDXS = (2, 5, 8, 10, 12, 14)


@dataclass
class LFM2Config:
    vocab_size: int = 65_536
    hidden_size: int = 1024
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 6656
    conv_L_cache: int = 3
    conv_bias: bool = False
    attn_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-5
    max_position_embeddings: int = 32_768
    full_attn_idxs: tuple[int, ...] = field(default_factory=lambda: DEFAULT_FULL_ATTN_IDXS)
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

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
        if self.conv_L_cache <= 0:
            raise ValueError("conv_L_cache must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        invalid_attn_layers = [
            layer_idx
            for layer_idx in self.full_attn_idxs
            if layer_idx < 0 or layer_idx >= self.num_hidden_layers
        ]
        if invalid_attn_layers:
            raise ValueError(
                f"full_attn_idxs contain out-of-range layers: {invalid_attn_layers}"
            )

    def layer_type(self, layer_idx: int) -> str:
        return "full_attention" if layer_idx in self.full_attn_idxs else "conv"


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


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 1e6, max_seq_len: int = 32_768):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (b, l) → cos/sin: (b, l, head_dim)
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
    # q, k: (b, h, l, d)  ; cos/sin: (b, l, d) → broadcast over heads.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out.to(q.dtype), k_out.to(k.dtype)


class ShortConvBlock(nn.Module):
    """Gated short convolution:  (B⊙x) → DepthwiseConv1d → (C⊙·) → out_proj."""

    def __init__(self, cfg: LFM2Config):
        super().__init__()
        d = cfg.hidden_size
        self.L_cache = cfg.conv_L_cache
        self.in_proj = nn.Linear(d, 3 * d, bias=cfg.conv_bias)
        self.conv = nn.Conv1d(
            in_channels=d,
            out_channels=d,
            kernel_size=cfg.conv_L_cache,
            groups=d,
            padding=cfg.conv_L_cache - 1,
            bias=cfg.conv_bias,
        )
        self.out_proj = nn.Linear(d, d, bias=cfg.conv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        BCx = self.in_proj(x).transpose(-1, -2)  # (b, 3d, l)
        B, C, xx = BCx.chunk(3, dim=-2)
        Bx = B * xx
        conv_out = self.conv(Bx)[..., :l]  # causal crop
        y = C * conv_out
        return self.out_proj(y.transpose(-1, -2).contiguous())


class GQAAttention(nn.Module):
    """Grouped-query attention with QK-RMSNorm and RoPE."""

    def __init__(self, cfg: LFM2Config):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.rep = self.n_heads // self.n_kv
        assert self.n_heads % self.n_kv == 0, "heads must be divisible by kv_heads"

        d = cfg.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=cfg.attn_bias)
        self.k_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=cfg.attn_bias)
        self.v_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=cfg.attn_bias)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=cfg.attn_bias)

        self.q_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        b, l, _ = x.shape
        q = self.q_proj(x).view(b, l, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, l, self.n_kv, self.head_dim)
        v = self.v_proj(x).view(b, l, self.n_kv, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)  # (b, H, l, d)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, l, self.n_heads * self.head_dim)
        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: LFM2Config):
        super().__init__()
        d, h = cfg.hidden_size, cfg.intermediate_size
        self.w1 = nn.Linear(d, h, bias=cfg.mlp_bias)  # gate
        self.w3 = nn.Linear(d, h, bias=cfg.mlp_bias)  # up
        self.w2 = nn.Linear(h, d, bias=cfg.mlp_bias)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: LFM2Config, layer_idx: int):
        super().__init__()
        self.is_attention = cfg.layer_type(layer_idx) == "full_attention"
        self.operator_norm = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.operator: nn.Module = (
            GQAAttention(cfg) if self.is_attention else ShortConvBlock(cfg)
        )
        self.feed_forward = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.operator_norm(x)
        if self.is_attention:
            assert cos is not None and sin is not None
            h = self.operator(h, cos, sin)
        else:
            h = self.operator(h)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class LFM2(nn.Module):
    def __init__(self, cfg: LFM2Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.embedding_norm = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.rope = RotaryEmbedding(
            cfg.head_dim, base=cfg.rope_theta, max_seq_len=cfg.max_position_embeddings
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
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, l = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, -1)

        x = self.embed_tokens(input_ids)
        cos, sin = self.rope(position_ids)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.embedding_norm(x)
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
