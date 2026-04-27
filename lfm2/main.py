"""Smoke test and a toy training loop for the PyTorch LFM2 implementation.

Uses a tiny LFM2 config so the model runs quickly on CPU. The copy-with-
separator task exercises the attention layers (long-range lookup across the
SEP token), while the shift task sanity-checks end-to-end plumbing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lfm2 import LFM2, LFM2Config


def tiny_config(vocab_size: int = 64) -> LFM2Config:
    """Small LFM2: same block pattern shape as 350M but scaled down for CPU."""
    return LFM2Config(
        vocab_size=vocab_size,
        hidden_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=384,
        conv_L_cache=3,
        full_attn_idxs=(2, 5),  # 2 attention blocks among 8 layers
        max_position_embeddings=256,
    )


def make_shift_batch(
    batch: int, seq_len: int, vocab: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    y = (x + 1) % vocab
    return x, y


def make_copy_batch(
    batch: int, seq_len: int, vocab: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len < 2:
        raise ValueError("copy task requires seq_len >= 2")
    sep = vocab - 1
    prefix_len = seq_len // 2
    prefix = torch.randint(0, vocab - 1, (batch, prefix_len), device=device)
    sep_col = torch.full((batch, 1), sep, device=device, dtype=prefix.dtype)
    full = torch.cat([prefix, sep_col, prefix], dim=1)
    if full.size(1) < seq_len + 1:
        # Pad odd lengths with one extra separator so the shifted batch keeps exactly seq_len steps.
        full = torch.cat([full, sep_col], dim=1)
    return full[:, :-1], full[:, 1:]


def smoke_test() -> None:
    torch.manual_seed(0)
    cfg = tiny_config(vocab_size=32)
    model = LFM2(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    logits = model(x)
    assert logits.shape == (2, 24, cfg.vocab_size), logits.shape
    logits.sum().backward()
    layer_types = [cfg.layer_type(i) for i in range(cfg.num_hidden_layers)]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] ok — logits={tuple(logits.shape)}  params={n_params:,}")
    print(f"[smoke] layer types: {layer_types}")


def train(task: str, steps: int, batch: int = 32, seq_len: int = 21) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = tiny_config(vocab_size=33)
    model = LFM2(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    sampler = {"shift": make_shift_batch, "copy": make_copy_batch}[task]
    print(
        f"[train:{task}] device={device.type}  "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    for step in range(1, steps + 1):
        x, y = sampler(batch, seq_len, cfg.vocab_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 50 == 0:
            acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"step {step:>4d}  loss={loss.item():.4f}  acc={acc:.3f}")

    model.eval()
    with torch.no_grad():
        x, y = sampler(1, seq_len, cfg.vocab_size, device)
        pred = model(x).argmax(-1)
    print("\ninput :", x[0].tolist())
    print("target:", y[0].tolist())
    print("pred  :", pred[0].tolist())


if __name__ == "__main__":
    smoke_test()
    print()
    train("shift", steps=200)
    print()
    train("copy", steps=500, seq_len=21)
