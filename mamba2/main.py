"""Smoke-test & toy training loop for the PyTorch Mamba-2 implementation.

Same skeleton as `mamba/main.py` so the two models can be compared at
identical scale. The shift task isolates pointwise mapping; the copy task
forces the SSM state to actually carry information across the separator.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mamba2 import Mamba2, Mamba2Config


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
        full = torch.cat([full, sep_col], dim=1)
    return full[:, :-1], full[:, 1:]


def smoke_test() -> None:
    torch.manual_seed(0)
    cfg = Mamba2Config(
        d_model=64, n_layer=2, vocab_size=32, d_state=32, headdim=16, n_groups=1
    )
    model = Mamba2(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, cfg.vocab_size), logits.shape
    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(g.abs().sum() > 0 for g in grads[:5])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] ok — logits={tuple(logits.shape)}  params={n_params:,}")


def train(
    task: str = "shift",
    steps: int = 300,
    batch: int = 32,
    seq_len: int = 16,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Mamba2Config(
        d_model=128, n_layer=2, vocab_size=17, d_state=32, headdim=32, n_groups=1
    )
    model = Mamba2(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    sampler = {"shift": make_shift_batch, "copy": make_copy_batch}[task]
    print(f"[train:{task}] device={device.type}  params={sum(p.numel() for p in model.parameters()):,}")

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
    train(task="shift", steps=300)
