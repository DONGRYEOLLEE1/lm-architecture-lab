# CLAUDE.md — Project Context for AI Assistants

## Purpose

This repo is a **research portfolio project** by an AI engineer. The goal is to understand modern sequence modeling architectures by implementing them from scratch in pure PyTorch. It is not a production system, not a training framework, and not meant to be installed as a library.

Intended audiences: other AI engineers, researchers, and anyone reading the code alongside the original papers.

---

## Repo Scope and Constraints

### What is in scope
- Pure-PyTorch model implementations faithful to their source papers
- Minimal smoke tests and toy training loops that run on CPU in seconds
- Lightweight experiments comparing architectural properties (memory, speed, convergence)
- Clear documentation of what each implementation faithfully reproduces vs. simplifies

### What is explicitly out of scope
- Distributed training, FSDP, gradient checkpointing
- Custom CUDA kernels (e.g., Mamba's official `selective_scan_cuda`)
- Inference optimization (KV caching, speculative decoding, quantization)
- Dataset pipelines, tokenization, evaluation harnesses
- Checkpoint loading from pretrained weights
- Production-grade packaging or installability

---

## Implementation Philosophy

**Readability over performance.** Every file should be readable alongside the paper. If there's a trade-off between a faster implementation and one that maps cleanly to the paper's equations, choose the paper-faithful version and note the trade-off in a comment.

**Minimal abstraction.** Avoid shared base classes, registry patterns, or frameworks that obscure what each model is doing. Three similar `forward()` methods across models is fine — a premature shared base class is not.

**Toy configs for smoke tests.** Each `main.py` uses a drastically reduced config (hidden_size ~128, layers ~4-6) so tests run on CPU in under 10 seconds. The full-scale config constants in each model file exist as documentation, not as defaults to actually run.

**No comments that explain what the code does.** Comments should explain WHY: a non-obvious initialization, a paper-specific design choice, a known approximation, or a constraint from the reference implementation.

---

## Package Layout

Each model lives in its own subdirectory with a nested package:

```
gemma4/
  gemma4/       ← actual package (has __init__.py)
    model.py
    __init__.py
  main.py       ← smoke test; run as: uv run gemma4/main.py
```

Running `uv run <model>/main.py` from the repo root works.
Running `uv run python -m <model>.main` from the repo root does NOT work — the outer directory is a namespace package with no `__init__.py`. This is a known limitation, not an oversight worth fixing unless the repo gains actual installation support.

---

## Execution Rules

**Always use `uv run` to execute Python scripts.** Never use `python ...`, `python3 ...`, or `./.venv/bin/python ...` directly.

```bash
# correct
uv run main.py
uv run notebooks/memory_vs_sequence_length.py
uv run pytest

# wrong — do not use
python main.py
python3 main.py
./.venv/bin/python main.py
```

This applies to all scripts, smoke tests, notebooks, and any command suggested in documentation or comments.

---

## What to Work On (Priority Order)

When making improvements to this repo, follow this priority order:

1. **Correctness of documented behavior** — if a comment says "faithful to paper X," the code must actually be faithful. Fix quietly discrepant initializations (e.g., `dt_proj.weight`).
2. **Experiment notebooks** — `notebooks/` additions that visualize architectural differences (memory scaling, attention patterns, SSM state dynamics).
3. **README clarity** — making the speculative vs. faithful distinction more visible per-model.
4. **New model implementations** — only add if the architecture makes a meaningfully different design bet than existing models.

Do NOT add:
- Training infra (dataloaders, logging, checkpointing)
- Packaging / pyproject `[scripts]` entrypoints
- CI/CD beyond a simple `pytest` smoke test
- Any model that would require a GPU to run its smoke test
