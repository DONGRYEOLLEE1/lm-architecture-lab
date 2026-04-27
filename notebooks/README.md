# Benchmark Scripts

These scripts are intentionally small and forward-pass only.

- `memory_vs_sequence_length.py` measures peak inference memory and saves `figures/memory_vs_sequence_length.png`
- `inference_time_vs_sequence_length.py` measures forward-pass wall-clock time and saves `figures/inference_time_vs_sequence_length.png`
- `parameter_count_comparison.py` compares model sizes and saves `figures/parameter_count_comparison.png`

Run them from the repo root:

```bash
uv run notebooks/memory_vs_sequence_length.py
uv run notebooks/inference_time_vs_sequence_length.py
uv run notebooks/parameter_count_comparison.py
```

If CUDA is unavailable, both scripts fall back to CPU and still produce figures.
