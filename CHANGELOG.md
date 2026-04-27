# Changelog

## 2026-04-27

- Fixed `dt_proj.weight` double initialization in `MambaBlock`: the custom dt-scale init (`dt_rank**-0.5 * dt_scale`) was being silently overwritten by the module-level `_init_weights` hook. Added `._no_reinit = True` flag after the custom init to preserve the paper's initialization. The selective time-step delta Δ is load-bearing for SSM stability — incorrect scaling destabilizes the model's ability to selectively update recurrent state.

## 2026-04-24

- Fixed an off-by-one bug in all three copy-task batch builders so `seq_len` now matches the returned training tensors exactly.
- Added RoPE validation in LFM2 and Gemma 4 so invalid head sizes or out-of-range positions fail loudly instead of silently producing incorrect phases.
- Made Mamba RMSNorm run its variance reduction in float32 and restore the input dtype, which keeps mixed-precision behavior aligned with the attention models.
- Added lightweight forward-pass benchmark scripts under `notebooks/` for memory and wall-clock comparisons across the available architectures.
