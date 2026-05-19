# Multi-Lag Phase-LS Doppler Paper Pipeline

This folder contains the reproducible analysis pipeline for the multi-lag
phase-regression Doppler work.

See `research_ideas.md` for tested estimator variants and next experiments.

The method being evaluated is:

```text
R_k(x, z) = mean_t s_{t+k}(x, z) conj(s_t(x, z))
phi_k = unwrap(angle(R_k))
w_k = |R_k|

omega_hat = argmin_omega sum_k w_k (phi_k - omega k)^2
```

The signed phase-LS image uses `omega_hat`, autocorrelation magnitude, and the
phase-linearity fit quality. It is compared against Kasai, higher-lag
autocorrelation, lag-product TMAS-style maps, and angular coherence weighting.

## Commands

Run the fast synthetic benchmark:

```bash
make -C paper synthetic
```

Run the synthetic research sweep for candidate phase estimators:

```bash
make -C paper research-sweep
```

Run a quick single-acquisition hard-drive smoke test:

```bash
make -C paper quick-kenny
```

Run the full Kenny hard-drive report:

```bash
make -C paper kenny
```

Run the full all-variations Kenny report:

```bash
make -C paper kenny-all-variations
```

Run the angular coherence weighting parameter sweep:

```bash
make -C paper angular-cf-sweep
```

This defaults to one acquisition because it runs SVD Doppler for each CF
variant. It uses the saved Kenny Doppler settings:
`low_cutoff=0.05`, `mean_subtract=False`, `method=fast`. For a slower
multi-acquisition run:

```bash
make -C paper angular-cf-sweep ANGULAR_CF_N_ACQ=8
```

Run everything:

```bash
make -C paper all
```

The hard-drive report expects:

```text
/Volumes/Extreme SSD/data/ultratrace_Kenny_lev_2026-04-27_16:42:18.h5
```

Override it with:

```bash
make -C paper kenny DATA="/path/to/ultratrace.h5"
```

## Outputs

Synthetic benchmark:

```text
paper/outputs/synthetic/synthetic_phase_ls_benchmark.png
paper/outputs/synthetic/synthetic_phase_ls_benchmark.pdf
paper/outputs/synthetic/synthetic_phase_ls_metrics.csv
paper/outputs/synthetic/synthetic_phase_ls_summary.json
```

Research sweep:

```text
paper/outputs/research_sweep/phase_ls_research_sweep.png
paper/outputs/research_sweep/phase_ls_research_sweep.pdf
paper/outputs/research_sweep/phase_ls_research_sweep.csv
paper/outputs/research_sweep/phase_ls_research_sweep.summary.json
```

Hard-drive report:

```text
paper/outputs/kenny_042716/kenny_042716_tmas_phase_ls_comparison.pdf
paper/outputs/kenny_042716/kenny_042716_all_variations.pdf
paper/outputs/kenny_042716/kenny_042716_angular_cf_parameter_sweep.pdf
paper/outputs/kenny_042716/kenny_042716_tmas_phase_ls_comparison.summary.json
```

## Notes

The paper scripts default to `CATERPILLAR_DISABLE_METAL=1` so they can run
with the Torch beamformer in non-interactive or sandboxed environments. This
does not change realtime defaults unless that environment variable is set.

The first paper claim to test is not that TMAS itself is new. The claim is that
autocorrelation-weighted multi-lag phase regression provides a signed
Doppler-like contrast image that preserves coherent flow while rejecting pixels
whose autocorrelation phase is not linear in lag.
