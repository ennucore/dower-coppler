# Dower Coppler Paper

Standalone paper repository for the Dower Coppler manuscript. This repo is intentionally separate from `~/dev/caterpillar`; the parent caterpillar repository is not used for commits here.

## Reproducing figures

The derived NPZ inputs used by the paper are committed under `data/`. Regenerate the paper figures with:

```bash
python3 scripts/generate_paper_figures.py
latexmk -pdf -interaction=nonstopmode paper.tex
```

The generated figures are written to `outputs/paper_figures/`. The current compiled manuscript is `paper.pdf`.

The "different recording" figure is configurable without editing the paper:

```bash
python3 scripts/generate_paper_figures.py \
  --external-recording-data data/bt24480388_2026-05-18_152605_txel0_h5_row-1_fine_xz_y-3p5to3p5mm_10elev_all20.npz \
  --external-recording-plane -1
```

`--external-recording-plane -1` selects the middle elevation plane.

## Provenance

See `reproducibility/manifest.json` for hashes, source ultratrace paths, acquisition ranges, and notes for each NPZ input.

For the primary September 21 source ultratrace
`/mnt/pocampus/lev/ultratrace_Head_monster_2025-09-21_21-32-01_y-20to20mm_30elev.h5`,
the HDF5 acquisition config stores `num_angles=5` and
`num_loops=700`, with raw `iq_frames` shaped
`(702, 5, 8, 134, 88)` for acquisition 200. The two extra loops are
noise loops. The runtime metadata reports an empirical transmit pulse
PRF of 1244.224 Hz for acquisition 200 and a mean of 1245.932 Hz over
acquisitions 200-399, so the compounded slow-time cadence used for
phase-to-velocity conversion is PRF / 5, approximately 249 Hz.

Relevant source scripts from the caterpillar worktree are backed up under `script_snapshots/`, including:

- `caterpillar_scripts/doppler_cnr_viewer.py`
- `caterpillar_scripts/beamform_sep21_full2d_fine_xz.py`
- `caterpillar_scripts/beamform_sep21_full2d_fine_xz_nosdk.py`
- `caterpillar_scripts/process_compound_h5_to_viewer.py`
- `caterpillar_scripts/compute_sep21_cached_lag1_color.py`
- `caterpillar_scripts/recompute_sep21_cached_compound_mid8.py`
- `caterpillar_imaging/doppler.py`

Figure 4 CNR values use the largest inscribed circle inside each exported segmented ROI, plus one shared background rectangle. The code path is `load_region_export()` -> `largest_inscribed_circle()` in `scripts/generate_paper_figures.py`.
