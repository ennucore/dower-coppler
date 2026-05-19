# Dower Coppler Paper

Standalone paper repository for the Dower Coppler manuscript. This repo is intentionally separate from `~/dev/caterpillar`; the parent caterpillar repository is not used for commits here.

## Reproducing figures

The derived NPZ inputs used by the paper are committed under `data/`. Regenerate the paper figures with:

```bash
python3 scripts/generate_paper_figures.py
latexmk -pdf -interaction=nonstopmode paper.tex
```

The generated figures are written to `outputs/paper_figures/`. The current compiled manuscript is `paper.pdf`.

## Provenance

See `reproducibility/manifest.json` for hashes, source ultratrace paths, acquisition ranges, and notes for each NPZ input.

Relevant source scripts from the caterpillar worktree are backed up under `script_snapshots/`, including:

- `caterpillar_scripts/doppler_cnr_viewer.py`
- `caterpillar_scripts/beamform_sep21_full2d_fine_xz.py`
- `caterpillar_scripts/beamform_sep21_full2d_fine_xz_nosdk.py`
- `caterpillar_scripts/process_compound_h5_to_viewer.py`
- `caterpillar_scripts/compute_sep21_cached_lag1_color.py`
- `caterpillar_scripts/recompute_sep21_cached_compound_mid8.py`
- `caterpillar_imaging/doppler.py`

Figure 4 CNR values use the largest inscribed circle inside each exported segmented ROI, plus one shared background rectangle. The code path is `load_region_export()` -> `largest_inscribed_circle()` in `scripts/generate_paper_figures.py`.
