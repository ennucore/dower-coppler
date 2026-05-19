# Dower Coppler Paper TODO

## Done

- [x] Correct the standalone Sep 21 NPZ timing metadata and velocity-like arrays so `frame_rate_hz` is the compounded slow-time cadence, with the empirical pulse PRF stored separately.
- [x] Verify the Sep 21 H5 timing ground truth on `monster`: acquisition 200 has `num_angles=5`, `num_loops=700`, `iq_frames` shape `(702, 5, 8, 134, 88)`, pulse PRF 1244.224 Hz, and acq. 200-399 mean pulse PRF 1245.932 Hz.
- [x] Update the Dower Coppler velocity formula to use `f_slow = PRF / num_angles` for compounded plane-wave data.
- [x] Make the TMAS/product-of-autocorrelation-magnitudes precursor explicit in the manuscript.
- [x] Add the ablation figure and text for `v_phi`, `G_R`, `R2`, `v_phi * G_R`, `v_phi * R2`, and full `v_phi * G_R * R2`.
- [x] Regenerate the elevation montage from the fine `y=-4..4 mm` run, using planes 2-7 inclusive.
- [x] Fix the elevation montage color-scale statement: the code now uses one shared symmetric 97th-percentile scale across displayed planes, and the caption says so.
- [x] Point the manuscript code/reproducibility link to `https://github.com/ennucore/dower-coppler`.
- [x] Add the derived NPZ inputs used by the paper figures to the standalone paper repository.
- [x] Address the V6 CNR outlier by reporting median CNR and noting that V6 is a one-pixel compact high-contrast ROI.
- [x] Compute split-half sign agreement for acquisitions 200-299 vs 300-399 in the vessel ROIs and save the summary in `outputs/paper_stats/split_half_sign_agreement.json`.

## Open

- [ ] Regenerate or remove the stale temporal-stability figure. The default figure script currently skips it when the viewer NPZ is already averaged.
- [ ] Fix temporal-stability panel labels if the figure is kept: labels should be buffer counts, not index ranges.
- [ ] Add real `\\author{}` list, affiliations, acknowledgments, funding, and conflicts of interest.
- [ ] Add a data availability statement for the raw H5 ultratraces; if raw data cannot be public, state the controlled-access / IRB limitation clearly.
- [ ] Check the realtime/acquisition-level color Doppler path in `caterpillar/acquire/acquisition.py`; it may still pass empirical pulse PRF into Doppler estimators for compounded slow-time data.
- [ ] Decide whether to redo the quantitative ROI/CNR analysis with ROIs selected from power Doppler/anatomy or by blinded selection, instead of Dower-selected ROIs.
- [ ] Add a cleaner repeatability table/figure if desired: split-half sign agreement, signed gCNR, and background false-positive rate.
- [ ] Confirm the paper figures use the manuscript formula `v_phi * G_R * R2`, not an older Huber/product TMAS variant.
- [ ] Build a minimal arXiv source bundle containing only `paper.tex`, bibliography/bbl, and included figure files; exclude NPZs, logs, screenshots, audits, caches, and backup folders.

## Stronger-paper items beyond a basic arXiv preprint

- [ ] Add a flow phantom or other ground-truth direction/velocity validation.
- [ ] Add more subjects or repeat sessions if available.
- [ ] Compare against additional signed Doppler baselines beyond Kasai, if easy from existing outputs.
