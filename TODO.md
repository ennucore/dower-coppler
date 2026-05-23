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
- [x] Regenerate the September 21 paper figures from the full 480-acquisition ultratrace outputs.
- [x] Address the V6/one-pixel CNR issue by replacing strict inscribed circles with tolerant circular ROIs and reporting medians.
- [x] Compute split-half sign agreement for acquisitions 200-299 vs 300-400 in the vessel ROIs and save the summary in `outputs/paper_stats/split_half_sign_agreement.json`.
- [x] Replace the stale temporal-stability figure with the fine-elevation per-acquisition sidecar montage for acquisitions 200-400, labeled by acquisition counts.
- [x] Add the temporal-stability sidecar path and default figure-generation parameters to the reproducibility manifest.
- [x] Confirm the paper figures use the manuscript formula `v_phi * G_R * R2`, not an older Huber/product TMAS variant.
- [x] Replace strict fully-inscribed CNR circles with tolerant circular ROIs requiring at least 80% overlap with the exported vessel mask, eliminating one-pixel signal ROIs.
- [x] Remove color Doppler from the Figure 5 left CNR bar chart while keeping color Doppler in the gCNR panels and text.
- [x] Crop Figures 2, 6, and 7 to the lateral range -1.5 to 1.5 cm and resize the Figure 2 colorbars to the image panels.
- [x] Make Table 2 simulation results reproducible from the standalone repo with `scripts/phase_ls_research_sweep.py`, `outputs/research_sweep/phase_ls_research_sweep.summary.json`, and generated `outputs/paper_stats/simulation_results_table.tex`.
- [x] Make the temporal-stability figure reproducible from a fresh clone by adding `data/head_2025-09-21_temporal_windows_plane4.npz` and making `scripts/generate_paper_figures.py` prefer that compact source over the large external sidecar.
- [x] Correct the May 18 external-recording provenance. The source H5 is under `/mnt/pocampus/lev/may18_txel_sweep_minus5_0_plus5/`, not `/home/monster/caterpillar/data/`; its H5 PRF is a transmit pulse PRF with `num_angles=5`, so the corrected compounded cadence is PRF / 5 = 295.252 Hz.
- [x] Correct legacy PRF/cadence metadata in the Sep 21 all-480 and May 18 NPZs, store `pulse_repetition_rate_hz` separately, and verify `dower_coppler = phase_velocity * geomean_r * phase_r2`.
- [x] Update the literature context to cite recent intact-adult-skull transcranial fUS work and narrow the novelty claim to the signed, coherence-weighted matrix-array result.
- [x] Regenerate paper figures/stats/PDF after the cadence fix and refresh `reproducibility/manifest.json` hashes.
- [x] Update the Doppler CNR viewer so the paper header dataset opens first with real PD, independent CD, and DC, while preserving stored `dower_coppler` instead of overwriting it with a derived alias.
- [x] Add the same segment-to-largest-tolerant-circle measurement path to the Doppler CNR viewer and launch it on the paper header dataset.
- [x] Make figure-output reproducibility less brittle by removing Matplotlib-generated figure PDFs from strict manifest hashes while continuing to hash deterministic PNGs and `paper.pdf`.
- [x] Fix `~/dev/caterpillar/caterpillar/acquire/acquisition.py` so realtime color Doppler uses compounded slow-time cadence (`pulse PRF / total_num_angles`) instead of pulse PRF for velocity/frequency estimators.
- [x] Build a minimal arXiv source bundle at `dist/dower-coppler-arxiv-source.tar.gz` containing `paper.tex`, `paper.bbl`, `references.bib`, included PNG figures, and included generated table `.tex` files, excluding NPZs, logs, screenshots, audits, caches, and backup folders.
- [x] Refresh publication cleanup state: regenerate external velocity tables, rebuild `paper.pdf`, ignore generated `dist/` and external diagnostics output, and update `reproducibility/manifest.json` hashes.

## Open

- [ ] Add real `\\author{}` list, affiliations, acknowledgments, funding, and conflicts of interest.
- [ ] Add a data availability statement for the raw H5 ultratraces; if raw data cannot be public, state the controlled-access / IRB limitation clearly.
- [ ] Decide whether to redo the quantitative ROI/CNR analysis with ROIs selected from power Doppler/anatomy or by blinded selection, instead of Dower-selected ROIs.
- [ ] Add a cleaner repeatability table/figure if desired: split-half sign agreement, signed gCNR, and background false-positive rate.

## Stronger-paper items beyond a basic arXiv preprint

- [ ] Add a flow phantom or other ground-truth direction/velocity validation.
- [ ] Add more subjects or repeat sessions if available.
- [ ] Compare against additional signed Doppler baselines beyond Kasai, if easy from existing outputs.
