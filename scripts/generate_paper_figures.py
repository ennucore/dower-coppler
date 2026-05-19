#!/usr/bin/env python3
"""Generate publication figures for the signed TMAS paper.

Reads the pre-computed Doppler .npz files and produces:
  1. Three-panel comparison (PD / CD / DC) — hero figure
  2. Temporal stability montage — DC at different averaging windows
  3. CNR bar chart from ROI measurements

Parameters match the doppler_cnr_viewer.py screenshots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "paper_figures"
DEFAULT_REGIONS_PATH = DATA_DIR / "cnr_measurement_20260515_191622.regions.json"
DEFAULT_ALL_PLANES_PATH = DATA_DIR / "head_2025-09-21_new_h5_recomputed_dower_acq200_399_mid8elev.npz"
DEFAULT_EXTERNAL_RECORDING_PATH = DATA_DIR / (
    "bt24480388_2026-05-18_152605_txel0_h5_row-1_fine_xz_y-3p5to3p5mm_10elev_all20.npz"
)

# Display parameters from the screenshots
PLANE = 7       # main display plane (header image, temporal stability)
CNR_PLANE = 2   # plane used for CNR/ROI analysis (vessel ROIs were tuned for this)
PD_DB_LIMITS = (-100.0, 140.0)
CD_ABS_PERCENTILE = 99.0
DC_ABS_PERCENTILE = 99.0
CNR_NOISE_MODE = "both"  # sqrt(var(signal) + var(background))

# Physical coordinates for plane 2 (from grid.h5, in cm)
X_RANGE_CM = (-1.5, 1.5)      # lateral (cropped to active region)
Z_RANGE_CM = (2.100, 3.790)   # axial (depth)
EXTENT_CM = [X_RANGE_CM[0], X_RANGE_CM[1], Z_RANGE_CM[0], Z_RANGE_CM[1]]
# Pixel columns corresponding to the crop (full range is -2.746 to 2.746 over 266 px)
_FULL_X_MIN, _FULL_X_MAX = -2.746, 2.746
_X_CROP_START = int(round((X_RANGE_CM[0] - _FULL_X_MIN) / (_FULL_X_MAX - _FULL_X_MIN) * 266))
_X_CROP_END = int(round((X_RANGE_CM[1] - _FULL_X_MIN) / (_FULL_X_MAX - _FULL_X_MIN) * 266))


def load_npz(path: Path) -> dict:
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    if {"phase_velocity", "geomean_r", "phase_r2"}.issubset(data):
        data["dower_coppler"] = (
            np.asarray(data["phase_velocity"], dtype=np.float32)
            * np.asarray(data["geomean_r"], dtype=np.float32)
            * np.asarray(data["phase_r2"], dtype=np.float32)
        ).astype(np.float32)
    return data


def metric_plane(data: dict, key: str, plane: int, acq_start: int = 0, acq_end: int | None = None) -> np.ndarray:
    """Return one 2D metric plane from either per-acq or already-averaged viewer files."""
    arr = np.asarray(data[key])
    if arr.ndim == 4:
        end = arr.shape[0] if acq_end is None else min(acq_end, arr.shape[0])
        return np.median(arr[acq_start:end, plane], axis=0)
    if arr.ndim == 3:
        return arr[plane]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported shape for {key}: {arr.shape}")


def selected_plane(data: dict, key: str, plane: int | None = None) -> tuple[np.ndarray, int]:
    """Return a 2D plane from an already-aggregated or per-plane viewer metric."""
    arr = np.asarray(data[key])
    if arr.ndim == 4:
        n_planes = arr.shape[1]
        plane_idx = n_planes // 2 if plane is None or plane < 0 else min(int(plane), n_planes - 1)
        return arr[0, plane_idx], plane_idx
    if arr.ndim == 3:
        n_planes = arr.shape[0]
        plane_idx = n_planes // 2 if plane is None or plane < 0 else min(int(plane), n_planes - 1)
        return arr[plane_idx], plane_idx
    if arr.ndim == 2:
        return arr, 0
    raise ValueError(f"Unsupported shape for {key}: {arr.shape}")


def axis_extent_cm(data: dict, shape: tuple[int, int]) -> list[float]:
    """Use saved physical axes when available, otherwise fall back to the original paper extent."""
    if "x_mm" in data and "z_mm" in data:
        x = np.asarray(data["x_mm"], dtype=float) / 10.0
        z = np.asarray(data["z_mm"], dtype=float) / 10.0
        if x.size == shape[1] and z.size == shape[0]:
            return [float(x.min()), float(x.max()), float(z.min()), float(z.max())]
    return EXTENT_CM


def load_region_export(path: Path) -> tuple[dict, list[np.ndarray], np.ndarray]:
    """Load viewer-exported masks, replacing signal ROIs with inscribed circles."""
    info = json.loads(path.read_text())
    shape = tuple(info["image_shape_rc"])
    signal_masks = []
    circle_records = []
    for selection in info["selections"]:
        mask = np.zeros(shape, dtype=bool)
        pixels = np.asarray(selection["signal_pixels_rc"], dtype=int)
        if pixels.size:
            mask[pixels[:, 0], pixels[:, 1]] = True
        circle_mask, center_row, center_col, radius = largest_inscribed_circle(mask)
        signal_masks.append(circle_mask)
        circle_records.append({
            "center_row": center_row,
            "center_col": center_col,
            "radius_px": radius,
            "original_pixels": int(mask.sum()),
            "circle_pixels": int(circle_mask.sum()),
        })

    # Use the first exported background rectangle for every ROI so the CNR
    # comparison is controlled by one common vessel-free reference region.
    bg_bounds = info["selections"][0]["background_bounds"]
    bg_mask = rect_mask(
        shape,
        int(bg_bounds["col_min"]),
        int(bg_bounds["row_min"]),
        int(bg_bounds["col_max"]) + 1,
        int(bg_bounds["row_max"]) + 1,
    )
    info["inscribed_circles"] = circle_records
    return info, signal_masks, bg_mask


def largest_inscribed_circle(mask: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Largest pixel-center circle fully contained in a binary ROI mask."""
    mask = np.asarray(mask, dtype=bool)
    circle = np.zeros_like(mask, dtype=bool)
    if not mask.any():
        return circle, np.nan, np.nan, np.nan
    dist = ndimage.distance_transform_edt(mask)
    row, col = np.unravel_index(int(np.argmax(dist)), dist.shape)
    radius = float(dist[row, col])
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    circle = ((yy - float(row)) ** 2 + (xx - float(col)) ** 2 <= max(0.0, radius - 1e-6) ** 2)
    return circle & mask, float(row), float(col), radius


def pd_image(raw: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Power Doppler in dB with fixed viewer-matched display limits."""
    img = 10.0 * np.log10(np.maximum(raw, 1e-12))
    return img.astype(np.float32), PD_DB_LIMITS[0], PD_DB_LIMITS[1]


def signed_image(raw: np.ndarray, percentile: float) -> tuple[np.ndarray, float, float]:
    """Signed image with symmetric limits."""
    img = raw.astype(np.float32)
    finite = img[np.isfinite(img)]
    lim = float(np.percentile(np.abs(finite), percentile)) if finite.size else 1.0
    return img, -lim, lim


def compute_cnr(signal: np.ndarray, background: np.ndarray) -> float:
    """CNR in dB using sqrt(var(signal) + var(background)) denominator."""
    s = signal[np.isfinite(signal)].astype(np.float64)
    b = background[np.isfinite(background)].astype(np.float64)
    if s.size == 0 or b.size == 0:
        return np.nan
    num = abs(float(np.mean(s)) - float(np.mean(b)))
    den = float(np.sqrt(float(np.var(s)) + float(np.var(b))))
    if den <= 0:
        return np.nan
    cnr = num / den
    return 20.0 * np.log10(cnr) if cnr > 0 else np.nan


def compute_gcnr(signal: np.ndarray, background: np.ndarray, bins: int = 256) -> float:
    s = signal[np.isfinite(signal)].ravel()
    b = background[np.isfinite(background)].ravel()
    if s.size == 0 or b.size == 0:
        return np.nan
    lo = min(float(s.min()), float(b.min()))
    hi = max(float(s.max()), float(b.max()))
    if hi <= lo:
        return 0.0
    hs, edges = np.histogram(s, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=edges, range=(lo, hi), density=True)
    overlap = np.sum(np.minimum(hs, hb) * np.diff(edges))
    return float(np.clip(1.0 - overlap, 0.0, 1.0))


def ellipse_mask(shape, cx, cy, rx, ry):
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def rect_mask(shape, x0, y0, x1, y1):
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    return mask


# ── Figure 1: Three-panel comparison ──────────────────────────────────


def _signed_log(arr: np.ndarray) -> np.ndarray:
    """Compress dynamic range: sign(x) * log10(1 + |x|)."""
    return np.sign(arr) * np.log10(1.0 + np.abs(arr))


def figure_three_panel(data: dict, output_dir: Path, acq_start: int = 0, acq_end: int = 480):
    """PD / CD / DC side-by-side from the per-acq dataset."""
    plane = min(PLANE, np.asarray(data["power_doppler"]).shape[-3] - 1)
    pd_raw = metric_plane(data, "power_doppler", plane, acq_start, acq_end)
    cd_raw = metric_plane(data, "color_doppler", plane, acq_start, acq_end)
    dc_raw = metric_plane(data, "dower_coppler", plane, acq_start, acq_end)

    extent = axis_extent_cm(data, pd_raw.shape)

    pd_img, pd_vmin, pd_vmax = pd_image(pd_raw)
    cd_img, cd_vmin, cd_vmax = signed_image(cd_raw, CD_ABS_PERCENTILE)

    dc_img, dc_vmin, dc_vmax = signed_image(dc_raw, DC_ABS_PERCENTILE)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    imshow_kw = dict(origin="lower", aspect="equal", extent=extent)

    axes[0].imshow(pd_img, cmap="magma", vmin=pd_vmin, vmax=pd_vmax, **imshow_kw)
    axes[0].set_title("Power Doppler", fontsize=11, fontweight="bold")

    axes[1].imshow(cd_img, cmap="seismic", vmin=cd_vmin, vmax=cd_vmax, **imshow_kw)
    axes[1].set_title("Color Doppler (Kasai)", fontsize=11, fontweight="bold")

    axes[2].imshow(dc_img, cmap="seismic", vmin=dc_vmin, vmax=dc_vmax, **imshow_kw)
    axes[2].set_title("Dower Coppler", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Lateral (cm)", fontsize=9)
        ax.set_ylabel("Depth (cm)", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.savefig(output_dir / "fig_three_panel_comparison.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_three_panel_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved three-panel comparison")


# ── Figure 2: Temporal stability montage ──────────────────────────────


def figure_temporal_stability(data: dict, output_dir: Path):
    """DC maps at different averaging windows showing stability."""
    windows = [
        (0, 250, "250 buffers"),
        (150, 250, "Buffers 150–250"),
        (200, 250, "Buffers 200–250"),
        (240, 250, "Buffers 240–250"),
        (245, 250, "Buffers 245–250"),
        (249, 250, "Buffer 249"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()

    # Common color scale from the 250-buffer average
    dc_full = np.median(data["dower_coppler"][:250, PLANE], axis=0)
    _, common_vmin, common_vmax = signed_image(dc_full, 99.0)

    imshow_kw = dict(origin="lower", aspect="equal",
                     extent=[_FULL_X_MIN, _FULL_X_MAX, Z_RANGE_CM[0], Z_RANGE_CM[1]])

    for idx, (start, end, label) in enumerate(windows):
        dc_raw = np.median(data["dower_coppler"][start:end, PLANE], axis=0)
        axes[idx].imshow(dc_raw.astype(np.float32), cmap="seismic",
                         vmin=common_vmin, vmax=common_vmax, **imshow_kw)
        axes[idx].set_title(label, fontsize=10, fontweight="bold")
        axes[idx].set_xlabel("Lateral (cm)", fontsize=8)
        axes[idx].set_ylabel("Depth (cm)", fontsize=8)
        axes[idx].tick_params(labelsize=7)

    fig.suptitle("Dower Coppler temporal stability", fontsize=12, fontweight="bold")
    fig.savefig(output_dir / "fig_temporal_stability.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_temporal_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved temporal stability montage")


# ── Figure: All elevation planes ──────────────────────────────────────


def figure_all_planes(data: dict, output_dir: Path, acq_start: int = 0, acq_end: int = 480):
    """Dower Coppler maps for all 10 elevation planes."""
    arr = np.asarray(data["dower_coppler"])
    n_planes = arr.shape[1] if arr.ndim == 4 else arr.shape[0] if arr.ndim == 3 else 1
    ncols = 2
    nrows = (n_planes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, nrows * 2.0), constrained_layout=True)
    axes = axes.ravel()

    for plane in range(n_planes):
        dc_full = metric_plane(data, "dower_coppler", plane, acq_start, acq_end)
        dc_img, dc_vmin, dc_vmax = signed_image(dc_full, 97.0)
        extent = axis_extent_cm(data, dc_full.shape)
        axes[plane].imshow(dc_full.astype(np.float32), origin="lower", aspect="equal",
                           extent=extent,
                           cmap="seismic", vmin=dc_vmin, vmax=dc_vmax)
        axes[plane].set_title(f"Plane {plane}", fontsize=10, fontweight="bold")
        axes[plane].set_xlabel("Lateral (cm)", fontsize=8)
        axes[plane].set_ylabel("Depth (cm)", fontsize=8)
        axes[plane].tick_params(labelsize=7)

    # Hide any unused axes
    for idx in range(n_planes, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Dower Coppler across elevation planes", fontsize=12, fontweight="bold")
    fig.savefig(output_dir / "fig_all_planes.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_all_planes.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved all-planes figure")


# ── Figure 3: CNR bar chart ───────────────────────────────────────────


def figure_cnr_comparison(
    data: dict,
    output_dir: Path,
    regions_path: Path,
    acq_start: int = 0,
    acq_end: int = 480,
):
    """CNR bar chart comparing PD, CD, DC across vessel ROIs.

    Signal ROIs are thresholded connected components from the DC image
    (matching the CV segment approach in the viewer). Each component is
    eroded by 1 pixel to avoid edge mixing. Background is a fixed
    rectangle in a vessel-free region.
    """
    region_info, signal_masks, bg_mask = load_region_export(regions_path)
    plane = int(region_info.get("plane", min(CNR_PLANE, np.asarray(data["power_doppler"]).shape[-3] - 1)))

    pd_raw = metric_plane(data, "power_doppler", plane, acq_start, acq_end)
    cd_raw = metric_plane(data, "color_doppler", plane, acq_start, acq_end)
    dc_raw = metric_plane(data, "dower_coppler", plane, acq_start, acq_end)

    # Convert PD to dB for measurement (matching viewer behavior)
    pd_meas = 10.0 * np.log10(np.maximum(pd_raw, 1e-12)).astype(np.float32)
    cd_meas = np.abs(cd_raw).astype(np.float32)
    dc_meas = np.abs(dc_raw).astype(np.float32)

    if pd_raw.shape != bg_mask.shape:
        raise ValueError(f"Region export shape {bg_mask.shape} does not match image shape {pd_raw.shape}")

    roi_labels = [f"V{i+1}" for i in range(len(signal_masks))]
    cnr_pd = []
    cnr_cd = []
    cnr_dc = []
    gcnr_pd = []
    gcnr_cd = []
    gcnr_dc = []
    for idx, sig_mask in enumerate(signal_masks):
        this_bg_mask = bg_mask & ~sig_mask

        cnr_pd.append(compute_cnr(pd_meas[sig_mask], pd_meas[this_bg_mask]))
        cnr_cd.append(compute_cnr(cd_meas[sig_mask], cd_meas[this_bg_mask]))
        cnr_dc.append(compute_cnr(dc_meas[sig_mask], dc_meas[this_bg_mask]))

        gcnr_pd.append(compute_gcnr(pd_meas[sig_mask], pd_meas[this_bg_mask]))
        gcnr_cd.append(compute_gcnr(cd_meas[sig_mask], cd_meas[this_bg_mask]))
        gcnr_dc.append(compute_gcnr(dc_meas[sig_mask], dc_meas[this_bg_mask]))

        circle = region_info["inscribed_circles"][idx]
        print(
            f"    {roi_labels[idx]}: center=({circle['center_col']:.0f},{circle['center_row']:.0f}), "
            f"radius={circle['radius_px']:.2f}px, signal_pixels={int(sig_mask.sum())}, "
            f"original_pixels={circle['original_pixels']}, background_pixels={int(this_bg_mask.sum())}"
        )

    # Plot CNR bar chart
    x = np.arange(len(signal_masks))
    bar_w = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax1.bar(x - bar_w, cnr_pd, bar_w, label="Power Doppler", color="#B45309", alpha=0.85)
    ax1.bar(x, cnr_cd, bar_w, label="Color Doppler (Kasai)", color="#6366F1", alpha=0.85)
    ax1.bar(x + bar_w, cnr_dc, bar_w, label="Dower Coppler", color="#DC2626", alpha=0.85)
    ax1.set_xlabel("Vessel ROI", fontsize=10)
    ax1.set_ylabel("CNR (dB)", fontsize=10)
    ax1.set_title("Contrast-to-Noise Ratio", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(roi_labels)
    ax1.legend(fontsize=8)
    ax1.axhline(0, color="k", linewidth=0.5, linestyle="--")

    ax2.bar(x - bar_w, gcnr_pd, bar_w, label="Power Doppler", color="#B45309", alpha=0.85)
    ax2.bar(x, gcnr_cd, bar_w, label="Color Doppler (Kasai)", color="#6366F1", alpha=0.85)
    ax2.bar(x + bar_w, gcnr_dc, bar_w, label="Dower Coppler", color="#DC2626", alpha=0.85)
    ax2.set_xlabel("Vessel ROI", fontsize=10)
    ax2.set_ylabel("gCNR", fontsize=10)
    ax2.set_title("Generalized CNR", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(roi_labels)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.1)

    fig.savefig(output_dir / "fig_cnr_comparison.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_cnr_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Print CNR table
    print("  CNR (dB) by ROI:")
    print(f"  {'ROI':>4s}  {'PD':>7s}  {'CD':>7s}  {'DC':>7s}  {'PD gCNR':>7s}  {'CD gCNR':>7s}  {'DC gCNR':>7s}")
    for i, label in enumerate(roi_labels):
        print(f"  {label:>4s}  {cnr_pd[i]:7.1f}  {cnr_cd[i]:7.1f}  {cnr_dc[i]:7.1f}  {gcnr_pd[i]:7.2f}  {gcnr_cd[i]:7.2f}  {gcnr_dc[i]:7.2f}")
    print(f"  Saved CNR comparison")


# ── Figure 4: Three-panel with ROI overlays ───────────────────────────


def figure_three_panel_with_rois(
    data: dict,
    output_dir: Path,
    regions_path: Path,
    acq_start: int = 0,
    acq_end: int = 480,
):
    """Three-panel with vessel ROI circles overlaid."""
    region_info, signal_masks, bg_mask = load_region_export(regions_path)
    plane = int(region_info.get("plane", min(CNR_PLANE, np.asarray(data["power_doppler"]).shape[-3] - 1)))

    pd_raw = metric_plane(data, "power_doppler", plane, acq_start, acq_end)
    cd_raw = metric_plane(data, "color_doppler", plane, acq_start, acq_end)
    dc_raw = metric_plane(data, "dower_coppler", plane, acq_start, acq_end)

    pd_img, pd_vmin, pd_vmax = pd_image(pd_raw)
    cd_img, cd_vmin, cd_vmax = signed_image(cd_raw, CD_ABS_PERCENTILE)
    dc_img, dc_vmin, dc_vmax = signed_image(dc_raw, DC_ABS_PERCENTILE)

    extent = axis_extent_cm(data, pd_raw.shape)

    def pixel_to_cm(col: float, row: float) -> tuple[float, float]:
        h, w = pd_raw.shape
        x0, x1, z0, z1 = extent
        x = x0 + (col + 0.5) / w * (x1 - x0)
        z = z0 + (row + 0.5) / h * (z1 - z0)
        return x, z

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    label_offsets_cm = {
        0: (0.05, 0.00),
        1: (0.05, 0.00),
        2: (0.05, 0.00),
        3: (0.05, 0.04),
        4: (0.05, -0.03),
        5: (-0.33, 0.00),
    }

    for ax, img, cmap, vmin, vmax, title in [
        (axes[0], pd_img, "magma", pd_vmin, pd_vmax, "Power Doppler"),
        (axes[1], cd_img, "seismic", cd_vmin, cd_vmax, "Color Doppler (Kasai)"),
        (axes[2], dc_img, "seismic", dc_vmin, dc_vmax, "Dower Coppler"),
    ]:
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="equal", extent=extent)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Lateral (cm)", fontsize=8)
        ax.set_ylabel("Depth (cm)", fontsize=8)
        ax.tick_params(labelsize=8)
        for i, sig_mask in enumerate(signal_masks):
            circle = region_info["inscribed_circles"][i]
            if not np.isfinite(circle["center_col"]):
                continue
            cx, cy = float(circle["center_col"]), float(circle["center_row"])
            x_cm, z_cm = pixel_to_cm(cx, cy)
            edge_col, _ = pixel_to_cm(cx + float(circle["radius_px"]), cy)
            radius_cm = abs(edge_col - x_cm)
            ax.add_patch(Ellipse(
                (x_cm, z_cm),
                2.0 * radius_cm,
                2.0 * radius_cm,
                fill=False,
                edgecolor="cyan",
                linewidth=1.2,
            ))
            dx, dz = label_offsets_cm.get(i, (0.05, 0.0))
            ax.text(x_cm + dx, z_cm + dz, f"V{i+1}", color="white", fontsize=7,
                    va="center", bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1))
        by, bx = np.where(bg_mask)
        if bx.size:
            x0, z0 = pixel_to_cm(float(bx.min()), float(by.min()))
            x1, z1 = pixel_to_cm(float(bx.max() + 1), float(by.max() + 1))
            ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0,
                                   fill=False, edgecolor="lime", linewidth=1.0, linestyle="--"))

    fig.savefig(output_dir / "fig_three_panel_with_rois.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_three_panel_with_rois.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved three-panel with ROIs")


# ── Figure: Multi-lag velocity vs Kasai color Doppler ────────────────


def figure_velocity_vs_color(data: dict, output_dir: Path, plane: int = 0):
    """Side-by-side velocity maps and Bland-Altman comparison."""
    if "phase_velocity" not in data or "color_doppler" not in data:
        print("  Skipping velocity-vs-color figure: missing phase_velocity or color_doppler")
        return

    n_planes = np.asarray(data["phase_velocity"]).shape[-3]
    plane = min(int(plane), n_planes - 1)
    phase_mm_s = metric_plane(data, "phase_velocity", plane) * 1000.0
    color_mm_s = metric_plane(data, "color_doppler", plane) * 1000.0
    extent = axis_extent_cm(data, phase_mm_s.shape)

    finite = np.isfinite(phase_mm_s) & np.isfinite(color_mm_s)
    if not finite.any():
        print("  Skipping velocity-vs-color figure: no finite pixels")
        return

    lim = float(np.percentile(np.abs(np.concatenate([phase_mm_s[finite], color_mm_s[finite]])), 99.0))
    lim = max(lim, 1e-6)

    mean = 0.5 * (phase_mm_s[finite] + color_mm_s[finite])
    diff = phase_mm_s[finite] - color_mm_s[finite]
    bias = float(np.mean(diff))
    sd = float(np.std(diff))
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    fig = plt.figure(figsize=(14, 4.2), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.25])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    imshow_kw = dict(origin="lower", aspect="equal", extent=extent, cmap="seismic", vmin=-lim, vmax=lim)
    im0 = axes[0].imshow(color_mm_s, **imshow_kw)
    axes[0].set_title("Kasai color Doppler", fontsize=11, fontweight="bold")
    axes[1].imshow(phase_mm_s, **imshow_kw)
    axes[1].set_title("Multi-lag phase velocity", fontsize=11, fontweight="bold")
    for ax in axes[:2]:
        ax.set_xlabel("Lateral (cm)", fontsize=9)
        ax.set_ylabel("Depth (cm)", fontsize=9)
        ax.tick_params(labelsize=8)
    cbar = fig.colorbar(im0, ax=axes[:2], shrink=0.85, pad=0.02)
    cbar.set_label("Velocity (mm/s)", fontsize=9)

    axes[2].scatter(mean, diff, s=3, alpha=0.18, color="#2563eb", edgecolors="none")
    axes[2].axhline(bias, color="#d97706", linestyle="--", linewidth=1.2, label=f"bias {bias:.2f}")
    axes[2].axhline(loa_low, color="#dc2626", linestyle="--", linewidth=1.0, label="-1.96 SD")
    axes[2].axhline(loa_high, color="#dc2626", linestyle="--", linewidth=1.0, label="+1.96 SD")
    axes[2].set_title("Bland-Altman", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Mean velocity (mm/s)", fontsize=9)
    axes[2].set_ylabel("Multi-lag - Kasai (mm/s)", fontsize=9)
    axes[2].tick_params(labelsize=8)
    axes[2].grid(True, alpha=0.25, linewidth=0.5)
    axes[2].legend(fontsize=7, loc="upper right", frameon=False)
    axes[2].text(
        0.03,
        0.03,
        f"n={mean.size:,}\nSD={sd:.2f} mm/s\nLoA [{loa_low:.1f}, {loa_high:.1f}]",
        transform=axes[2].transAxes,
        fontsize=8,
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
    )

    fig.savefig(output_dir / "fig_velocity_vs_color_bland_altman.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_velocity_vs_color_bland_altman.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(
        "  Saved velocity-vs-color/Bland-Altman figure "
        f"(bias={bias:.3g} mm/s, SD={sd:.3g} mm/s)"
    )


# ── Figure: Dower Coppler ablation ────────────────────────────────────


def figure_dower_ablation(data: dict, output_dir: Path, plane: int | None = None):
    """Show the main Dower Coppler components on the same image plane."""
    required = {"phase_velocity", "geomean_r", "phase_r2"}
    if not required.issubset(data):
        print("  Skipping Dower ablation figure: missing phase_velocity/geomean_r/phase_r2")
        return

    n_planes = np.asarray(data["phase_velocity"]).shape[-3] if np.asarray(data["phase_velocity"]).ndim >= 3 else 1
    plane_idx = min(CNR_PLANE, n_planes - 1) if plane is None else min(int(plane), n_planes - 1)

    v_phi = metric_plane(data, "phase_velocity", plane_idx)
    g_r = metric_plane(data, "geomean_r", plane_idx)
    r2 = metric_plane(data, "phase_r2", plane_idx)
    v_phi_g_r = v_phi * g_r
    if "phase_velocity_r2" in data:
        v_phi_r2 = metric_plane(data, "phase_velocity_r2", plane_idx)
    else:
        v_phi_r2 = v_phi * r2

    extent = axis_extent_cm(data, v_phi.shape)
    panels = [
        (r"$v_\phi$ alone", v_phi, "seismic", "signed"),
        (r"$v_\phi \cdot G_R$", v_phi_g_r, "seismic", "signed"),
        (r"$v_\phi \cdot R^2$", v_phi_r2, "seismic", "signed"),
        (r"$G_R$ alone", g_r, "magma", "unsigned"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes = axes.ravel()
    for ax, (title, img, cmap, mode) in zip(axes, panels):
        if mode == "signed":
            img_plot, vmin, vmax = signed_image(img, DC_ABS_PERCENTILE)
        else:
            img_plot = img.astype(np.float32)
            finite = img_plot[np.isfinite(img_plot)]
            vmin = 0.0
            vmax = float(np.percentile(finite, 99.0)) if finite.size else 1.0
        im = ax.imshow(
            img_plot,
            origin="lower",
            aspect="equal",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Lateral (cm)", fontsize=9)
        ax.set_ylabel("Depth (cm)", fontsize=9)
        ax.tick_params(labelsize=8)
        cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Dower Coppler ablation components", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "fig_dower_ablation.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_dower_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Dower ablation figure")


# ── Figure: External recording check ──────────────────────────────────


def figure_external_recording(data: dict, output_dir: Path, plane: int | None = None):
    """Dower Coppler image from a separate recording, configurable by input file."""
    dc_raw, plane_idx = selected_plane(data, "dower_coppler", plane)
    dc_img, dc_vmin, dc_vmax = signed_image(dc_raw, DC_ABS_PERCENTILE)
    extent = axis_extent_cm(data, dc_img.shape)

    source_h5 = str(np.asarray(data.get("source_h5", "unknown")).item())
    acq_count = int(np.asarray(data.get("acq_count", 0)).item()) if "acq_count" in data else 0
    frame_rate = float(np.asarray(data.get("frame_rate_hz", np.nan)).item()) if "frame_rate_hz" in data else np.nan

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
    im = ax.imshow(
        dc_img,
        cmap="seismic",
        vmin=dc_vmin,
        vmax=dc_vmax,
        origin="lower",
        aspect="equal",
        extent=extent,
    )
    ax.set_title("Different recording: May 18, 2026, tx_el=0", fontsize=11, fontweight="bold")
    ax.set_xlabel("Lateral (cm)", fontsize=9)
    ax.set_ylabel("Depth (cm)", fontsize=9)
    ax.tick_params(labelsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Dower Coppler (a.u.)", fontsize=9)
    ax.text(
        0.02,
        0.02,
        f"middle plane {plane_idx}; {acq_count} acquisitions; PRF {frame_rate:.1f} Hz",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.65, pad=2),
    )

    fig.savefig(output_dir / "fig_external_recording_may18.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_external_recording_may18.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved external recording figure from {source_h5} (plane {plane_idx})")


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument("--all-planes-data", type=Path, default=DEFAULT_ALL_PLANES_PATH)
    parser.add_argument("--external-recording-data", type=Path, default=DEFAULT_EXTERNAL_RECORDING_PATH)
    parser.add_argument(
        "--external-recording-plane",
        type=int,
        default=-1,
        help="Elevation plane for the external-recording figure; -1 selects the middle plane.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.regions.exists():
        region_info = json.loads(args.regions.read_text())
        per_acq_path = Path(region_info["dataset_path"])
        if not per_acq_path.is_absolute():
            per_acq_path = ROOT / per_acq_path
        if not per_acq_path.exists():
            per_acq_path = args.data_dir / per_acq_path.name
    else:
        per_acq_path = args.data_dir / "head_2025-09-21_per_acq_doppler_full_post_cutoff.npz"
    print("Loading per-acq dataset...")
    per_acq = load_npz(per_acq_path)
    print(f"  Path: {per_acq_path}")
    print(f"  Shape: {per_acq['power_doppler'].shape}")

    print("Loading all-planes dataset...")
    all_planes = load_npz(args.all_planes_data)
    print(f"  Path: {args.all_planes_data}")
    print(f"  Shape: {all_planes['dower_coppler'].shape}")

    print("\nGenerating figures:")
    figure_three_panel(per_acq, args.output_dir)
    if np.asarray(per_acq["dower_coppler"]).ndim == 4:
        figure_temporal_stability(per_acq, args.output_dir)
    else:
        print("  Skipping temporal stability montage for already-averaged viewer dataset")

    figure_cnr_comparison(per_acq, args.output_dir, args.regions, acq_start=0, acq_end=480)
    figure_three_panel_with_rois(per_acq, args.output_dir, args.regions, acq_start=0, acq_end=480)
    figure_dower_ablation(per_acq, args.output_dir)
    figure_velocity_vs_color(per_acq, args.output_dir, plane=0)

    figure_all_planes(all_planes, args.output_dir)

    if args.external_recording_data.exists():
        print("Loading external recording dataset...")
        external_recording = load_npz(args.external_recording_data)
        print(f"  Path: {args.external_recording_data}")
        print(f"  Shape: {external_recording['dower_coppler'].shape}")
        figure_external_recording(
            external_recording,
            args.output_dir,
            plane=None if args.external_recording_plane < 0 else args.external_recording_plane,
        )
    else:
        print(f"  Skipping external recording figure: {args.external_recording_data} not found")

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
