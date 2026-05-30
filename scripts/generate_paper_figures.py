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
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "paper_figures"
DEFAULT_REGIONS_PATH = DATA_DIR / "cnr_measurement_20260515_191622.regions.json"
DEFAULT_MAIN_DATA_PATH = DATA_DIR / "head_2025-09-21_full2dtx_fast8_fine_xz_yidx14_15_acq000_479.npz"
DEFAULT_ALL_PLANES_PATH = DATA_DIR / "head_2025-09-21_full2dtx_fast8_fine_xz_y-4to4mm_10elev_acq000_479.npz"
DEFAULT_ALL_PLANES_START = 2
DEFAULT_ALL_PLANES_END = 7
DEFAULT_TEMPORAL_SUMMARY_PATH = DATA_DIR / "head_2025-09-21_temporal_windows_plane4.npz"
DEFAULT_SPLIT_HALF_PATH = DATA_DIR / "head_2025-09-21_split_half_plane4_acq200_400.npz"
DEFAULT_TEMPORAL_PER_ACQ_DIR = Path(
    "/Users/lev/dev/caterpillar/results/doppler_cnr_gui/"
    "head_2025-09-21_full2dtx_fast8_fine_xz_y-4to4mm_10elev_acq200_400_per_acq"
)
DEFAULT_TEMPORAL_PLANE = 4
TEMPORAL_WINDOWS = [
    (200, 400, "201 acquisitions"),
    (241, 400, "160 acquisitions"),
    (281, 400, "120 acquisitions"),
    (321, 400, "80 acquisitions"),
    (361, 400, "40 acquisitions"),
    (391, 400, "10 acquisitions"),
]
DEFAULT_SPLIT_A = (200, 299)
DEFAULT_SPLIT_B = (300, 400)
DEFAULT_EXTERNAL_RECORDING_PATH = DATA_DIR / (
    "bt24480388_2026-05-18_152605_txel0_h5_row-1_fine_xz_y-3p5to3p5mm_10elev_all20.npz"
)
LEGACY_COMPOUND_ANGLE_RULES = (
    (
        ("head_2025-09-21", "sep21_cached", "2025-09-21"),
        5,
        "Sep 21 H5 acq config stores num_angles=5",
    ),
    (
        ("bt24480388_2026-05-18_152605", "ultratrace_BT24480388_monster_2026-05-18_15:26:05"),
        5,
        "May 18 H5 acq config stores num_angles=5",
    ),
)

# Display parameters from the screenshots
PLANE = 7       # main display plane (header image, temporal stability)
CNR_PLANE = 2   # plane used for CNR/ROI analysis (vessel ROIs were tuned for this)
PD_DB_PERCENTILES = (1.0, 99.5)
CD_ABS_PERCENTILE = 99.0
DC_ABS_PERCENTILE = 99.0
TEMPORAL_ABS_PERCENTILE = 97.0
SPLIT_HALF_ABS_PERCENTILE = 97.0
SPLIT_HALF_AGREEMENT_TOP_PCT = 10
CNR_NOISE_MODE = "both"  # sqrt(var(signal) + var(background))
ROI_CIRCLE_MIN_OVERLAP = 0.80
ROI_CIRCLE_MAX_RADIUS_PX = 16.0
ROI_CIRCLE_RADIUS_STEP_PX = 0.25

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
    data = apply_compound_frame_rate_correction(data, path)
    if {"phase_velocity", "geomean_r", "phase_r2"}.issubset(data):
        data["dower_coppler"] = (
            np.asarray(data["phase_velocity"], dtype=np.float32)
            * np.asarray(data["geomean_r"], dtype=np.float32)
            * np.asarray(data["phase_r2"], dtype=np.float32)
        ).astype(np.float32)
    return data


def validate_in_vivo_baselines(data: dict, path: Path) -> None:
    """Reject stale placeholder baselines in the main in-vivo figure input.

    The paper compares Dower Coppler against standard SVD-filtered power
    Doppler and an independent lag-1 Kasai color Doppler estimate. Older Sep21
    caches stored power_doppler=abs(dower_coppler) and
    color_doppler=phase_velocity as viewer placeholders; those are not valid
    baselines for Figure 1, CNR/gCNR, or the Bland-Altman comparison.
    """
    required = {"power_doppler", "dower_coppler", "color_doppler", "phase_velocity"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{path} is missing required in-vivo baseline fields: {missing}")

    power = np.asarray(data["power_doppler"])
    dower_abs = np.abs(np.asarray(data["dower_coppler"]))
    if power.shape == dower_abs.shape and np.array_equal(power, dower_abs):
        raise ValueError(
            f"{path} has stale power_doppler=abs(dower_coppler); regenerate it as "
            "SVD-filtered power Doppler sum_t |S'_t|^2 before making paper figures."
        )

    color = np.asarray(data["color_doppler"])
    phase = np.asarray(data["phase_velocity"])
    if color.shape == phase.shape and np.array_equal(color, phase):
        raise ValueError(
            f"{path} has stale color_doppler=phase_velocity; regenerate it as an "
            "independent lag-1 Kasai estimate angle(R_1) before making paper figures."
        )


def _legacy_compound_angle_rule(data: dict, path: Path) -> tuple[int, str] | None:
    source_h5 = ""
    if "source_h5" in data:
        source_h5 = str(np.asarray(data["source_h5"]).item())
    haystack = f"{path.name} {source_h5}"
    for tokens, num_angles, note in LEGACY_COMPOUND_ANGLE_RULES:
        if any(token in haystack for token in tokens):
            return num_angles, note
    return None


def apply_compound_frame_rate_correction(data: dict, path: Path) -> dict:
    """Correct legacy velocity fields from pulse PRF to compound cadence.

    Legacy NPZs used the empirical transmit pulse repetition rate directly for
    phase-to-velocity conversion. For five-angle plane-wave compounding, the
    Doppler slow-time cadence is pulse PRF / 5.
    Corrected NPZs store velocity_scale_corrected=True or compound_frame_rate_hz.
    """
    already_corrected = bool(np.asarray(data.get("velocity_scale_corrected", False)).item())
    rule = _legacy_compound_angle_rule(data, path)
    if rule is None or already_corrected or "compound_frame_rate_hz" in data:
        return data

    num_angles, timing_note = rule
    out = dict(data)
    scale = np.float32(1.0 / float(num_angles))
    for key in ("phase_velocity", "color_doppler", "phase_velocity_r2"):
        if key in out:
            out[key] = (np.asarray(out[key], dtype=np.float32) * scale).astype(np.float32)

    if {"phase_velocity", "geomean_r", "phase_r2"}.issubset(out):
        out["dower_coppler"] = (
            np.asarray(out["phase_velocity"], dtype=np.float32)
            * np.asarray(out["geomean_r"], dtype=np.float32)
            * np.asarray(out["phase_r2"], dtype=np.float32)
        ).astype(np.float32)
    if {"phase_velocity", "geomean_r", "huber_quality"}.issubset(out):
        out["dower_huber_quality"] = (
            np.asarray(out["phase_velocity"], dtype=np.float32)
            * np.asarray(out["geomean_r"], dtype=np.float32)
            * np.asarray(out["huber_quality"], dtype=np.float32)
        ).astype(np.float32)

    if "frame_rate_hz" in out:
        pulse_prf = np.float32(np.asarray(out["frame_rate_hz"]).item())
        out["pulse_repetition_rate_hz"] = pulse_prf
        out["compound_frame_rate_hz"] = np.float32(float(pulse_prf) * float(scale))
        out["frame_rate_hz"] = out["compound_frame_rate_hz"]
    out["num_compound_angles"] = np.int32(num_angles)
    out["velocity_scale_correction"] = scale
    out["velocity_scale_corrected"] = np.bool_(True)
    out["timing_note"] = np.asarray(
        f"{timing_note}; Doppler slow-time samples are compounded frames, so velocity conversion uses pulse PRF / {num_angles}."
    )
    return out


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


def crop_lateral_cm(image: np.ndarray, data: dict, x_range: tuple[float, float] = X_RANGE_CM) -> tuple[np.ndarray, list[float]]:
    """Crop a 2D image to the requested lateral range in centimeters."""
    img = np.asarray(image)
    extent = axis_extent_cm(data, img.shape)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {img.shape}")

    if "x_mm" in data:
        x_cm = np.asarray(data["x_mm"], dtype=float) / 10.0
        if x_cm.size == img.shape[1]:
            keep = (x_cm >= x_range[0]) & (x_cm <= x_range[1])
            if np.any(keep):
                return img[:, keep], [float(x_range[0]), float(x_range[1]), extent[2], extent[3]]

    x_centers = np.linspace(extent[0], extent[1], img.shape[1])
    keep = (x_centers >= x_range[0]) & (x_centers <= x_range[1])
    if np.any(keep):
        return img[:, keep], [float(x_range[0]), float(x_range[1]), extent[2], extent[3]]
    return img, extent


def load_region_export(path: Path) -> tuple[dict, list[np.ndarray], np.ndarray]:
    """Load viewer-exported masks, replacing signal ROIs with tolerant circles."""
    info = json.loads(path.read_text())
    shape = tuple(info["image_shape_rc"])
    signal_masks = []
    circle_records = []
    for selection in info["selections"]:
        mask = np.zeros(shape, dtype=bool)
        pixels = np.asarray(selection["signal_pixels_rc"], dtype=int)
        if pixels.size:
            mask[pixels[:, 0], pixels[:, 1]] = True
        strict_mask, _, _, strict_radius = largest_inscribed_circle(mask)
        circle_mask, center_row, center_col, radius, overlap, inside_pixels = largest_tolerant_circle(
            mask,
            min_overlap=ROI_CIRCLE_MIN_OVERLAP,
            max_radius=ROI_CIRCLE_MAX_RADIUS_PX,
            radius_step=ROI_CIRCLE_RADIUS_STEP_PX,
        )
        signal_masks.append(circle_mask)
        circle_records.append({
            "center_row": center_row,
            "center_col": center_col,
            "radius_px": radius,
            "strict_radius_px": strict_radius,
            "min_overlap": ROI_CIRCLE_MIN_OVERLAP,
            "overlap_fraction": overlap,
            "inside_original_pixels": inside_pixels,
            "outside_original_pixels": int(circle_mask.sum()) - int(inside_pixels),
            "original_pixels": int(mask.sum()),
            "strict_circle_pixels": int(strict_mask.sum()),
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


def largest_tolerant_circle(
    mask: np.ndarray,
    min_overlap: float,
    max_radius: float,
    radius_step: float,
) -> tuple[np.ndarray, float, float, float, float, int]:
    """Largest circle whose center is in the ROI and whose pixels mostly overlap it."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        empty = np.zeros_like(mask, dtype=bool)
        return empty, np.nan, np.nan, np.nan, np.nan, 0

    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    candidates = np.argwhere(mask)
    best = None
    radii = np.arange(1.0, max_radius + 0.5 * radius_step, radius_step)
    for row, col in candidates:
        row_f = float(row)
        col_f = float(col)
        dist2 = (yy - row_f) ** 2 + (xx - col_f) ** 2
        for radius in radii:
            circle = dist2 <= float(radius) ** 2
            circle_pixels = int(circle.sum())
            if circle_pixels == 0:
                continue
            inside_pixels = int((circle & mask).sum())
            overlap = inside_pixels / float(circle_pixels)
            if overlap < min_overlap:
                continue
            score = (circle_pixels, float(radius), overlap)
            if best is None or score > best[0]:
                best = (score, circle.copy(), row_f, col_f, float(radius), overlap, inside_pixels)

    if best is None:
        circle, row, col, radius = largest_inscribed_circle(mask)
        return circle, row, col, radius, 1.0, int(circle.sum())

    _, circle, row, col, radius, overlap, inside_pixels = best
    return circle, row, col, radius, overlap, inside_pixels


def pd_image(raw: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Power Doppler in dB with robust limits from positive-power pixels."""
    img = 10.0 * np.log10(np.maximum(raw, 1e-12))
    finite_positive = img[np.isfinite(img) & (np.asarray(raw) > 0)]
    if finite_positive.size:
        vmin, vmax = np.percentile(finite_positive, PD_DB_PERCENTILES)
        if float(vmax) > float(vmin):
            return img.astype(np.float32), float(vmin), float(vmax)
    return img.astype(np.float32), -100.0, 140.0


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

    pd_img, pd_vmin, pd_vmax = pd_image(pd_raw)
    cd_img, cd_vmin, cd_vmax = signed_image(cd_raw, CD_ABS_PERCENTILE)

    dc_img, dc_vmin, dc_vmax = signed_image(dc_raw, DC_ABS_PERCENTILE)

    # Hero figure is cropped to the active lateral region.  Color scales are
    # computed from the full plane above, so the visible region renders
    # identically and only the lateral extent is trimmed.
    hero_x_range = (-1.3, 1.3)
    pd_img, extent = crop_lateral_cm(pd_img, data, hero_x_range)
    cd_img, _ = crop_lateral_cm(cd_img, data, hero_x_range)
    dc_img, _ = crop_lateral_cm(dc_img, data, hero_x_range)

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
    """Legacy temporal-stability figure for per-acquisition stacked NPZs."""
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

    # Common color scale from the displayed crop of the 250-buffer average.
    dc_full, extent = crop_lateral_cm(np.median(data["dower_coppler"][:250, PLANE], axis=0), data)
    _, common_vmin, common_vmax = signed_image(dc_full, 99.0)

    imshow_kw = dict(origin="lower", aspect="equal", extent=extent)

    for idx, (start, end, label) in enumerate(windows):
        dc_raw, _ = crop_lateral_cm(np.median(data["dower_coppler"][start:end, PLANE], axis=0), data)
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


def temporal_windows_from_sidecar(per_acq_dir: Path, plane: int) -> tuple[list[tuple[int, int, str, np.ndarray]], dict]:
    """Load the compact temporal-stability windows from per-acquisition sidecars."""
    acq_start = min(start for start, _, _ in TEMPORAL_WINDOWS)
    acq_end = max(end for _, end, _ in TEMPORAL_WINDOWS)
    acq_images = []
    acqs = []
    meta = {}
    for acq in range(acq_start, acq_end + 1):
        path = per_acq_dir / f"acq_{acq}.npz"
        data = load_npz(path)
        acq_images.append(np.asarray(data["dower_coppler"][plane], dtype=np.float32))
        acqs.append(acq)
        if not meta:
            meta = {k: data[k] for k in ("x_mm", "y_mm", "z_mm") if k in data}

    acqs_arr = np.asarray(acqs)
    stack = np.stack(acq_images, axis=0)
    images = []
    for start, end, label in TEMPORAL_WINDOWS:
        keep = (acqs_arr >= start) & (acqs_arr <= end)
        images.append((start, end, label, np.median(stack[keep], axis=0).astype(np.float32)))
    return images, meta


def save_temporal_summary(
    path: Path,
    images: list[tuple[int, int, str, np.ndarray]],
    meta: dict,
    plane: int,
    source_per_acq_dir: Path,
) -> None:
    """Save the compact data needed to reproduce the temporal-stability figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": np.stack([img for _, _, _, img in images], axis=0).astype(np.float32),
        "window_start_acq": np.asarray([start for start, _, _, _ in images], dtype=np.int32),
        "window_end_acq": np.asarray([end for _, end, _, _ in images], dtype=np.int32),
        "window_label": np.asarray([label for _, _, label, _ in images]),
        "plane": np.asarray(plane, dtype=np.int32),
        "source_per_acq_dir": np.asarray(str(source_per_acq_dir)),
        "generated_by": np.asarray("scripts/generate_paper_figures.py --refresh-temporal-summary"),
        "note": np.asarray("Median Dower Coppler maps for the temporal-stability windows used in Figure 6."),
    }
    for key in ("x_mm", "y_mm", "z_mm"):
        if key in meta:
            payload[key] = np.asarray(meta[key])
    np.savez_compressed(path, **payload)
    print(f"  Saved compact temporal summary to {path}")


def load_temporal_summary(path: Path) -> tuple[list[tuple[int, int, str, np.ndarray]], dict, int]:
    """Load compact temporal-stability data committed with the paper repo."""
    with np.load(path) as z:
        starts = np.asarray(z["window_start_acq"], dtype=int)
        ends = np.asarray(z["window_end_acq"], dtype=int)
        labels = [str(label) for label in np.asarray(z["window_label"])]
        imgs = np.asarray(z["images"], dtype=np.float32)
        plane = int(np.asarray(z["plane"]).item())
        meta = {k: z[k] for k in ("x_mm", "y_mm", "z_mm") if k in z}
    images = [(int(start), int(end), label, imgs[idx]) for idx, (start, end, label) in enumerate(zip(starts, ends, labels))]
    return images, meta, plane


def plot_temporal_stability_images(
    images: list[tuple[int, int, str, np.ndarray]],
    meta: dict,
    output_dir: Path,
    plane: int,
    source_label: str,
) -> None:
    """Render temporal-stability images after they have been reduced to windows."""
    data_for_extent = {k: np.asarray(v) for k, v in meta.items()}
    cropped_images = []
    for start, end, label, img in images:
        cropped_img, extent = crop_lateral_cm(img, data_for_extent)
        cropped_images.append((start, end, label, cropped_img))
    images = cropped_images
    finite_abs = np.concatenate([np.abs(img[np.isfinite(img)]).ravel() for _, _, _, img in images])
    common_lim = float(np.percentile(finite_abs, TEMPORAL_ABS_PERCENTILE)) if finite_abs.size else 1.0
    y_label = ""
    if "y_mm" in meta:
        y_mm = np.asarray(meta["y_mm"], dtype=float)
        if plane < y_mm.size:
            y_label = f" (y={y_mm[plane]:.2f} mm)"

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.3), constrained_layout=True)
    axes = axes.ravel()
    for ax, (start, end, label, img) in zip(axes, images):
        ax.imshow(
            img,
            cmap="seismic",
            vmin=-common_lim,
            vmax=common_lim,
            origin="lower",
            aspect="equal",
            extent=extent,
        )
        ax.set_title(f"{label}\nacqs {start}-{end}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Lateral (cm)", fontsize=8)
        ax.set_ylabel("Depth (cm)", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Dower Coppler temporal stability, fine-elevation plane {plane}{y_label}",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(output_dir / "fig_temporal_stability.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_temporal_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved temporal stability montage from {source_label}")


def figure_temporal_stability_from_sidecar(
    per_acq_dir: Path,
    output_dir: Path,
    plane: int = DEFAULT_TEMPORAL_PLANE,
    summary_path: Path | None = None,
):
    """Dower Coppler maps from per-acquisition sidecars with count-based labels."""
    images, meta = temporal_windows_from_sidecar(per_acq_dir, plane)
    if summary_path is not None:
        save_temporal_summary(summary_path, images, meta, plane, per_acq_dir)
    plot_temporal_stability_images(images, meta, output_dir, plane, str(per_acq_dir))


def figure_temporal_stability_from_summary(summary_path: Path, output_dir: Path):
    """Dower Coppler temporal-stability montage from the compact committed NPZ."""
    images, meta, plane = load_temporal_summary(summary_path)
    plot_temporal_stability_images(images, meta, output_dir, plane, str(summary_path))


def split_half_from_sidecar(
    per_acq_dir: Path,
    plane: int,
    split_a: tuple[int, int],
    split_b: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load two median Dower maps from per-acquisition sidecars."""
    meta = {}

    def load_window(start: int, end: int) -> np.ndarray:
        nonlocal meta
        images = []
        for acq in range(start, end + 1):
            data = load_npz(per_acq_dir / f"acq_{acq}.npz")
            images.append(np.asarray(data["dower_coppler"][plane], dtype=np.float32))
            if not meta:
                meta = {k: data[k] for k in ("x_mm", "y_mm", "z_mm") if k in data}
        return np.median(np.stack(images, axis=0), axis=0).astype(np.float32)

    return load_window(*split_a), load_window(*split_b), meta


def save_split_half_summary_from_sidecar(
    per_acq_dir: Path,
    path: Path,
    plane: int,
    split_a: tuple[int, int],
    split_b: tuple[int, int],
) -> None:
    """Save compact split-half maps using the current Dower formula."""
    split_a_img, split_b_img, meta = split_half_from_sidecar(per_acq_dir, plane, split_a, split_b)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "plane": np.asarray(plane, dtype=np.int32),
        "source_per_acq_dir": np.asarray(str(per_acq_dir)),
        "dower_coppler_split_a": split_a_img,
        "dower_coppler_split_b": split_b_img,
        "split_a_acqs": np.asarray(split_a, dtype=np.int32),
        "split_b_acqs": np.asarray(split_b, dtype=np.int32),
        "generated_by": np.asarray("scripts/generate_paper_figures.py --refresh-split-half-summary"),
        "note": np.asarray("Median Dower Coppler split-half maps regenerated from phase_velocity * geomean_r * phase_r2."),
    }
    for key in ("x_mm", "y_mm", "z_mm"):
        if key in meta:
            payload[key] = np.asarray(meta[key])
    np.savez_compressed(path, **payload)
    print(f"  Saved compact split-half summary to {path}")


def figure_split_half_consistency(split_path: Path, output_dir: Path) -> None:
    """Compare Dower Coppler maps from the first and second acquisition halves."""
    with np.load(split_path) as z:
        split_a = np.asarray(z["dower_coppler_split_a"], dtype=np.float32)
        split_b = np.asarray(z["dower_coppler_split_b"], dtype=np.float32)
        split_a_acqs = tuple(int(v) for v in np.asarray(z["split_a_acqs"]))
        split_b_acqs = tuple(int(v) for v in np.asarray(z["split_b_acqs"]))
        plane = int(np.asarray(z["plane"]).item())
        meta = {k: z[k] for k in ("x_mm", "y_mm", "z_mm") if k in z}

    valid = np.isfinite(split_a) & np.isfinite(split_b) & (split_a != 0) & (split_b != 0)
    absmax = np.maximum(np.abs(split_a), np.abs(split_b))
    if valid.any():
        threshold = float(np.percentile(absmax[valid], 100.0 - SPLIT_HALF_AGREEMENT_TOP_PCT))
        top_mask = valid & (absmax >= threshold)
        agree = np.sign(split_a) == np.sign(split_b)
        top_agreement = float(np.mean(agree[top_mask])) if top_mask.any() else float("nan")
        all_agreement = float(np.mean(agree[valid]))
    else:
        top_mask = np.zeros_like(valid)
        agree = np.zeros_like(valid)
        top_agreement = float("nan")
        all_agreement = float("nan")

    agreement_img = np.full(split_a.shape, np.nan, dtype=np.float32)
    agreement_img[top_mask & ~agree] = 0.0
    agreement_img[top_mask & agree] = 1.0

    data_for_extent = {k: np.asarray(v) for k, v in meta.items()}
    split_a_crop, extent = crop_lateral_cm(split_a, data_for_extent)
    split_b_crop, _ = crop_lateral_cm(split_b, data_for_extent)
    agreement_crop, _ = crop_lateral_cm(agreement_img, data_for_extent)
    finite_abs = np.concatenate([
        np.abs(split_a_crop[np.isfinite(split_a_crop)]).ravel(),
        np.abs(split_b_crop[np.isfinite(split_b_crop)]).ravel(),
    ])
    lim = float(np.percentile(finite_abs, SPLIT_HALF_ABS_PERCENTILE)) if finite_abs.size else 1.0

    y_label = ""
    if "y_mm" in meta:
        y_mm = np.asarray(meta["y_mm"], dtype=float)
        if plane < y_mm.size:
            y_label = f", y={y_mm[plane]:.2f} mm"

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    panels = [
        (axes[0], split_a_crop, f"First half\nacqs {split_a_acqs[0]}-{split_a_acqs[1]}"),
        (axes[1], split_b_crop, f"Second half\nacqs {split_b_acqs[0]}-{split_b_acqs[1]}"),
    ]
    for ax, img, title in panels:
        im = ax.imshow(
            img,
            cmap="seismic",
            vmin=-lim,
            vmax=lim,
            origin="lower",
            aspect="equal",
            extent=extent,
        )
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Lateral (cm)", fontsize=8)
        ax.set_ylabel("Depth (cm)", fontsize=8)
        ax.tick_params(labelsize=7)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.04)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=7)

    cmap = ListedColormap(["#8b5cf6", "#16a34a"])
    cmap.set_bad("#eeeeee")
    axes[2].imshow(
        agreement_crop,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        aspect="equal",
        extent=extent,
    )
    axes[2].set_title(
        f"Sign agreement\n"
        f"top {SPLIT_HALF_AGREEMENT_TOP_PCT}% |Dower|: {100.0 * top_agreement:.1f}%",
        fontsize=10,
        fontweight="bold",
    )
    axes[2].set_xlabel("Lateral (cm)", fontsize=8)
    axes[2].set_ylabel("Depth (cm)", fontsize=8)
    axes[2].tick_params(labelsize=7)
    axes[2].text(
        0.02,
        0.02,
        f"green=agree, purple=disagree\nall finite nonzero: {100.0 * all_agreement:.1f}%",
        transform=axes[2].transAxes,
        fontsize=7,
        color="black",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
    )

    fig.suptitle(f"Split-half Dower Coppler consistency, plane {plane}{y_label}", fontsize=12, fontweight="bold")
    fig.savefig(output_dir / "fig_split_half_consistency.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_split_half_consistency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved split-half consistency figure from {split_path}")


# ── Figure: All elevation planes ──────────────────────────────────────


def figure_all_planes(
    data: dict,
    output_dir: Path,
    acq_start: int = 0,
    acq_end: int = 480,
    plane_start: int = DEFAULT_ALL_PLANES_START,
    plane_end: int = DEFAULT_ALL_PLANES_END,
):
    """Dower Coppler maps for a selected inclusive range of elevation planes."""
    arr = np.asarray(data["dower_coppler"])
    n_planes = arr.shape[1] if arr.ndim == 4 else arr.shape[0] if arr.ndim == 3 else 1
    plane_start = max(0, min(int(plane_start), n_planes - 1))
    plane_end = max(plane_start, min(int(plane_end), n_planes - 1))
    planes = list(range(plane_start, plane_end + 1))
    ncols = 2
    nrows = (len(planes) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, nrows * 2.0), constrained_layout=True)
    axes = axes.ravel()

    plane_images = []
    extent = None
    for plane in planes:
        img, extent = crop_lateral_cm(metric_plane(data, "dower_coppler", plane, acq_start, acq_end), data)
        plane_images.append(img)
    finite_abs = np.concatenate([np.abs(img[np.isfinite(img)]).ravel() for img in plane_images])
    common_lim = float(np.percentile(finite_abs, 97.0)) if finite_abs.size else 1.0

    for ax_idx, (plane, dc_full) in enumerate(zip(planes, plane_images)):
        title = f"Plane {plane}"
        if "y_mm" in data:
            y_mm = np.asarray(data["y_mm"], dtype=float)
            if plane < y_mm.size:
                title += f" ({y_mm[plane]:.1f} mm)"
        axes[ax_idx].imshow(dc_full.astype(np.float32), origin="lower", aspect="equal",
                            extent=extent,
                            cmap="seismic", vmin=-common_lim, vmax=common_lim)
        axes[ax_idx].set_title(title, fontsize=10, fontweight="bold")
        axes[ax_idx].set_xlabel("Lateral (cm)", fontsize=8)
        axes[ax_idx].set_ylabel("Depth (cm)", fontsize=8)
        axes[ax_idx].tick_params(labelsize=7)

    # Hide any unused axes
    for idx in range(len(planes), len(axes)):
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
    cd_signed_meas = cd_raw.astype(np.float32)
    dc_signed_meas = dc_raw.astype(np.float32)

    if pd_raw.shape != bg_mask.shape:
        raise ValueError(f"Region export shape {bg_mask.shape} does not match image shape {pd_raw.shape}")

    roi_labels = [f"V{i+1}" for i in range(len(signal_masks))]
    cnr_pd = []
    cnr_cd = []
    cnr_dc = []
    gcnr_pd = []
    gcnr_cd = []
    gcnr_dc = []
    signed_gcnr_cd = []
    signed_gcnr_dc = []
    for idx, sig_mask in enumerate(signal_masks):
        this_bg_mask = bg_mask & ~sig_mask

        cnr_pd.append(compute_cnr(pd_meas[sig_mask], pd_meas[this_bg_mask]))
        cnr_cd.append(compute_cnr(cd_meas[sig_mask], cd_meas[this_bg_mask]))
        cnr_dc.append(compute_cnr(dc_meas[sig_mask], dc_meas[this_bg_mask]))

        gcnr_pd.append(compute_gcnr(pd_meas[sig_mask], pd_meas[this_bg_mask]))
        gcnr_cd.append(compute_gcnr(cd_meas[sig_mask], cd_meas[this_bg_mask]))
        gcnr_dc.append(compute_gcnr(dc_meas[sig_mask], dc_meas[this_bg_mask]))
        signed_gcnr_cd.append(compute_gcnr(cd_signed_meas[sig_mask], cd_signed_meas[this_bg_mask]))
        signed_gcnr_dc.append(compute_gcnr(dc_signed_meas[sig_mask], dc_signed_meas[this_bg_mask]))

        circle = region_info["inscribed_circles"][idx]
        print(
            f"    {roi_labels[idx]}: center=({circle['center_col']:.0f},{circle['center_row']:.0f}), "
            f"radius={circle['radius_px']:.2f}px, signal_pixels={int(sig_mask.sum())}, "
            f"overlap={circle['overlap_fraction']:.2f}, inside={circle['inside_original_pixels']}, "
            f"outside={circle['outside_original_pixels']}, original_pixels={circle['original_pixels']}, "
            f"background_pixels={int(this_bg_mask.sum())}"
        )

    # Plot CNR bar chart
    x = np.arange(len(signal_masks))
    bar_w = 0.25

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    cnr_bar_w = 0.32
    ax1.bar(x - cnr_bar_w / 2, cnr_pd, cnr_bar_w, label="Power Doppler", color="#B45309", alpha=0.85)
    ax1.bar(x + cnr_bar_w / 2, cnr_dc, cnr_bar_w, label="Dower Coppler", color="#DC2626", alpha=0.85)
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
    ax2.set_title("Magnitude gCNR", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(roi_labels)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.1)

    signed_bar_w = 0.32
    ax3.bar(x - signed_bar_w / 2, signed_gcnr_cd, signed_bar_w, label="Color Doppler (Kasai)", color="#6366F1", alpha=0.85)
    ax3.bar(x + signed_bar_w / 2, signed_gcnr_dc, signed_bar_w, label="Dower Coppler", color="#DC2626", alpha=0.85)
    ax3.set_xlabel("Vessel ROI", fontsize=10)
    ax3.set_ylabel("signed gCNR", fontsize=10)
    ax3.set_title("Directional gCNR", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(roi_labels)
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 1.1)

    fig.savefig(output_dir / "fig_cnr_comparison.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_cnr_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Print CNR table
    print("  CNR (dB) by ROI:")
    print(f"  {'ROI':>4s}  {'PD':>7s}  {'CD':>7s}  {'DC':>7s}  {'PD gCNR':>7s}  {'CD gCNR':>7s}  {'DC gCNR':>7s}  {'CD sgCNR':>8s}  {'DC sgCNR':>8s}")
    for i, label in enumerate(roi_labels):
        print(
            f"  {label:>4s}  {cnr_pd[i]:7.1f}  {cnr_cd[i]:7.1f}  {cnr_dc[i]:7.1f}  "
            f"{gcnr_pd[i]:7.2f}  {gcnr_cd[i]:7.2f}  {gcnr_dc[i]:7.2f}  "
            f"{signed_gcnr_cd[i]:8.2f}  {signed_gcnr_dc[i]:8.2f}"
        )
    stats_dir = output_dir.parent / "paper_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    cnr_payload = {
        "source_regions": str(regions_path),
        "plane": int(plane),
        "roi_labels": roi_labels,
        "metrics": {
            "cnr_db": {
                "power_doppler": [float(v) for v in cnr_pd],
                "color_doppler_abs": [float(v) for v in cnr_cd],
                "dower_coppler_abs": [float(v) for v in cnr_dc],
            },
            "gcnr": {
                "power_doppler": [float(v) for v in gcnr_pd],
                "color_doppler_abs": [float(v) for v in gcnr_cd],
                "dower_coppler_abs": [float(v) for v in gcnr_dc],
                "color_doppler_signed": [float(v) for v in signed_gcnr_cd],
                "dower_coppler_signed": [float(v) for v in signed_gcnr_dc],
            },
        },
        "summary": {
            "median_cnr_db": {
                "power_doppler": float(np.nanmedian(cnr_pd)),
                "color_doppler_abs": float(np.nanmedian(cnr_cd)),
                "dower_coppler_abs": float(np.nanmedian(cnr_dc)),
            },
            "range_cnr_db": {
                "power_doppler": [float(np.nanmin(cnr_pd)), float(np.nanmax(cnr_pd))],
                "color_doppler_abs": [float(np.nanmin(cnr_cd)), float(np.nanmax(cnr_cd))],
                "dower_coppler_abs": [float(np.nanmin(cnr_dc)), float(np.nanmax(cnr_dc))],
            },
        },
    }
    (stats_dir / "cnr_gcnr_comparison.json").write_text(json.dumps(cnr_payload, indent=2) + "\n")
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

    fig = plt.figure(figsize=(10.2, 5.1), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.05, 1.10])
    ax_color = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[1, 0])
    ax_ba = fig.add_subplot(gs[:, 1])

    imshow_kw = dict(origin="lower", aspect="equal", extent=extent, cmap="seismic", vmin=-lim, vmax=lim)
    im0 = ax_color.imshow(color_mm_s, **imshow_kw)
    ax_color.set_title("Kasai color Doppler", fontsize=11, fontweight="bold")
    ax_phase.imshow(phase_mm_s, **imshow_kw)
    ax_phase.set_title("Multi-lag phase velocity", fontsize=11, fontweight="bold")
    for ax in (ax_color, ax_phase):
        ax.set_xlabel("Lateral (cm)", fontsize=9)
        ax.set_ylabel("Depth (cm)", fontsize=9)
        ax.tick_params(labelsize=8)
    cbar = fig.colorbar(im0, ax=[ax_color, ax_phase], shrink=0.9, pad=0.02)
    cbar.set_label("Velocity (mm/s)", fontsize=9)

    ax_ba.scatter(mean, diff, s=3, alpha=0.18, color="#2563eb", edgecolors="none")
    ax_ba.axhline(bias, color="#d97706", linestyle="--", linewidth=1.2, label=f"bias {bias:.2f}")
    ax_ba.axhline(loa_low, color="#dc2626", linestyle="--", linewidth=1.0, label="-1.96 SD")
    ax_ba.axhline(loa_high, color="#dc2626", linestyle="--", linewidth=1.0, label="+1.96 SD")
    ax_ba.set_title("Bland-Altman", fontsize=11, fontweight="bold")
    ax_ba.set_xlabel("Mean velocity (mm/s)", fontsize=9)
    ax_ba.set_ylabel("Multi-lag - Kasai (mm/s)", fontsize=9)
    ax_ba.tick_params(labelsize=8)
    ax_ba.grid(True, alpha=0.25, linewidth=0.5)
    ax_ba.legend(fontsize=7, loc="upper right", frameon=False)
    ax_ba.text(
        0.03,
        0.03,
        f"n={mean.size:,}\nSD={sd:.2f} mm/s\nLoA [{loa_low:.1f}, {loa_high:.1f}]",
        transform=ax_ba.transAxes,
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
    if "dower_coppler" in data:
        dower = metric_plane(data, "dower_coppler", plane_idx)
    else:
        dower = v_phi * g_r * r2

    v_phi, extent = crop_lateral_cm(v_phi, data)
    g_r, _ = crop_lateral_cm(g_r, data)
    r2, _ = crop_lateral_cm(r2, data)
    v_phi_g_r, _ = crop_lateral_cm(v_phi_g_r, data)
    v_phi_r2, _ = crop_lateral_cm(v_phi_r2, data)
    dower, _ = crop_lateral_cm(dower, data)
    panels = [
        (r"$v_\phi$ alone", v_phi, "seismic", "signed"),
        (r"$G_R$ alone", g_r, "magma", "unsigned"),
        (r"$R^2$ alone", r2, "viridis", "unit"),
        (r"$v_\phi \cdot G_R$", v_phi_g_r, "seismic", "signed"),
        (r"$v_\phi \cdot R^2$", v_phi_r2, "seismic", "signed"),
        (r"$v_\phi \cdot G_R \cdot R^2$", dower, "seismic", "signed"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 5.8), constrained_layout=False)
    axes = axes.ravel()
    for ax, (title, img, cmap, mode) in zip(axes, panels):
        if mode == "signed":
            img_plot, vmin, vmax = signed_image(img, DC_ABS_PERCENTILE)
        elif mode == "unit":
            img_plot = img.astype(np.float32)
            vmin, vmax = 0.0, 1.0
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
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.04)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Dower Coppler ablation components", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.10, top=0.86, wspace=0.34, hspace=0.55)
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
    compound_rate = float(np.asarray(data.get("compound_frame_rate_hz", frame_rate)).item())

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
        f"middle plane {plane_idx}; {acq_count} acquisitions; cadence {compound_rate:.1f} Hz",
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
    parser.add_argument("--main-data", type=Path, default=DEFAULT_MAIN_DATA_PATH)
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument("--all-planes-data", type=Path, default=DEFAULT_ALL_PLANES_PATH)
    parser.add_argument("--all-planes-start", type=int, default=DEFAULT_ALL_PLANES_START)
    parser.add_argument("--all-planes-end", type=int, default=DEFAULT_ALL_PLANES_END)
    parser.add_argument("--temporal-summary-data", type=Path, default=DEFAULT_TEMPORAL_SUMMARY_PATH)
    parser.add_argument("--split-half-data", type=Path, default=DEFAULT_SPLIT_HALF_PATH)
    parser.add_argument("--temporal-per-acq-dir", type=Path, default=DEFAULT_TEMPORAL_PER_ACQ_DIR)
    parser.add_argument("--temporal-plane", type=int, default=DEFAULT_TEMPORAL_PLANE)
    parser.add_argument("--split-a", nargs=2, type=int, default=DEFAULT_SPLIT_A, metavar=("START", "END"))
    parser.add_argument("--split-b", nargs=2, type=int, default=DEFAULT_SPLIT_B, metavar=("START", "END"))
    parser.add_argument(
        "--refresh-temporal-summary",
        action="store_true",
        help="Recompute the compact temporal-stability NPZ from --temporal-per-acq-dir.",
    )
    parser.add_argument(
        "--refresh-split-half-summary",
        action="store_true",
        help="Recompute the compact split-half NPZ from --temporal-per-acq-dir.",
    )
    parser.add_argument("--external-recording-data", type=Path, default=DEFAULT_EXTERNAL_RECORDING_PATH)
    parser.add_argument(
        "--external-recording-plane",
        type=int,
        default=-1,
        help="Elevation plane for the external-recording figure; -1 selects the middle plane.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_acq_path = args.main_data
    if not per_acq_path.is_absolute():
        per_acq_path = ROOT / per_acq_path
    print("Loading per-acq dataset...")
    per_acq = load_npz(per_acq_path)
    print(f"  Path: {per_acq_path}")
    print(f"  Shape: {per_acq['power_doppler'].shape}")
    validate_in_vivo_baselines(per_acq, per_acq_path)

    print("Loading all-planes dataset...")
    all_planes = load_npz(args.all_planes_data)
    print(f"  Path: {args.all_planes_data}")
    print(f"  Shape: {all_planes['dower_coppler'].shape}")

    print("\nGenerating figures:")
    figure_three_panel(per_acq, args.output_dir)
    if args.refresh_temporal_summary:
        if not args.temporal_per_acq_dir.exists():
            raise FileNotFoundError(f"Temporal sidecar not found: {args.temporal_per_acq_dir}")
        figure_temporal_stability_from_sidecar(
            args.temporal_per_acq_dir,
            args.output_dir,
            plane=args.temporal_plane,
            summary_path=args.temporal_summary_data,
        )
    elif args.temporal_summary_data.exists():
        figure_temporal_stability_from_summary(args.temporal_summary_data, args.output_dir)
    elif args.temporal_per_acq_dir.exists():
        figure_temporal_stability_from_sidecar(
            args.temporal_per_acq_dir,
            args.output_dir,
            plane=args.temporal_plane,
        )
    elif np.asarray(per_acq["dower_coppler"]).ndim == 4:
        figure_temporal_stability(per_acq, args.output_dir)
    else:
        print("  Skipping temporal stability montage: no per-acq sidecar and viewer dataset is already averaged")

    if args.refresh_split_half_summary:
        if not args.temporal_per_acq_dir.exists():
            raise FileNotFoundError(f"Temporal sidecar not found: {args.temporal_per_acq_dir}")
        save_split_half_summary_from_sidecar(
            args.temporal_per_acq_dir,
            args.split_half_data,
            args.temporal_plane,
            tuple(args.split_a),
            tuple(args.split_b),
        )
        figure_split_half_consistency(args.split_half_data, args.output_dir)
    elif args.split_half_data.exists():
        figure_split_half_consistency(args.split_half_data, args.output_dir)
    else:
        print(f"  Skipping split-half consistency figure: {args.split_half_data} not found")

    figure_cnr_comparison(per_acq, args.output_dir, args.regions, acq_start=0, acq_end=480)
    figure_three_panel_with_rois(per_acq, args.output_dir, args.regions, acq_start=0, acq_end=480)
    figure_dower_ablation(per_acq, args.output_dir)
    figure_velocity_vs_color(per_acq, args.output_dir, plane=0)

    figure_all_planes(
        all_planes,
        args.output_dir,
        plane_start=args.all_planes_start,
        plane_end=args.all_planes_end,
    )

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
