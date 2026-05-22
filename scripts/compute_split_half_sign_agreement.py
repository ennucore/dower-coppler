#!/usr/bin/env python3
"""Compute split-half sign agreement for the paper ROI masks.

This script intentionally imports the ROI loader from generate_paper_figures.py
so the split-half statistics use the same tolerant circular ROIs as the CNR
figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import generate_paper_figures as gpf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "paper_stats" / "split_half_sign_agreement.json"
DEFAULT_SPLIT_SUMMARY = ROOT / "data" / "head_2025-09-21_split_half_plane4_all480.npz"


def load_metric_window(per_acq_dir: Path, metric: str, plane: int, start: int, end: int) -> np.ndarray:
    images = []
    for acq in range(start, end + 1):
        data = gpf.load_npz(per_acq_dir / f"acq_{acq}.npz")
        images.append(np.asarray(data[metric][plane], dtype=np.float32))
    return np.median(np.stack(images, axis=0), axis=0)


def sign_agreement(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    valid = mask & np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    count = int(valid.sum())
    if count == 0:
        return float("nan"), 0
    return float(np.mean(np.sign(a[valid]) == np.sign(b[valid]))), count


def metric_summary(
    per_acq_dir: Path,
    metric: str,
    plane: int,
    split_a: tuple[int, int],
    split_b: tuple[int, int],
    signal_masks: list[np.ndarray],
    bg_mask: np.ndarray,
) -> dict:
    a = load_metric_window(per_acq_dir, metric, plane, *split_a)
    b = load_metric_window(per_acq_dir, metric, plane, *split_b)

    per_roi = []
    all_vessel_mask = np.zeros_like(signal_masks[0], dtype=bool)
    for idx, mask in enumerate(signal_masks, start=1):
        agree, finite_nonzero = sign_agreement(a, b, mask)
        per_roi.append({
            "roi": f"V{idx}",
            "pixels": int(mask.sum()),
            "finite_nonzero_pixels": finite_nonzero,
            "sign_agreement": agree,
            "split_a_mean": float(np.nanmean(a[mask])),
            "split_b_mean": float(np.nanmean(b[mask])),
        })
        all_vessel_mask |= mask

    vessel_agree, vessel_finite_nonzero = sign_agreement(a, b, all_vessel_mask)
    bg_values = np.concatenate([
        np.abs(a[bg_mask][np.isfinite(a[bg_mask])]),
        np.abs(b[bg_mask][np.isfinite(b[bg_mask])]),
    ])
    bg95 = float(np.percentile(bg_values, 95.0)) if bg_values.size else float("nan")
    if np.isfinite(bg95):
        active_vessel = all_vessel_mask & ((np.abs(a) > bg95) | (np.abs(b) > bg95))
        active_bg = bg_mask & ((np.abs(a) > bg95) | (np.abs(b) > bg95))
    else:
        active_vessel = np.zeros_like(all_vessel_mask, dtype=bool)
        active_bg = np.zeros_like(bg_mask, dtype=bool)
    active_vessel_agree, active_vessel_count = sign_agreement(a, b, active_vessel)
    active_bg_agree, active_bg_count = sign_agreement(a, b, active_bg)

    out = {
        "per_roi": per_roi,
        "all_vessel_pixels": int(all_vessel_mask.sum()),
        "vessel_finite_nonzero_pixels": vessel_finite_nonzero,
        "vessel_sign_agreement": vessel_agree,
        "active_vessel_pixels_thresholded_by_bg95": active_vessel_count,
        "active_vessel_sign_agreement": active_vessel_agree,
        "background_abs95_threshold": bg95,
        "background_active_pixels": active_bg_count,
        "background_active_sign_agreement": active_bg_agree,
    }

    if metric == "dower_coppler":
        valid = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
        absmax = np.maximum(np.abs(a), np.abs(b))
        whole_plane = {
            "all_finite_nonzero_pixels": int(valid.sum()),
            "all_finite_nonzero_sign_agreement": (
                float(np.mean(np.sign(a[valid]) == np.sign(b[valid]))) if valid.any() else float("nan")
            ),
        }
        finite_abs = absmax[valid]
        for pct in (50, 25, 10, 5, 3, 1):
            if finite_abs.size:
                threshold = float(np.percentile(finite_abs, 100 - pct))
                top_mask = valid & (absmax >= threshold)
                whole_plane[f"top_{pct}pct_abs_pixels"] = int(top_mask.sum())
                whole_plane[f"top_{pct}pct_abs_sign_agreement"] = (
                    float(np.mean(np.sign(a[top_mask]) == np.sign(b[top_mask]))) if top_mask.any() else float("nan")
                )
            else:
                whole_plane[f"top_{pct}pct_abs_pixels"] = 0
                whole_plane[f"top_{pct}pct_abs_sign_agreement"] = float("nan")
        out[f"whole_plane_plane{plane}"] = whole_plane

    return out


def metric_summary_from_arrays(
    a: np.ndarray,
    b: np.ndarray,
    metric: str,
    plane: int,
    signal_masks: list[np.ndarray],
    bg_mask: np.ndarray,
) -> dict:
    per_roi = []
    all_vessel_mask = np.zeros_like(signal_masks[0], dtype=bool)
    for idx, mask in enumerate(signal_masks, start=1):
        agree, finite_nonzero = sign_agreement(a, b, mask)
        per_roi.append({
            "roi": f"V{idx}",
            "pixels": int(mask.sum()),
            "finite_nonzero_pixels": finite_nonzero,
            "sign_agreement": agree,
            "split_a_mean": float(np.nanmean(a[mask])),
            "split_b_mean": float(np.nanmean(b[mask])),
        })
        all_vessel_mask |= mask

    vessel_agree, vessel_finite_nonzero = sign_agreement(a, b, all_vessel_mask)
    bg_values = np.concatenate([
        np.abs(a[bg_mask][np.isfinite(a[bg_mask])]),
        np.abs(b[bg_mask][np.isfinite(b[bg_mask])]),
    ])
    bg95 = float(np.percentile(bg_values, 95.0)) if bg_values.size else float("nan")
    if np.isfinite(bg95):
        active_vessel = all_vessel_mask & ((np.abs(a) > bg95) | (np.abs(b) > bg95))
        active_bg = bg_mask & ((np.abs(a) > bg95) | (np.abs(b) > bg95))
    else:
        active_vessel = np.zeros_like(all_vessel_mask, dtype=bool)
        active_bg = np.zeros_like(bg_mask, dtype=bool)
    active_vessel_agree, active_vessel_count = sign_agreement(a, b, active_vessel)
    active_bg_agree, active_bg_count = sign_agreement(a, b, active_bg)

    out = {
        "per_roi": per_roi,
        "all_vessel_pixels": int(all_vessel_mask.sum()),
        "vessel_finite_nonzero_pixels": vessel_finite_nonzero,
        "vessel_sign_agreement": vessel_agree,
        "active_vessel_pixels_thresholded_by_bg95": active_vessel_count,
        "active_vessel_sign_agreement": active_vessel_agree,
        "background_abs95_threshold": bg95,
        "background_active_pixels": active_bg_count,
        "background_active_sign_agreement": active_bg_agree,
    }

    if metric == "dower_coppler":
        valid = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
        absmax = np.maximum(np.abs(a), np.abs(b))
        whole_plane = {
            "all_finite_nonzero_pixels": int(valid.sum()),
            "all_finite_nonzero_sign_agreement": (
                float(np.mean(np.sign(a[valid]) == np.sign(b[valid]))) if valid.any() else float("nan")
            ),
        }
        finite_abs = absmax[valid]
        for pct in (50, 25, 10, 5, 3, 1):
            if finite_abs.size:
                threshold = float(np.percentile(finite_abs, 100 - pct))
                top_mask = valid & (absmax >= threshold)
                whole_plane[f"top_{pct}pct_abs_pixels"] = int(top_mask.sum())
                whole_plane[f"top_{pct}pct_abs_sign_agreement"] = (
                    float(np.mean(np.sign(a[top_mask]) == np.sign(b[top_mask]))) if top_mask.any() else float("nan")
                )
            else:
                whole_plane[f"top_{pct}pct_abs_pixels"] = 0
                whole_plane[f"top_{pct}pct_abs_sign_agreement"] = float("nan")
        out[f"whole_plane_plane{plane}"] = whole_plane

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paper split-half sign agreement")
    parser.add_argument("--per-acq-dir", type=Path, default=gpf.DEFAULT_TEMPORAL_PER_ACQ_DIR)
    parser.add_argument("--split-summary", type=Path, default=DEFAULT_SPLIT_SUMMARY)
    parser.add_argument("--regions", type=Path, default=gpf.DEFAULT_REGIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-a", nargs=2, type=int, default=(0, 239), metavar=("START", "END"))
    parser.add_argument("--split-b", nargs=2, type=int, default=(240, 479), metavar=("START", "END"))
    args = parser.parse_args()

    region_info, signal_masks, bg_mask = gpf.load_region_export(args.regions)
    split_a = tuple(args.split_a)
    split_b = tuple(args.split_b)

    split_data = gpf.load_npz(args.split_summary) if args.split_summary.exists() else None
    if split_data is not None:
        y_mm = np.asarray(split_data.get("y_mm", np.arange(gpf.DEFAULT_TEMPORAL_PLANE + 1)), dtype=float)
        plane = int(np.asarray(split_data.get("plane", gpf.DEFAULT_TEMPORAL_PLANE)).item())
        roi_y = float(region_info.get("elevation_y_mm", y_mm[plane] if plane < y_mm.size else plane))
        split_a = tuple(int(x) for x in np.asarray(split_data.get("split_a_acqs", split_a)))
        split_b = tuple(int(x) for x in np.asarray(split_data.get("split_b_acqs", split_b)))
        source = str(split_data.get("source_per_acq_dir", args.split_summary))
    else:
        first = gpf.load_npz(args.per_acq_dir / f"acq_{split_a[0]}.npz")
        y_mm = np.asarray(first.get("y_mm", np.arange(first["dower_coppler"].shape[0])), dtype=float)
        roi_y = float(region_info.get("elevation_y_mm", y_mm[gpf.DEFAULT_TEMPORAL_PLANE]))
        plane = int(np.argmin(np.abs(y_mm - roi_y))) if y_mm.size else gpf.DEFAULT_TEMPORAL_PLANE
        source = str(args.per_acq_dir)

    summary = {
        "source_per_acq_dir": source,
        "split_a_acqs": list(split_a),
        "split_b_acqs": list(split_b),
        "split_a_count": split_a[1] - split_a[0] + 1,
        "split_b_count": split_b[1] - split_b[0] + 1,
        "roi_y_mm_reference": roi_y,
        "fine_elev_plane": plane,
        "fine_elev_plane_y_mm": float(y_mm[plane]) if y_mm.size else None,
        "roi_region_export": str(args.regions.relative_to(ROOT)) if args.regions.is_relative_to(ROOT) else str(args.regions),
        "roi_circle_mode": {
            "type": "largest tolerant circle centered inside exported mask",
            "min_overlap": gpf.ROI_CIRCLE_MIN_OVERLAP,
            "max_radius_px": gpf.ROI_CIRCLE_MAX_RADIUS_PX,
            "radius_step_px": gpf.ROI_CIRCLE_RADIUS_STEP_PX,
            "circles": region_info["inscribed_circles"],
        },
        "metrics": {},
    }
    if split_data is not None:
        for metric in ("dower_coppler", "phase_velocity", "color_doppler"):
            if f"{metric}_split_a" not in split_data or f"{metric}_split_b" not in split_data:
                continue
            summary["metrics"][metric] = metric_summary_from_arrays(
                np.asarray(split_data[f"{metric}_split_a"], dtype=np.float32),
                np.asarray(split_data[f"{metric}_split_b"], dtype=np.float32),
                metric,
                plane,
                signal_masks,
                bg_mask,
            )
    else:
        summary["metrics"] = {
            "dower_coppler": metric_summary(args.per_acq_dir, "dower_coppler", plane, split_a, split_b, signal_masks, bg_mask),
            "phase_velocity": metric_summary(args.per_acq_dir, "phase_velocity", plane, split_a, split_b, signal_masks, bg_mask),
            "color_doppler": metric_summary(args.per_acq_dir, "color_doppler", plane, split_a, split_b, signal_masks, bg_mask),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    dower = summary["metrics"]["dower_coppler"]
    print(
        "Saved split-half sign agreement: "
        f"{dower['vessel_sign_agreement']:.3f} across "
        f"{dower['vessel_finite_nonzero_pixels']} finite nonzero vessel pixels"
    )


if __name__ == "__main__":
    main()
