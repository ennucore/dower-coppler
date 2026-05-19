#!/usr/bin/env python3
"""Recompute the Sep 21 middle-8 cached-compound Doppler NPZ used for fig_all_planes.

This script operates on the cached beamformed `compound_image` arrays inside the
ultratrace H5. It does not redo raw IQ beamforming. It is the provenance script
for `data/head_2025-09-21_new_h5_recomputed_dower_acq200_399_mid8elev.npz`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from caterpillar.imaging.doppler import _lag_autocorrelations, _unwrap_lag_phases, lag_phase_linear_fit, svd_filter_fast

LOW_CUTOFF = 0.08
MAX_LAG = 5
SOUND_SPEED = 1600.0
HUBER_DELTA = 0.7
HUBER_ITERATIONS = 5


def metadata(h5: h5py.File, start: int) -> tuple[float, float]:
    meta = h5[f"acquisitions/{start}/meta"]
    cfg_raw = meta["acquisition_config"][()]
    rt_raw = meta["runtime_metadata"][()]
    if isinstance(cfg_raw, bytes):
        cfg_raw = cfg_raw.decode()
    if isinstance(rt_raw, bytes):
        rt_raw = rt_raw.decode()
    cfg = json.loads(cfg_raw)
    rt = json.loads(rt_raw)
    frame_rate = float(rt.get("empirical_pulse_repetition_rate_hz") or (cfg["requested_prf_hz"] / cfg["num_angles"]))
    return frame_rate, float(cfg["tx_freq_hz"])


def wls_origin(phases: torch.Tensor, weights: torch.Tensor, lags: list[int]) -> torch.Tensor:
    lag_values = torch.as_tensor(lags, device=phases.device, dtype=phases.dtype)
    lag_values = lag_values.view(-1, *([1] * (phases.ndim - 1)))
    return (weights * lag_values * phases).sum(dim=0) / (weights * lag_values.square()).sum(dim=0).clamp_min(1e-12)


def all_metrics(stack_np: np.ndarray, frame_rate: float, tx_freq: float, device: torch.device):
    with torch.no_grad():
        sig = torch.from_numpy(stack_np.astype(np.complex64, copy=False)).to(device)
        sig = sig - sig.mean(dim=0, keepdim=True)
        filt = svd_filter_fast(sig, low_cutoff=LOW_CUTOFF, high_cutoff=1.0)
        lags, rk = _lag_autocorrelations(filt, MAX_LAG)
        phases = _unwrap_lag_phases(torch.angle(rk))
        weights = rk.abs()
        slope, r2 = lag_phase_linear_fit(filt, max_lag=MAX_LAG, weighted=True, fit_intercept=False)
        velocity = slope * (float(frame_rate) / (2.0 * np.pi)) * (SOUND_SPEED / (2.0 * float(tx_freq)))
        geomean_r = torch.exp(torch.log(weights.clamp_min(1e-30)).mean(dim=0))
        signed_scale = (slope / np.pi).clamp(-1.0, 1.0)
        huber_slope = slope
        lag_values = torch.as_tensor(lags, device=device, dtype=phases.dtype).view(-1, *([1] * (phases.ndim - 1)))
        for _ in range(HUBER_ITERATIONS):
            residual = phases - lag_values * huber_slope
            robust = torch.clamp(HUBER_DELTA / residual.abs().clamp_min(1e-6), max=1.0)
            huber_slope = wls_origin(phases, weights * robust, lags)
        residual = phases - lag_values * huber_slope
        robust = torch.clamp(HUBER_DELTA / residual.abs().clamp_min(1e-6), max=1.0)
        huber_quality = ((weights * robust).sum(dim=0) / weights.sum(dim=0).clamp_min(1e-12)).clamp(0.0, 1.0)
        signed_scale_huber = (huber_slope / np.pi).clamp(-1.0, 1.0)
        dower = velocity * geomean_r * r2
        out = {
            "dower_coppler": dower,
            "phase_velocity": velocity,
            "phase_velocity_r2": velocity * r2,
            "phase_r2": r2,
            "signed_scale": signed_scale,
            "huber_quality": huber_quality,
            "geomean_r": geomean_r,
            "signed_scale_huber_quality": signed_scale_huber * huber_quality,
            "signed_geomean_r": signed_scale * geomean_r,
            "signed_geomean_r_huber_quality": signed_scale_huber * geomean_r * huber_quality,
            "dower_huber_quality": dower * huber_quality,
            "power_doppler": dower.abs(),
            "color_doppler": velocity,
        }
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", type=int, default=200)
    p.add_argument("--stop", type=int, default=399)
    p.add_argument("--y-indices", type=int, nargs="+", default=list(range(11, 19)))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    y_indices = np.asarray(args.y_indices, dtype=np.int32)
    sums = None
    count = 0
    with h5py.File(args.h5, "r") as h5:
        frame_rate, tx_freq = metadata(h5, args.start)
        grid = h5[f"acquisitions/{args.start}/meta/grid"]
        x_mm = np.unique(grid["x"][:]).astype(np.float32) * 1000
        y_mm = np.unique(grid["y"][:]).astype(np.float32)[y_indices] * 1000
        z_mm = np.unique(grid["z"][:]).astype(np.float32) * 1000
        for acq in range(args.start, args.stop + 1):
            stack = h5[f"acquisitions/{acq}/meta/compound_image"][:, y_indices, :, :].astype(np.complex64)
            metrics = all_metrics(stack, frame_rate, tx_freq, device)
            if sums is None:
                sums = {k: np.zeros_like(v, dtype=np.float64) for k, v in metrics.items()}
            for k, v in metrics.items():
                sums[k] += v
            count += 1
            print(f"acq {acq}: {count}/{args.stop - args.start + 1}", flush=True)
    arrays = {k: (v / count).astype(np.float32) for k, v in sums.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        **arrays,
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
        y_indices=y_indices,
        frame_rate_hz=np.float32(frame_rate),
        tx_freq_hz=np.float32(tx_freq),
        sound_speed=np.float32(SOUND_SPEED),
        low_cutoff=np.float32(LOW_CUTOFF),
        max_lag=np.int32(MAX_LAG),
        huber_delta=np.float32(HUBER_DELTA),
        huber_iterations=np.int32(HUBER_ITERATIONS),
        first_acq=np.int32(args.start),
        last_acq=np.int32(args.stop),
        source_h5=str(args.h5),
        note="Recomputed from cached beamformed compound_image in ultratrace H5, middle 8 elevation planes.",
    )


if __name__ == "__main__":
    main()
