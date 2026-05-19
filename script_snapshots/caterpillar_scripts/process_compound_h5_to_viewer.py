#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from caterpillar.acquire.acquisition import Acquisition
from caterpillar.imaging.doppler import power_doppler
from caterpillar.utils.io.hdf5 import read_meta_group

LOW_CUTOFF = 0.08
MAX_LAG = 5
SKIP_FIRST_FRAMES = 5
HUBER_DELTA = 0.7
HUBER_ITERATIONS = 5


def _slow_time_frame_rate(config, runtime) -> tuple[float, float, int]:
    pulse_prf = float(getattr(runtime, "empirical_pulse_repetition_rate_hz", 0.0) or config.requested_prf_hz)
    num_angles = max(1, int(getattr(config, "total_num_angles", config.num_angles) or 1))
    return pulse_prf / float(num_angles), pulse_prf, num_angles


def _lag_autocorrelations(sig: torch.Tensor, max_lag: int) -> tuple[list[int], torch.Tensor]:
    lags = list(range(1, min(max_lag, sig.shape[0] - 1) + 1))
    return lags, torch.stack([(sig[k:] * torch.conj(sig[:-k])).mean(dim=0) for k in lags], dim=0)


def _unwrap_lag_phases(phases: torch.Tensor) -> torch.Tensor:
    if phases.shape[0] <= 1:
        return phases
    diffs = phases[1:] - phases[:-1]
    wrapped = torch.remainder(diffs + np.pi, 2.0 * np.pi) - np.pi
    return torch.cat((phases[:1], phases[1:] + torch.cumsum(wrapped - diffs, dim=0)), dim=0)


def _wls_origin(phases: torch.Tensor, weights: torch.Tensor, lags: list[int]) -> torch.Tensor:
    lag_values = torch.as_tensor(lags, device=phases.device, dtype=phases.dtype)
    lag_values = lag_values.view(-1, *([1] * (phases.ndim - 1)))
    return (weights * lag_values * phases).sum(dim=0) / (weights * lag_values.square()).sum(dim=0).clamp_min(1e-12)


def _phase_fit(sig: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lags, rk = _lag_autocorrelations(sig, MAX_LAG)
    phases = _unwrap_lag_phases(torch.angle(rk))
    weights = rk.abs()
    slope = _wls_origin(phases, weights, lags)
    lag_values = torch.as_tensor(lags, device=sig.device, dtype=phases.dtype)
    lag_values = lag_values.view(-1, *([1] * (sig.ndim - 1)))
    weight_sum_raw = weights.sum(dim=0)
    weight_sum = weight_sum_raw.clamp_min(1e-12)
    predicted = lag_values * slope
    phase_mean = (weights * phases).sum(dim=0) / weight_sum
    r2 = (1.0 - (weights * (phases - predicted).square()).sum(dim=0) / (weights * (phases - phase_mean).square()).sum(dim=0).clamp_min(1e-12)).clamp(0.0, 1.0)
    r2 = torch.where(weight_sum_raw > 1e-12, r2, torch.zeros_like(r2))
    huber_slope = slope
    robust = torch.ones_like(weights)
    for _ in range(HUBER_ITERATIONS):
        residual = phases - lag_values * huber_slope
        robust = torch.clamp(HUBER_DELTA / residual.abs().clamp_min(1e-6), max=1.0)
        huber_slope = _wls_origin(phases, weights * robust, lags)
    huber_quality = ((weights * robust).sum(dim=0) / weight_sum).clamp(0.0, 1.0)
    geomean_r = torch.exp(torch.log(weights.clamp_min(1e-30)).mean(dim=0))
    return slope, r2, huber_slope, huber_quality, geomean_r


def _grid_axes_mm(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.unique(np.asarray(grid["x"], dtype=np.float64)) * 1000.0
    y = np.unique(np.asarray(grid.get("y", np.array([0.0])), dtype=np.float64)) * 1000.0
    z = np.unique(np.asarray(grid["z"], dtype=np.float64)) * 1000.0
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def _middle_raw_iq_image(h5_path: Path) -> tuple[np.ndarray | None, int]:
    with h5py.File(h5_path, "r") as h5:
        if "acquisitions" in h5:
            ids = sorted(int(k) for k in h5["acquisitions"].keys())
            acq_index = ids[len(ids) // 2]
            config, runtime, kwargs = read_meta_group(h5["acquisitions"][str(acq_index)]["meta"])
            acq = Acquisition(config=config, runtime_metadata=runtime, **kwargs)
            try:
                if acq.iq_frames is None:
                    acq.unbox()
            except ValueError:
                return None, int(acq_index)
            iq = np.asarray(acq.iq_frames)
        else:
            acq_index = 0
            if "iq_frames" not in h5:
                return None, int(acq_index)
            iq = np.asarray(h5["iq_frames"])
    mag = np.median(np.abs(iq), axis=(0, 1))
    if mag.ndim == 3:
        mag = mag.reshape((mag.shape[0] * mag.shape[1], mag.shape[2]))
    else:
        mag = mag.reshape((-1, mag.shape[-1]))
    return mag.astype(np.float32), int(acq_index)


def _process_acq(h5_path: Path, acq_index: int, device: torch.device):
    with h5py.File(h5_path, "r") as h5:
        if "acquisitions" in h5:
            meta = h5["acquisitions"][str(acq_index)]["meta"]
        else:
            meta = h5["meta"]
        config, runtime, kwargs = read_meta_group(meta, skip_keys={"iq_frames", "raw_frames"})
        compound = np.asarray(meta["compound_image"], dtype=np.complex64)
        grid = {axis: np.asarray(meta["grid"][axis]) for axis in meta["grid"].keys()}
    if compound.ndim == 4 and compound.shape[1] == 1:
        compound = compound[:, 0]
    sig = torch.from_numpy(compound).to(device)
    frame_rate, pulse_prf, num_angles = _slow_time_frame_rate(config, runtime)
    pd, filt = power_doppler(sig, low_cutoff=LOW_CUTOFF, high_cutoff=1.0, mean_subtract=True, skip_first_frames=SKIP_FIRST_FRAMES, method="fast", separate_3d_svd=False)
    fit_sig = filt[SKIP_FIRST_FRAMES:]
    _, rk = _lag_autocorrelations(fit_sig, MAX_LAG)
    r1 = rk[0]
    color = torch.angle(r1) * frame_rate / (2.0 * np.pi)
    color = color * config.speed_of_sound / (2.0 * config.tx_freq_hz)
    slope, r2, huber_slope, huber_quality, geomean_r = _phase_fit(fit_sig)
    phase_velocity = slope * frame_rate / (2.0 * np.pi)
    phase_velocity = phase_velocity * config.speed_of_sound / (2.0 * config.tx_freq_hz)
    signed_scale = (slope / np.pi).clamp(-1.0, 1.0)
    signed_scale_huber = (huber_slope / np.pi).clamp(-1.0, 1.0)
    phase_velocity_geomean_r = phase_velocity * geomean_r
    phase_velocity_r2 = phase_velocity * r2
    dower = phase_velocity_geomean_r * r2
    out = {
        "power_doppler": pd,
        "color_doppler": color,
        "dower_coppler": dower,
        "phase_velocity": phase_velocity,
        "v_phi": phase_velocity,
        "phase_velocity_geomean_r": phase_velocity_geomean_r,
        "v_phi_G_R": phase_velocity_geomean_r,
        "phase_velocity_r2": phase_velocity_r2,
        "v_phi_R2": phase_velocity_r2,
        "phase_r2": r2,
        "R2": r2,
        "signed_scale": signed_scale,
        "huber_quality": huber_quality,
        "geomean_r": geomean_r,
        "G_R": geomean_r,
        "v_phi_G_R_R2": dower,
        "signed_scale_huber_quality": signed_scale_huber * huber_quality,
        "signed_geomean_r": signed_scale * geomean_r,
        "signed_geomean_r_huber_quality": signed_scale_huber * geomean_r * huber_quality,
        "dower_huber_quality": dower * huber_quality,
    }
    out_np = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in out.items()}
    axes = _grid_axes_mm(grid)
    meta_out = {
        "frame_rate_hz": float(frame_rate),
        "compound_frame_rate_hz": float(frame_rate),
        "pulse_repetition_rate_hz": float(pulse_prf),
        "num_compound_angles": int(num_angles),
        "velocity_scale_corrected": True,
        "tx_freq_hz": float(config.tx_freq_hz),
        "sound_speed": float(config.speed_of_sound),
        "grid_shape": list(compound.shape[1:]),
    }
    del sig, filt, fit_sig
    torch.cuda.empty_cache()
    gc.collect()
    return out_np, axes, meta_out


def _save(out_path: Path, metrics: list[dict[str, np.ndarray]], axes, raw_iq, raw_iq_acq, source, indices, meta, label):
    out = {key: np.median(np.stack([m[key] for m in metrics], axis=0), axis=0)[None].astype(np.float32) for key in metrics[0]}
    x_mm, y_mm, z_mm = axes
    raw_fields = {}
    if raw_iq is not None:
        raw_fields = {
            "raw_iq_magnitude": raw_iq[None].astype(np.float32),
            "raw_iq_acq_index": np.asarray(raw_iq_acq, dtype=np.int32),
            "raw_iq_source_h5": np.asarray(str(source)),
        }
    tmp = out_path.with_suffix(".partial.npz")
    np.savez_compressed(
        tmp,
        **out,
        **raw_fields,
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
        frame_rate_hz=np.asarray(meta["frame_rate_hz"], dtype=np.float32),
        tx_freq_hz=np.asarray(meta["tx_freq_hz"], dtype=np.float32),
        sound_speed=np.asarray(meta["sound_speed"], dtype=np.float32),
        low_cutoff=np.asarray(LOW_CUTOFF, dtype=np.float32),
        max_lag=np.asarray(MAX_LAG, dtype=np.int32),
        first_acq=np.asarray(indices[0], dtype=np.int32),
        last_acq=np.asarray(indices[-1], dtype=np.int32),
        source_h5=np.asarray(str(source)),
        note=np.asarray(f"Processed from existing beamformed compound_image; raw IQ from middle acq; {label}."),
    )
    tmp.replace(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/doppler_cnr_gui"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    with h5py.File(args.h5, "r") as h5:
        indices = sorted(int(k) for k in h5["acquisitions"].keys()) if "acquisitions" in h5 else [0]
    raw_iq, raw_iq_acq = _middle_raw_iq_image(args.h5)
    per_acq = []
    axes = meta = None
    started = time.time()
    for pos, idx in enumerate(indices, start=1):
        t0 = time.time()
        m, axes, meta = _process_acq(args.h5, idx, device)
        per_acq.append(m)
        print(f"{args.h5.name} acq {idx} ({pos}/{len(indices)}) {time.time()-t0:.1f}s elapsed={(time.time()-started)/60:.1f}min", flush=True)
    half = len(indices) // 2
    windows = [(f"all{len(indices)}", per_acq, indices), (f"first{half}", per_acq[:half], indices[:half]), (f"last{len(indices)-half}", per_acq[half:], indices[half:])]
    written = []
    for label, metrics, idxs in windows:
        if not metrics:
            continue
        out_path = args.out_dir / f"{args.prefix}_{label}.npz"
        _save(out_path, metrics, axes, raw_iq, raw_iq_acq, args.h5, idxs, meta, label)
        print(f"wrote {out_path}", flush=True)
        written.append(str(out_path))
    (args.out_dir / f"{args.prefix}_metadata.json").write_text(json.dumps({"source_h5": str(args.h5), "outputs": written, "meta": meta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
