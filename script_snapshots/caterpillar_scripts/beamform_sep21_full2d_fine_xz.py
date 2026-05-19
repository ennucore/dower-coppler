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
from caterpillar.acquire.tx_delay_backfill import (
    _METADATA_ONLY_SKIP_KEYS,
    compute_full_2d_tx_delays,
)
from caterpillar.imaging.beamform import BeamformingParameters
from caterpillar.imaging.doppler import (
    _lag_autocorrelations,
    _unwrap_lag_phases,
    lag_phase_linear_fit,
    signed_tmas_wls_huber,
    svd_filter_fast,
)
from caterpillar.imaging.grid import GridParams
from caterpillar.imaging.mach_beamform import MachBeamformer, mach_beamform
from caterpillar.utils.io.hdf5 import read_meta_group


DEFAULT_H5 = Path(
    "/mnt/pocampus/lev/ultratrace_Head_monster_2025-09-21_21-32-01_y-20to20mm_30elev.h5"
)
DEFAULT_OUT = Path(
    "/home/monster/caterpillar/results/doppler_cnr_gui/"
    "head_2025-09-21_full2dtx_fine_xz_yidx14_15_acq200_399.npz"
)
LOW_CUTOFF = 0.08
MAX_LAG = 5
SOUND_SPEED = 1600.0
HUBER_DELTA = 0.7
HUBER_ITERATIONS = 5
DOPPLER_DEVICE = torch.device("cpu")
SKIP_META_KEYS = set(_METADATA_ONLY_SKIP_KEYS) - {"iq_frames"}


def _metadata(h5: h5py.File, start: int) -> tuple[float, float]:
    meta = h5[f"acquisitions/{start}/meta"]
    cfg_raw = meta["acquisition_config"][()]
    rt_raw = meta["runtime_metadata"][()]
    if isinstance(cfg_raw, bytes):
        cfg_raw = cfg_raw.decode()
    if isinstance(rt_raw, bytes):
        rt_raw = rt_raw.decode()
    cfg = json.loads(cfg_raw)
    rt = json.loads(rt_raw)
    frame_rate = float(
        rt.get("empirical_pulse_repetition_rate_hz")
        or (cfg["requested_prf_hz"] / cfg["num_angles"])
    )
    return frame_rate, float(cfg["tx_freq_hz"])


def _load_acq_with_full_tx(
    h5: h5py.File,
    acq_index: int,
    delay_cache: dict[int, object],
) -> Acquisition:
    meta = h5["acquisitions"][str(acq_index)]["meta"]
    config, runtime_metadata, kwargs = read_meta_group(meta, skip_keys=SKIP_META_KEYS)
    key = hash(config)
    if key not in delay_cache:
        delay_cache[key] = compute_full_2d_tx_delays(config, mock=True)
    full = delay_cache[key]
    runtime_metadata.tx_delays = [d.copy() for d in full.tx_delays]
    runtime_metadata.tx_delays_elev = [d.copy() for d in full.tx_delays_elev]

    acq = Acquisition(config=config, runtime_metadata=runtime_metadata, **kwargs)
    if acq.iq_frames is None:
        acq.unbox()
    if acq.iq_frames is None:
        raise RuntimeError(f"acq {acq_index} has no IQ frames")
    return acq


def _wls_origin_phase_slope(
    phases: torch.Tensor,
    weights: torch.Tensor,
    lag_values: torch.Tensor,
) -> torch.Tensor:
    numerator = (weights * lag_values * phases).sum(dim=0)
    denominator = (weights * lag_values.square()).sum(dim=0).clamp_min(1e-12)
    return numerator / denominator


def _huber_lag_maps(sig: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lags, rk = _lag_autocorrelations(sig, max_lag=MAX_LAG)
    if rk is None:
        z = torch.zeros_like(sig[0].real)
        return z, z, z
    phases = _unwrap_lag_phases(torch.angle(rk))
    weights = rk.abs()
    lag_values = torch.as_tensor(lags, device=sig.device, dtype=phases.dtype)
    lag_values = lag_values.view(-1, *([1] * (sig.ndim - 1)))
    slope = _wls_origin_phase_slope(phases, weights, lag_values)
    delta = max(float(HUBER_DELTA), 1e-6)
    for _ in range(max(0, int(HUBER_ITERATIONS))):
        residual = phases - lag_values * slope
        robust = torch.clamp(delta / residual.abs().clamp_min(1e-6), max=1.0)
        slope = _wls_origin_phase_slope(phases, weights * robust, lag_values)
    residual = phases - lag_values * slope
    robust = torch.clamp(delta / residual.abs().clamp_min(1e-6), max=1.0)
    weight_sum = weights.sum(dim=0)
    quality = (weights * robust).sum(dim=0) / weight_sum.clamp_min(1e-12)
    quality = torch.where(weight_sum > 1e-12, quality, torch.zeros_like(quality))
    geomean_r = torch.exp(torch.log(weights.clamp_min(1e-30)).mean(dim=0))
    signed_scale = (slope / np.pi).clamp(-1.0, 1.0)
    return signed_scale, quality.clamp(0.0, 1.0), geomean_r


def _acq_metrics(
    compound_np: np.ndarray,
    frame_rate: float,
    tx_freq: float,
) -> tuple[np.ndarray, ...]:
    with torch.no_grad():
        sig = torch.from_numpy(compound_np.astype(np.complex64, copy=False)).to(
            DOPPLER_DEVICE
        )
        sig = sig - sig.mean(dim=0, keepdim=True)
        filt = svd_filter_fast(sig, low_cutoff=LOW_CUTOFF, high_cutoff=1.0)
        dower = signed_tmas_wls_huber(filt, max_lag=MAX_LAG)
        slope, r2 = lag_phase_linear_fit(
            filt,
            max_lag=MAX_LAG,
            weighted=True,
            fit_intercept=False,
        )
        _, rk1 = _lag_autocorrelations(filt, max_lag=1)
        color = torch.angle(rk1[0]) * (float(frame_rate) / (2.0 * np.pi)) * (
            SOUND_SPEED / (2.0 * float(tx_freq))
        )
        signed_scale, huber_quality, geomean_r = _huber_lag_maps(filt)
        velocity = slope * (float(frame_rate) / (2.0 * np.pi)) * (
            SOUND_SPEED / (2.0 * float(tx_freq))
        )
        return tuple(
            x.detach().cpu().numpy().astype(np.float32)
            for x in (
                dower,
                velocity,
                velocity * r2,
                r2,
                signed_scale,
                huber_quality,
                geomean_r,
                signed_scale * huber_quality,
                signed_scale * geomean_r,
                signed_scale * geomean_r * huber_quality,
                dower * huber_quality,
                color,
            )
        )


def _make_grid_params(
    h5: h5py.File,
    start: int,
    y_indices: np.ndarray,
    y_range_mm: tuple[float, float, int] | None,
) -> tuple[GridParams, np.ndarray, np.ndarray, np.ndarray]:
    grid = h5[f"acquisitions/{start}/meta/grid"]
    x = np.unique(np.asarray(grid["x"]))
    y = np.unique(np.asarray(grid["y"]))
    z = np.unique(np.asarray(grid["z"]))
    if y_range_mm is None:
        y_sel = y[y_indices]
        y_range = (float(y_sel[0]), float(y_sel[-1]), int(len(y_sel)))
    else:
        y0_mm, y1_mm, count = y_range_mm
        y_range = (float(y0_mm) / 1000.0, float(y1_mm) / 1000.0, int(count))
    params = GridParams(
        x_range=(float(x.min()), float(x.max()), int(len(x) * 2)),
        y_range=y_range,
        z_range=(float(z.min()), float(z.max()), int(len(z) * 2)),
    )
    return params, x, y, z


def _write_output(
    out: Path,
    arrays: dict[str, np.ndarray],
    *,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    y_indices: np.ndarray,
    y_range_mm: tuple[float, float, int] | None,
    frame_rate: float,
    tx_freq: float,
    h5: Path,
    start: int,
    stop: int,
    original_grid_yzx: tuple[int, int, int],
    fine_grid_yzx: tuple[int, int, int],
    count: int,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial.npz")
    np.savez_compressed(
        tmp,
        **arrays,
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
        y_indices=y_indices.astype(np.int32),
        y_range_mm=(
            np.asarray(y_range_mm, dtype=np.float32)
            if y_range_mm is not None
            else np.asarray([], dtype=np.float32)
        ),
        frame_rate_hz=np.float32(frame_rate),
        tx_freq_hz=np.float32(tx_freq),
        sound_speed=np.float32(SOUND_SPEED),
        low_cutoff=np.float32(LOW_CUTOFF),
        max_lag=np.int32(MAX_LAG),
        huber_delta=np.float32(HUBER_DELTA),
        huber_iterations=np.int32(HUBER_ITERATIONS),
        first_acq=np.int32(start),
        last_acq=np.int32(stop),
        acq_count=np.int32(count),
        source_h5=str(h5),
        original_grid_yzx=np.array(original_grid_yzx, dtype=np.int32),
        fine_grid_yzx=np.array(fine_grid_yzx, dtype=np.int32),
        tx_delay_mode=np.asarray("recovered_full_2d_mock_poseidon"),
        note=np.asarray(
            "Fine x/z Sep 2025 rebeamform using recovered full 2D TX delays; "
            "original y indices 14 and 15 unless overridden."
        ),
    )
    tmp.replace(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--stop", type=int, default=399)
    parser.add_argument("--y-indices", type=int, nargs="+", default=[14, 15])
    parser.add_argument(
        "--y-range-mm",
        type=float,
        nargs=3,
        metavar=("START", "STOP", "COUNT"),
        default=None,
        help="Override y-indices with a custom elevation range in millimeters.",
    )
    parser.add_argument("--n-chunks", type=int, default=24)
    parser.add_argument("--max-acqs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"{args.out} exists; choose a new --out or pass --overwrite")
    if args.out.with_suffix(".partial.npz").exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.out.with_suffix('.partial.npz')} exists; remove it or pass --overwrite"
        )

    t_start = time.time()
    beamformer = MachBeamformer()
    y_range_mm = (
        (float(args.y_range_mm[0]), float(args.y_range_mm[1]), int(args.y_range_mm[2]))
        if args.y_range_mm is not None
        else None
    )
    y_indices = (
        np.arange(y_range_mm[2], dtype=np.int32)
        if y_range_mm is not None
        else np.asarray(args.y_indices, dtype=np.int32)
    )
    delay_cache: dict[int, object] = {}

    names = [
        "dower_coppler",
        "phase_velocity",
        "phase_velocity_r2",
        "phase_r2",
        "signed_scale",
        "huber_quality",
        "geomean_r",
        "signed_scale_huber_quality",
        "signed_geomean_r",
        "signed_geomean_r_huber_quality",
        "dower_huber_quality",
        "color_doppler",
    ]

    with h5py.File(args.h5, "r") as h5:
        requested_indices = list(range(args.start, args.stop + 1))
        if args.max_acqs is not None:
            requested_indices = requested_indices[: args.max_acqs]
        frame_rate, tx_freq = _metadata(h5, args.start)
        grid_params, x0, y0, z0 = _make_grid_params(
            h5,
            args.start,
            y_indices,
            y_range_mm,
        )
        x_mm = (
            np.linspace(grid_params.x_range[0], grid_params.x_range[1], grid_params.x_range[2])
            .astype(np.float32)
            * 1000
        )
        y_mm = (
            np.linspace(grid_params.y_range[0], grid_params.y_range[1], grid_params.y_range[2])
            .astype(np.float32)
            * 1000
        )
        z_mm = (
            np.linspace(grid_params.z_range[0], grid_params.z_range[1], grid_params.z_range[2])
            .astype(np.float32)
            * 1000
        )
        shape = (len(y_mm), len(z_mm), len(x_mm))
        print(
            json.dumps(
                {
                    "source": str(args.h5),
                    "out": str(args.out),
                    "acq_range": [args.start, args.stop],
                    "requested_count": len(requested_indices),
                    "orig_grid_yzx": [int(len(y0)), int(len(z0)), int(len(x0))],
                    "fine_grid_yzx": list(shape),
                    "y_indices": y_indices.tolist(),
                    "y_range_mm": list(y_range_mm) if y_range_mm is not None else None,
                    "y_mm": y_mm.tolist(),
                    "frame_rate_hz": frame_rate,
                    "tx_freq_hz": tx_freq,
                    "n_chunks": args.n_chunks,
                },
                indent=2,
            ),
            flush=True,
        )

        sums = None
        count = 0
        for acq_idx in requested_indices:
            t0 = time.time()
            acq = _load_acq_with_full_tx(h5, acq_idx, delay_cache)
            params = BeamformingParameters.create(
                acq.config,
                row_index=-1,
                mean_subtract_channels=True,
                n_chunks=args.n_chunks,
            )
            params.grid_params = grid_params
            params.interpolation_mode = "linear"
            compound, _angular, grid = mach_beamform(
                acq.iq_frames,
                acq.config,
                acq.runtime_metadata,
                params,
                beamformer=beamformer,
                compound_right_away=True,
            )
            compound = np.asarray(compound, dtype=np.complex64)
            if compound.shape[1:] != shape:
                raise RuntimeError(
                    f"unexpected compound shape {compound.shape}; expected frames,{shape}"
                )
            metrics = _acq_metrics(compound, frame_rate, tx_freq)
            if sums is None:
                sums = [np.zeros_like(m, dtype=np.float64) for m in metrics]
            for s, m in zip(sums, metrics):
                s += m
            count += 1
            del acq, compound, _angular, grid, metrics
            gc.collect()
            print(
                f"acq {acq_idx}: processed {count}/{len(requested_indices)} "
                f"in {time.time() - t0:.1f}s; elapsed={(time.time() - t_start) / 60:.1f}min",
                flush=True,
            )

    if sums is None or count == 0:
        raise RuntimeError("no acquisitions processed")
    arrays = {name: (s / count).astype(np.float32) for name, s in zip(names, sums)}
    arrays["power_doppler"] = np.abs(arrays["dower_coppler"]).astype(np.float32)
    _write_output(
        args.out,
        arrays,
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
        y_indices=y_indices,
        y_range_mm=y_range_mm,
        frame_rate=frame_rate,
        tx_freq=tx_freq,
        h5=args.h5,
        start=args.start,
        stop=requested_indices[-1],
        original_grid_yzx=(len(y0), len(z0), len(x0)),
        fine_grid_yzx=shape,
        count=count,
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "count": count,
                "shape": list(arrays["dower_coppler"].shape),
                "elapsed_min": (time.time() - t_start) / 60,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
