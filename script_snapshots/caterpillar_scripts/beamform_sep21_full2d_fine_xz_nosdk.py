from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

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
LOW_CUTOFF = 0.08
MAX_LAG = 5
SOUND_SPEED = 1600.0
HUBER_DELTA = 0.7
HUBER_ITERATIONS = 5
DOPPLER_DEVICE = torch.device("cpu")
SKIP_META_KEYS = {
    "raw_frames",
    "compound_image",
    "angular_images",
    "doppler_img",
    "doppler_signal",
    "sideways_compound_image",
    "color_doppler_frames",
}


def _metadata(h5: h5py.File, start: int) -> tuple[float, float, float, int]:
    meta = h5[f"acquisitions/{start}/meta"]
    cfg_raw = meta["acquisition_config"][()]
    rt_raw = meta["runtime_metadata"][()]
    if isinstance(cfg_raw, bytes):
        cfg_raw = cfg_raw.decode()
    if isinstance(rt_raw, bytes):
        rt_raw = rt_raw.decode()
    cfg = json.loads(cfg_raw)
    rt = json.loads(rt_raw)
    num_angles = max(1, int(cfg.get("num_angles", 1)))
    pulse_prf = float(
        rt.get("empirical_pulse_repetition_rate_hz")
        or cfg["requested_prf_hz"]
    )
    # Slow-time samples are compounded frames, not individual transmit pulses.
    frame_rate = pulse_prf / float(num_angles)
    return frame_rate, float(cfg["tx_freq_hz"]), pulse_prf, num_angles


def _load_delays(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    with np.load(path) as z:
        tx_delays = np.asarray(z["tx_delays"], dtype=np.float64)
        tx_delays_elev = np.asarray(z["tx_delays_elev"], dtype=np.float64)
    if tx_delays.ndim != 3:
        raise ValueError(f"tx_delays must be (A,C,R_tx), got {tx_delays.shape}")
    if tx_delays_elev.ndim != 2:
        raise ValueError(f"tx_delays_elev must be (A,R_rx), got {tx_delays_elev.shape}")
    if tx_delays.shape[0] != tx_delays_elev.shape[0]:
        raise ValueError("tx_delays and tx_delays_elev angle counts differ")
    return [d.copy() for d in tx_delays], [d.copy() for d in tx_delays_elev]


def _load_acq(
    h5: h5py.File,
    acq_index: int,
    tx_delays: list[np.ndarray] | None,
    tx_delays_elev: list[np.ndarray] | None,
):
    meta = h5["acquisitions"][str(acq_index)]["meta"]
    config, runtime_metadata, kwargs = read_meta_group(meta, skip_keys=SKIP_META_KEYS)
    if tx_delays is not None and tx_delays_elev is not None:
        runtime_metadata.tx_delays = [d.copy() for d in tx_delays]
        runtime_metadata.tx_delays_elev = [d.copy() for d in tx_delays_elev]
    iq = kwargs.get("iq_frames")
    if iq is None:
        if "iq_frames" not in meta:
            raise RuntimeError(f"acq {acq_index} has no iq_frames dataset")
        iq = meta["iq_frames"][()]
    return config, runtime_metadata, np.asarray(iq)


def _wls_origin_phase_slope(
    phases: torch.Tensor,
    weights: torch.Tensor,
    lag_values: torch.Tensor,
) -> torch.Tensor:
    return (weights * lag_values * phases).sum(dim=0) / (
        weights * lag_values.square()
    ).sum(dim=0).clamp_min(1e-12)


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
    r1_to_r5_product = weights.prod(dim=0)
    signed_scale = (slope / np.pi).clamp(-1.0, 1.0)
    return signed_scale, quality.clamp(0.0, 1.0), geomean_r, r1_to_r5_product


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
        power = filt[5:].abs().square().sum(dim=0)
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
        signed_scale, huber_quality, geomean_r, r1_to_r5_product = _huber_lag_maps(filt)
        velocity = slope * (float(frame_rate) / (2.0 * np.pi)) * (
            SOUND_SPEED / (2.0 * float(tx_freq))
        )
        return tuple(
            x.detach().cpu().numpy().astype(np.float32)
            for x in (
                dower,
                power,
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
                velocity * r1_to_r5_product * r2,
                color,
            )
        )


def _load_per_acq_metrics(
    path: Path,
    names: list[str],
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, ...]:
    with np.load(path) as z:
        metrics = tuple(np.asarray(z[name], dtype=np.float32) for name in names)
    for name, metric in zip(names, metrics):
        if metric.shape != shape:
            raise ValueError(f"{path} {name} has shape {metric.shape}, expected {shape}")
    return metrics


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument(
        "--tx-delays-npz",
        type=Path,
        default=None,
        help="Recovered full 2D delay sidecar. Omit to use saved legacy/projection TX metadata.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--stop", type=int, default=399)
    parser.add_argument("--y-indices", type=int, nargs="+", default=[14, 15])
    parser.add_argument("--y-range-mm", type=float, nargs=3, default=None)
    parser.add_argument("--n-chunks", type=int, default=24)
    parser.add_argument("--max-acqs", type=int, default=None)
    parser.add_argument("--tx-delay-downsample-rows", type=int, default=8)
    parser.add_argument(
        "--tx-delay-downsample-method",
        choices=["wavefront_fit", "centered", "mean"],
        default="wavefront_fit",
    )
    parser.add_argument("--tx-delay-downsample-fit-points", type=int, default=512)
    parser.add_argument(
        "--save-per-acq",
        action="store_true",
        help=(
            "Write standard display arrays for each acquisition into a sidecar "
            "directory while keeping the main output as the mean image."
        ),
    )
    parser.add_argument(
        "--resume-per-acq",
        action="store_true",
        help="When --save-per-acq is enabled, reuse existing valid sidecar files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume_per_acq and not args.save_per_acq:
        raise ValueError("--resume-per-acq requires --save-per-acq")

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"{args.out} exists; choose a new --out or pass --overwrite")
    if args.out.with_suffix(".partial.npz").exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.out.with_suffix('.partial.npz')} exists; remove it or pass --overwrite"
        )
    per_acq_dir = args.out.parent / f"{args.out.stem}_per_acq"
    if (
        args.save_per_acq
        and per_acq_dir.exists()
        and not args.overwrite
        and not args.resume_per_acq
    ):
        raise FileExistsError(
            f"{per_acq_dir} exists; choose a new --out, remove it, or pass --overwrite"
        )

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
    if args.tx_delays_npz is None:
        tx_delays = tx_delays_elev = None
        tx_delay_mode = "saved_legacy_projection"
    else:
        tx_delays, tx_delays_elev = _load_delays(args.tx_delays_npz)
        tx_delay_mode = "recovered_full_2d_sidecar"
    tx_delay_downsample_rows = (
        None if args.tx_delay_downsample_rows <= 0 else int(args.tx_delay_downsample_rows)
    )
    if tx_delays is not None and tx_delay_downsample_rows is not None:
        tx_delay_mode = (
            f"{tx_delay_mode}_fast{tx_delay_downsample_rows}"
            f"_{args.tx_delay_downsample_method}"
        )
    beamformer = MachBeamformer()
    t_start = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_per_acq:
        per_acq_dir.mkdir(parents=True, exist_ok=True)

    names = [
        "dower_coppler",
        "power_doppler",
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
        "phase_velocity_r1_to_r5_r2_fit_quality",
        "color_doppler",
    ]

    with h5py.File(args.h5, "r") as h5:
        requested_indices = list(range(args.start, args.stop + 1))
        if args.max_acqs is not None:
            requested_indices = requested_indices[: args.max_acqs]
        frame_rate, tx_freq, pulse_prf, num_angles = _metadata(h5, args.start)
        grid_params, x0, y0, z0 = _make_grid_params(h5, args.start, y_indices, y_range_mm)
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
                    "tx_delays_npz": str(args.tx_delays_npz) if args.tx_delays_npz else None,
                    "tx_delay_mode": tx_delay_mode,
                    "out": str(args.out),
                    "save_per_acq": bool(args.save_per_acq),
                    "resume_per_acq": bool(args.resume_per_acq),
                    "per_acq_dir": str(per_acq_dir) if args.save_per_acq else None,
                    "requested_count": len(requested_indices),
                    "orig_grid_yzx": [int(len(y0)), int(len(z0)), int(len(x0))],
                    "fine_grid_yzx": list(shape),
                    "y_indices": y_indices.tolist(),
                    "y_range_mm": list(y_range_mm) if y_range_mm is not None else None,
                    "y_mm": y_mm.tolist(),
                    "n_chunks": args.n_chunks,
                    "tx_delay_downsample_rows": tx_delay_downsample_rows,
                    "tx_delay_downsample_method": args.tx_delay_downsample_method,
                    "tx_delay_downsample_fit_points": args.tx_delay_downsample_fit_points,
                },
                indent=2,
            ),
            flush=True,
        )

        sums = None
        count = 0
        processed_indices = []
        for acq_idx in requested_indices:
            t0 = time.time()
            per_acq_path = per_acq_dir / f"acq_{acq_idx:03d}.npz"
            if args.resume_per_acq and per_acq_path.exists():
                try:
                    metrics = _load_per_acq_metrics(per_acq_path, names, shape)
                except Exception as exc:
                    print(
                        f"acq {acq_idx}: could not reuse {per_acq_path}: {exc}; recomputing",
                        flush=True,
                    )
                else:
                    if sums is None:
                        sums = [np.zeros_like(m, dtype=np.float64) for m in metrics]
                    for s, m in zip(sums, metrics):
                        s += m
                    count += 1
                    processed_indices.append(acq_idx)
                    print(
                        f"acq {acq_idx}: reused sidecar {count}/{len(requested_indices)} "
                        f"in {time.time() - t0:.1f}s; elapsed={(time.time() - t_start) / 60:.1f}min",
                        flush=True,
                    )
                    continue
            config, runtime, iq = _load_acq(h5, acq_idx, tx_delays, tx_delays_elev)
            params = BeamformingParameters.create(
                config,
                row_index=-1,
                mean_subtract_channels=True,
                n_chunks=args.n_chunks,
                tx_delay_downsample_rows=tx_delay_downsample_rows,
                tx_delay_downsample_method=args.tx_delay_downsample_method,
                tx_delay_downsample_fit_points=args.tx_delay_downsample_fit_points,
            )
            params.grid_params = grid_params
            params.interpolation_mode = "linear"
            compound, _angular, grid = mach_beamform(
                iq,
                config,
                runtime,
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
            if args.save_per_acq:
                per_acq_arrays = {
                    name: metric.astype(np.float32, copy=False)
                    for name, metric in zip(names, metrics)
                }
                per_acq_tmp = per_acq_path.with_suffix(".partial.npz")
                # Per-acq sidecars are intentionally uncompressed because zip
                # compression is a CPU bottleneck during long beamforming runs.
                np.savez(
                    per_acq_tmp,
                    **per_acq_arrays,
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
                    compound_frame_rate_hz=np.float32(frame_rate),
                    pulse_repetition_rate_hz=np.float32(pulse_prf),
                    num_compound_angles=np.int32(num_angles),
                    velocity_scale_corrected=np.asarray(True),
                    tx_freq_hz=np.float32(tx_freq),
                    sound_speed=np.float32(SOUND_SPEED),
                    low_cutoff=np.float32(LOW_CUTOFF),
                    max_lag=np.int32(MAX_LAG),
                    huber_delta=np.float32(HUBER_DELTA),
                    huber_iterations=np.int32(HUBER_ITERATIONS),
                    first_acq=np.int32(acq_idx),
                    last_acq=np.int32(acq_idx),
                    acq_count=np.int32(1),
                    source_h5=str(args.h5),
                    original_grid_yzx=np.array([len(y0), len(z0), len(x0)], dtype=np.int32),
                    fine_grid_yzx=np.array(shape, dtype=np.int32),
                    tx_delay_mode=np.asarray(tx_delay_mode),
                    tx_delays_npz=str(args.tx_delays_npz) if args.tx_delays_npz else "",
                    tx_delay_downsample_rows=np.asarray(
                        -1 if tx_delay_downsample_rows is None else tx_delay_downsample_rows,
                        dtype=np.int32,
                    ),
                    tx_delay_downsample_method=np.asarray(args.tx_delay_downsample_method),
                    tx_delay_downsample_fit_points=np.asarray(
                        args.tx_delay_downsample_fit_points,
                        dtype=np.int32,
                    ),
                    save_per_acq=np.asarray(True),
                    saved_display_mode=np.asarray("single_acq"),
                    acq_indices=np.asarray([acq_idx], dtype=np.int32),
                    selection_window_acqs=np.asarray(1, dtype=np.int32),
                    note=np.asarray(
                        "Single-acquisition fine x/z Sep 2025 rebeamform using "
                        "recovered full 2D TX delay sidecar with optional TX-row "
                        "wavefront-fit downsampling."
                    ),
                )
                per_acq_tmp.replace(per_acq_path)
            count += 1
            processed_indices.append(acq_idx)
            del config, runtime, iq, compound, _angular, grid, metrics
            gc.collect()
            print(
                f"acq {acq_idx}: processed {count}/{len(requested_indices)} "
                f"in {time.time() - t0:.1f}s; elapsed={(time.time() - t_start) / 60:.1f}min",
                flush=True,
            )

    if sums is None or count == 0:
        raise RuntimeError("no acquisitions processed")
    mean_arrays = {name: (s / count).astype(np.float32) for name, s in zip(names, sums)}
    arrays = dict(mean_arrays)
    saved_display_mode = (
        "mean_across_acqs_with_per_acq_sidecar"
        if args.save_per_acq
        else "mean_across_acqs"
    )

    tmp = args.out.with_suffix(".partial.npz")
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
        compound_frame_rate_hz=np.float32(frame_rate),
        pulse_repetition_rate_hz=np.float32(pulse_prf),
        num_compound_angles=np.int32(num_angles),
        velocity_scale_corrected=np.asarray(True),
        tx_freq_hz=np.float32(tx_freq),
        sound_speed=np.float32(SOUND_SPEED),
        low_cutoff=np.float32(LOW_CUTOFF),
        max_lag=np.int32(MAX_LAG),
        huber_delta=np.float32(HUBER_DELTA),
        huber_iterations=np.int32(HUBER_ITERATIONS),
        first_acq=np.int32(args.start),
        last_acq=np.int32(requested_indices[-1]),
        acq_count=np.int32(count),
        source_h5=str(args.h5),
        original_grid_yzx=np.array([len(y0), len(z0), len(x0)], dtype=np.int32),
        fine_grid_yzx=np.array(shape, dtype=np.int32),
        tx_delay_mode=np.asarray(tx_delay_mode),
        tx_delays_npz=str(args.tx_delays_npz) if args.tx_delays_npz else "",
        tx_delay_downsample_rows=np.asarray(
            -1 if tx_delay_downsample_rows is None else tx_delay_downsample_rows,
            dtype=np.int32,
        ),
        tx_delay_downsample_method=np.asarray(args.tx_delay_downsample_method),
        tx_delay_downsample_fit_points=np.asarray(
            args.tx_delay_downsample_fit_points,
            dtype=np.int32,
        ),
        save_per_acq=np.asarray(bool(args.save_per_acq)),
        saved_display_mode=np.asarray(saved_display_mode),
        per_acq_dir=np.asarray(str(per_acq_dir) if args.save_per_acq else ""),
        acq_indices=np.asarray(processed_indices, dtype=np.int32),
        selection_window_acqs=np.asarray(1, dtype=np.int32),
        note=np.asarray(
            "Fine x/z Sep 2025 rebeamform using recovered full 2D TX delay sidecar "
            "with optional TX-row wavefront-fit downsampling."
        ),
    )
    tmp.replace(args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "count": count,
                "shape": list(arrays["dower_coppler"].shape),
                "saved_display_mode": saved_display_mode,
                "per_acq_dir": str(per_acq_dir) if args.save_per_acq else None,
                "elapsed_min": (time.time() - t_start) / 60,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
