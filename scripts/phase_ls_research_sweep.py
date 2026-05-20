#!/usr/bin/env python3
"""Generate the manuscript simulation table for multi-lag phase estimators.

The simulation in Table 2 is synthetic by design: it stresses individual
failure modes before spending minutes rebeamforming the raw data. This script
is the source of truth for the paper table and writes both machine-readable
metrics and the LaTeX rows included by paper.tex.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHOD_LABELS = {
    "kasai": "Kasai R1",
    "wls_origin": "WLS origin",
    "wls_free": "WLS free intercept",
    "huber_wls": "Huber WLS",
    "leave_one_out": "Leave-one-lag-out",
    "circular_grid": "Circular grid",
    "circular_refined": "Circular + WLS refine",
    "circular_sign_wls": "Circular sign + WLS magnitude",
}

PAPER_METHODS = [
    ("kasai", "Kasai"),
    ("wls_origin", "WLS"),
    ("huber_wls", "Huber-WLS"),
    ("circular_sign_wls", "Circ-WLS"),
]

SCENARIOS = {
    "awgn": "single coherent flow + AWGN",
    "decorrelation": "single flow + phase random walk",
    "lag_outlier": "single flow + corrupted high-lag autocorrelation",
    "clutter_leak": "single flow + residual slow clutter component",
    "two_component": "primary flow + weaker opposing flow",
}

PAPER_SCENARIO_LABELS = {
    "awgn": "AWGN",
    "decorrelation": "Decorrelation",
    "lag_outlier": "Lag outlier",
    "clutter_leak": "Clutter leak",
    "two_component": "Two-component",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research_sweep"))
    parser.add_argument("--n-frames", type=int, default=64)
    parser.add_argument("--n-trials", type=int, default=1024)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[-12, -6, 0, 6, 12])
    parser.add_argument("--omega", type=float, nargs="+", default=[0.35, 0.8, 1.35])
    parser.add_argument(
        "--paper-table",
        type=Path,
        default=Path("outputs/paper_stats/simulation_results_table.tex"),
        help="Path for the generated LaTeX Table 2 block included by paper.tex.",
    )
    return parser.parse_args()


def complex_normal(shape: tuple[int, ...], sigma: float, rng: np.random.Generator) -> np.ndarray:
    return sigma * (
        rng.standard_normal(shape).astype(np.float32)
        + 1j * rng.standard_normal(shape).astype(np.float32)
    )


def simulate_signal(
    *,
    scenario: str,
    omega: float,
    snr_db: float,
    n_frames: int,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.arange(n_frames, dtype=np.float32)[:, None]
    phi0 = (2.0 * np.pi * rng.random((1, n_trials))).astype(np.float32)
    signal = np.exp(1j * (float(omega) * t + phi0)).astype(np.complex64)

    if scenario == "decorrelation":
        walk = (0.18 * rng.standard_normal((n_frames, n_trials))).astype(np.float32)
        signal = np.exp(1j * (float(omega) * t + phi0 + np.cumsum(walk, axis=0))).astype(np.complex64)
    elif scenario == "clutter_leak":
        clutter_phi = (2.0 * np.pi * rng.random((1, n_trials))).astype(np.float32)
        clutter = 0.65 * np.exp(1j * (0.04 * t + clutter_phi)).astype(np.complex64)
        signal = signal + clutter
    elif scenario == "two_component":
        phi2 = (2.0 * np.pi * rng.random((1, n_trials))).astype(np.float32)
        signal = signal + 0.45 * np.exp(1j * (-0.65 * float(omega) * t + phi2)).astype(np.complex64)

    signal_power = float(np.mean(np.abs(signal) ** 2))
    noise_power = signal_power * 10.0 ** (-float(snr_db) / 10.0)
    noise_sigma = float(np.sqrt(noise_power / 2.0))
    return signal + complex_normal(signal.shape, noise_sigma, rng)


def lag_autocorrelations(sig: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    lags = np.arange(1, min(max_lag, sig.shape[0] - 1) + 1, dtype=np.float32)
    rk = []
    for lag in lags.astype(int):
        rk.append(np.mean(sig[lag:] * np.conj(sig[:-lag]), axis=0))
    return lags, np.stack(rk, axis=0).astype(np.complex64)


def corrupt_high_lag(rk: np.ndarray, rng: np.random.Generator, fraction: float = 0.25) -> np.ndarray:
    rk = rk.copy()
    mask = rng.random(rk.shape[1]) < fraction
    phase = 2.0 * np.pi * rng.random(int(mask.sum())) - np.pi
    rk[-1, mask] = 0.25 * np.abs(rk[-1, mask]) * np.exp(1j * phase)
    return rk


def unwrap_lag_phases(phases: np.ndarray) -> np.ndarray:
    if phases.shape[0] <= 1:
        return phases
    diffs = phases[1:] - phases[:-1]
    wrapped = np.remainder(diffs + np.pi, 2.0 * np.pi) - np.pi
    correction = np.cumsum(wrapped - diffs, axis=0)
    return np.concatenate((phases[:1], phases[1:] + correction), axis=0).astype(np.float32)


def weighted_origin_fit(phases: np.ndarray, weights: np.ndarray, lags: np.ndarray) -> np.ndarray:
    x = lags[:, None]
    denom = np.sum(weights * x**2, axis=0)
    return np.sum(weights * x * phases, axis=0) / np.maximum(denom, 1e-12)


def weighted_free_fit(phases: np.ndarray, weights: np.ndarray, lags: np.ndarray) -> np.ndarray:
    x = lags[:, None]
    wsum = np.maximum(np.sum(weights, axis=0), 1e-12)
    xbar = np.sum(weights * x, axis=0) / wsum
    ybar = np.sum(weights * phases, axis=0) / wsum
    xc = x - xbar
    yc = phases - ybar
    denom = np.sum(weights * xc**2, axis=0)
    return np.sum(weights * xc * yc, axis=0) / np.maximum(denom, 1e-12)


def fit_quality(phases: np.ndarray, weights: np.ndarray, lags: np.ndarray, slope: np.ndarray) -> np.ndarray:
    x = lags[:, None]
    pred = x * slope
    wsum = np.maximum(np.sum(weights, axis=0), 1e-12)
    ybar = np.sum(weights * phases, axis=0) / wsum
    sse = np.sum(weights * (phases - pred) ** 2, axis=0)
    sst = np.sum(weights * (phases - ybar) ** 2, axis=0)
    return np.clip(1.0 - sse / np.maximum(sst, 1e-12), 0.0, 1.0)


def huber_wls(phases: np.ndarray, weights: np.ndarray, lags: np.ndarray, delta: float = 0.7) -> np.ndarray:
    slope = weighted_origin_fit(phases, weights, lags)
    x = lags[:, None]
    for _ in range(5):
        resid = phases - x * slope
        robust = np.minimum(float(delta) / np.maximum(np.abs(resid), 1e-6), 1.0)
        slope = weighted_origin_fit(phases, weights * robust, lags)
    return slope


def leave_one_out_fit(phases: np.ndarray, weights: np.ndarray, lags: np.ndarray) -> np.ndarray:
    candidates = [weighted_origin_fit(phases, weights, lags)]
    if phases.shape[0] >= 4:
        for i in range(phases.shape[0]):
            keep = np.ones(phases.shape[0], dtype=bool)
            keep[i] = False
            candidates.append(weighted_origin_fit(phases[keep], weights[keep], lags[keep]))
    slopes = np.stack(candidates, axis=0)
    pred = lags[None, :, None] * slopes[:, None, :]
    score = np.sum(weights[None] * np.abs(phases[None] - pred), axis=1)
    best = np.argmin(score, axis=0)
    return slopes[best, np.arange(slopes.shape[1])]


def circular_grid_fit(rk: np.ndarray, lags: np.ndarray, grid_size: int = 721) -> np.ndarray:
    phase = np.angle(rk)
    weights = np.abs(rk)
    grid = np.linspace(-np.pi, np.pi, grid_size, dtype=np.float32)
    score = np.sum(
        weights[None] * np.cos(phase[None] - grid[:, None, None] * lags[None, :, None]),
        axis=1,
    )
    return grid[np.argmax(score, axis=0)]


def circular_refined_fit(rk: np.ndarray, lags: np.ndarray) -> np.ndarray:
    coarse = circular_grid_fit(rk, lags)
    phase = np.angle(rk)
    residual = np.angle(np.exp(1j * (phase - lags[:, None] * coarse[None, :])))
    unwrapped = lags[:, None] * coarse[None, :] + residual
    return weighted_origin_fit(unwrapped, np.abs(rk), lags)


def estimate_all(rk: np.ndarray, lags: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    phases = unwrap_lag_phases(np.angle(rk))
    weights = np.abs(rk)
    estimates = {
        "kasai": np.angle(rk[0]),
        "wls_origin": weighted_origin_fit(phases, weights, lags),
        "wls_free": weighted_free_fit(phases, weights, lags),
        "huber_wls": huber_wls(phases, weights, lags),
        "leave_one_out": leave_one_out_fit(phases, weights, lags),
        "circular_grid": circular_grid_fit(rk, lags),
        "circular_refined": circular_refined_fit(rk, lags),
    }
    estimates["circular_sign_wls"] = np.abs(estimates["wls_origin"]) * np.sign(estimates["circular_grid"])
    quality = fit_quality(phases, weights, lags, estimates["wls_origin"])
    return estimates, quality


def wrapped_error(est: np.ndarray, target: float) -> np.ndarray:
    return np.angle(np.exp(1j * (est - float(target))))


def summarize(est: np.ndarray, target: float) -> dict[str, float]:
    err = wrapped_error(est, target)
    abs_err = np.abs(err)
    sign = 1.0 if target >= 0 else -1.0
    return {
        "median_abs_error_rad": float(np.median(abs_err)),
        "rmse_rad": float(np.sqrt(np.mean(err**2))),
        "p90_abs_error_rad": float(np.quantile(abs_err, 0.90)),
        "sign_error_rate": float(np.mean((np.sign(est) * sign) < 0)),
    }


def run(args: argparse.Namespace) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for scenario_idx, scenario in enumerate(SCENARIOS):
        for snr_idx, snr_db in enumerate(args.snr_db):
            for omega_idx, omega in enumerate(args.omega):
                seed = args.seed + 1000 * scenario_idx + 100 * snr_idx + omega_idx
                rng = np.random.default_rng(seed)
                sig = simulate_signal(
                    scenario=scenario,
                    omega=omega,
                    snr_db=snr_db,
                    n_frames=args.n_frames,
                    n_trials=args.n_trials,
                    rng=rng,
                )
                lags, rk = lag_autocorrelations(sig, args.max_lag)
                if scenario == "lag_outlier":
                    rk = corrupt_high_lag(rk, rng)
                estimates, quality = estimate_all(rk, lags)
                for method, est in estimates.items():
                    row = {
                        "scenario": scenario,
                        "method": method,
                        "snr_db": float(snr_db),
                        "omega_rad_per_frame": float(omega),
                        "quality_mean": float(np.mean(quality)),
                        "quality_median": float(np.median(quality)),
                    }
                    row.update(summarize(est, omega))
                    rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "method",
        "snr_db",
        "omega_rad_per_frame",
        "median_abs_error_rad",
        "rmse_rad",
        "p90_abs_error_rad",
        "sign_error_rate",
        "quality_mean",
        "quality_median",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, float | str]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for scenario in SCENARIOS:
        out[scenario] = {}
        for method in METHOD_LABELS:
            sub = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
            if not sub:
                continue
            out[scenario][method] = {
                "mean_median_abs_error_rad": float(np.mean([float(r["median_abs_error_rad"]) for r in sub])),
                "mean_rmse_rad": float(np.mean([float(r["rmse_rad"]) for r in sub])),
                "mean_sign_error_rate": float(np.mean([float(r["sign_error_rate"]) for r in sub])),
                "mean_quality": float(np.mean([float(r["quality_mean"]) for r in sub])),
            }
    return out


def write_plots(output_dir: Path, summary: dict[str, dict[str, dict[str, float]]]) -> None:
    from matplotlib import pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(METHOD_LABELS)
    x = np.arange(len(methods))
    fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(3.8 * len(SCENARIOS), 7.0), constrained_layout=True)
    fig.suptitle("Candidate multi-lag phase estimators: synthetic stress sweep", fontsize=13)

    for col, scenario in enumerate(SCENARIOS):
        med = [summary[scenario][m]["mean_median_abs_error_rad"] for m in methods]
        sign = [summary[scenario][m]["mean_sign_error_rate"] for m in methods]
        axes[0, col].bar(x, med, color="tab:blue", alpha=0.82)
        axes[1, col].bar(x, sign, color="tab:red", alpha=0.82)
        axes[0, col].set_title(scenario.replace("_", " "))
        axes[0, col].set_ylabel("mean median abs error [rad]")
        axes[1, col].set_ylabel("mean sign error")
        axes[1, col].set_ylim(0, 1)
        for ax in axes[:, col]:
            ax.set_xticks(x)
            ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=60, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"phase_ls_research_sweep.{ext}", dpi=180)
    plt.close(fig)


def format_best(value: float, best_value: float, decimals: int, scale: float = 1.0) -> str:
    """Format a table value, bolding ties after rounding to displayed precision."""
    display_value = round(value * scale, decimals)
    display_best = round(best_value * scale, decimals)
    text = f"{display_value:.{decimals}f}"
    if display_value == display_best:
        return rf"\textbf{{{text}}}"
    return text


def write_paper_table(path: Path, summary: dict[str, dict[str, dict[str, float]]]) -> None:
    """Write the exact LaTeX Table 2 block consumed by the manuscript."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by scripts/phase_ls_research_sweep.py.",
        "% Do not edit this table by hand; rerun the script instead.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Estimator performance averaged over SNR and velocity",
        r"  conditions.  Best displayed values in each scenario are shown in",
        r"  \textbf{bold}; ties are determined after rounding to the shown",
        r"  precision.}",
        r"\label{tab:sim_results}",
        r"\small",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"Scenario & Estimator & Med.\ abs.\ err. & RMSE & Sign err. \\",
        r"         &           & (rad)             & (rad) & (\%) \\",
        r"\midrule",
    ]
    scenario_items = list(SCENARIOS)
    for scenario_idx, scenario in enumerate(scenario_items):
        scenario_label = PAPER_SCENARIO_LABELS[scenario]
        scenario_summary = summary[scenario]
        med_best = min(scenario_summary[method]["mean_median_abs_error_rad"] for method, _ in PAPER_METHODS)
        rmse_best = min(scenario_summary[method]["mean_rmse_rad"] for method, _ in PAPER_METHODS)
        sign_best = min(scenario_summary[method]["mean_sign_error_rate"] for method, _ in PAPER_METHODS)

        for method_idx, (method, method_label) in enumerate(PAPER_METHODS):
            stats = scenario_summary[method]
            med = format_best(stats["mean_median_abs_error_rad"], med_best, decimals=3)
            rmse = format_best(stats["mean_rmse_rad"], rmse_best, decimals=3)
            sign = format_best(stats["mean_sign_error_rate"], sign_best, decimals=1, scale=100.0)
            scenario_cell = scenario_label if method_idx == 0 else ""
            lines.append(f"{scenario_cell:<13} & {method_label:<10} & {med} & {rmse} & {sign} \\\\")
        if scenario_idx != len(scenario_items) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run(args)
    summary = aggregate(rows)
    write_csv(args.output_dir / "phase_ls_research_sweep.csv", rows)
    with (args.output_dir / "phase_ls_research_sweep.summary.json").open("w") as f:
        json.dump(
            {
                "n_frames": args.n_frames,
                "n_trials": args.n_trials,
                "max_lag": args.max_lag,
                "snr_db": args.snr_db,
                "omega": args.omega,
                "scenarios": SCENARIOS,
                "paper_methods": [{"method": method, "label": label} for method, label in PAPER_METHODS],
                "paper_table": str(args.paper_table),
                "seed": args.seed,
                "numpy_version": np.__version__,
                "summary": summary,
            },
            f,
            indent=2,
        )
    write_paper_table(args.paper_table, summary)
    write_plots(args.output_dir, summary)
    print(f"Saved research sweep to {args.output_dir}")
    print(f"Saved paper table to {args.paper_table}")


if __name__ == "__main__":
    main()
