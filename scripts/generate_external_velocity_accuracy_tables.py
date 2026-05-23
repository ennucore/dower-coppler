#!/usr/bin/env python3
"""Generate paper tables for external single- vs multi-lag velocity checks.

The manuscript includes compact LaTeX tables for two spatially structured
velocity tests:

* a deterministic PyMUST-compatible synthetic phantom with a known velocity
  field; and
* the PALA InSilicoFlow IQ001 block from Zenodo record 4343435, compared to
  trajectory-derived aliased axial phase velocity.

This script converts the diagnostic CSV outputs into the exact table files
included by paper.tex, and writes a compact JSON/CSV summary for auditability.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PYMUST_ROWS = [
    ("envelope", "Kasai CD", "Envelope", "Kasai"),
    ("envelope", "multi-lag CD", "Envelope", "Multi-lag"),
    ("center_core", "Kasai CD", "Core", "Kasai"),
    ("center_core", "multi-lag CD", "Core", "Multi-lag"),
]

ZENODO_ROWS = [
    ("raw_positions_count_ge_5", "Kasai CD", "Raw $n\\ge5$", "Kasai"),
    ("raw_positions_count_ge_5", "multi-lag CD", "Raw $n\\ge5$", "Multi-lag"),
    ("target_nearest", "Kasai CD", "Target-nearest", "Kasai"),
    ("target_nearest", "multi-lag CD", "Target-nearest", "Multi-lag"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pymust-csv",
        type=Path,
        default=Path("outputs/external_dataset_checks/pymust/pymust_velocity_accuracy.csv"),
    )
    parser.add_argument(
        "--zenodo-csv",
        type=Path,
        default=Path("outputs/external_dataset_checks/zenodo_4343435/zenodo_velocity_estimators_with_svd.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_stats"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Regenerate the external diagnostics before running this table generator."
        )
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pick_row(rows: list[dict[str, str]], *, roi: str, estimator: str, truth: str | None = None) -> dict[str, str]:
    matches = [row for row in rows if row.get("roi") == roi and row.get("estimator") == estimator]
    if truth is not None:
        matches = [row for row in matches if row.get("truth") == truth]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for roi={roi!r}, estimator={estimator!r}, truth={truth!r}; got {len(matches)}")
    return matches[0]


def fmt(value: str | float, ndigits: int = 2) -> str:
    return f"{float(value):.{ndigits}f}"


def write_pymust_table(path: Path, selected: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{PyMUST-compatible signed velocity accuracy for the lag-1",
        r"  Kasai estimator and the $K=5$ multi-lag phase-velocity estimator.",
        r"  The synthetic slow-time phantom used a known parabolic axial-flow",
        r"  field with peak velocity \SI{115}{\milli\meter\per\second}.  The",
        r"  full-envelope ROI includes low-velocity vessel edges; the center-core",
        r"  ROI excludes those edge pixels.}",
        r"\label{tab:pymust_velocity_accuracy}",
        r"\footnotesize",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"ROI & Estimator & Bias & MAE & RMSE \\",
        r" & & \multicolumn{3}{c}{(\si{\milli\meter\per\second})} \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row['paper_roi']} & {row['paper_estimator']} & "
            f"{fmt(row['bias_mm_s'])} & {fmt(row['mae_mm_s'])} & {fmt(row['rmse_mm_s'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_zenodo_table(path: Path, selected: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Zenodo PALA InSilicoFlow signed axial phase-velocity",
        r"  accuracy for lag-1 Kasai and $K=5$ multi-lag phase velocity.",
        r"  Trajectory-derived velocities exceed the \SI{12.32}{\milli\meter\per\second}",
        r"  pulse-Doppler Nyquist limit, so errors are computed against aliased",
        r"  axial phase velocity.  Estimator signs are multiplied by $-1$ to",
        r"  align the PALA increasing-$z$ coordinate with the Doppler phase",
        r"  convention.}",
        r"\label{tab:zenodo_velocity_accuracy}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"ROI & Estimator & Bias & MAE & RMSE & $r$ \\",
        r" & & \multicolumn{3}{c}{(\si{\milli\meter\per\second})} & \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row['paper_roi']} & {row['paper_estimator']} & "
            f"{fmt(row['bias_mm_s'])} & {fmt(row['mae_mm_s'])} & "
            f"{fmt(row['rmse_mm_s'])} & {fmt(row['pearson_r'], 3)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_selected_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "dataset",
        "roi",
        "estimator",
        "truth",
        "paper_roi",
        "paper_estimator",
        "n_pixels",
        "estimator_sign_multiplier",
        "bias_mm_s",
        "mae_mm_s",
        "rmse_mm_s",
        "pearson_r",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_ready(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        converted: dict[str, object] = {}
        for key, value in row.items():
            if key in {"n_pixels"}:
                converted[key] = int(float(value))
            elif key.endswith("_mm_s") or key in {"pearson_r", "estimator_sign_multiplier"}:
                converted[key] = float(value)
            else:
                converted[key] = value
        out.append(converted)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pymust_rows = read_csv(args.pymust_csv)
    zenodo_rows = read_csv(args.zenodo_csv)

    selected_pymust: list[dict[str, object]] = []
    for roi, estimator, paper_roi, paper_estimator in PYMUST_ROWS:
        row = dict(pick_row(pymust_rows, roi=roi, estimator=estimator))
        row.update(
            {
                "dataset": "pymust",
                "truth": "known_velocity",
                "paper_roi": paper_roi,
                "paper_estimator": paper_estimator,
                "estimator_sign_multiplier": row.get("estimator_sign_multiplier", "1"),
            }
        )
        selected_pymust.append(row)

    selected_zenodo: list[dict[str, object]] = []
    for roi, estimator, paper_roi, paper_estimator in ZENODO_ROWS:
        row = dict(pick_row(zenodo_rows, roi=roi, estimator=estimator, truth="aliased_phase_velocity"))
        row.update({"paper_roi": paper_roi, "paper_estimator": paper_estimator})
        selected_zenodo.append(row)

    write_pymust_table(args.output_dir / "pymust_velocity_accuracy_table.tex", selected_pymust)
    write_zenodo_table(args.output_dir / "zenodo_velocity_accuracy_table.tex", selected_zenodo)

    selected = selected_pymust + selected_zenodo
    write_selected_csv(args.output_dir / "external_velocity_accuracy_selected.csv", selected)
    summary = {
        "description": "Selected rows used by manuscript tables for external single- vs multi-lag velocity checks.",
        "generator": "scripts/generate_external_velocity_accuracy_tables.py",
        "inputs": {
            "pymust": str(args.pymust_csv),
            "zenodo": str(args.zenodo_csv),
        },
        "outputs": {
            "pymust_table": str(args.output_dir / "pymust_velocity_accuracy_table.tex"),
            "zenodo_table": str(args.output_dir / "zenodo_velocity_accuracy_table.tex"),
            "selected_csv": str(args.output_dir / "external_velocity_accuracy_selected.csv"),
        },
        "notes": [
            "PyMUST rows compare signed velocity against the known synthetic parabolic flow field.",
            "Zenodo rows compare against trajectory-derived aliased axial phase velocity because physical velocities exceed Nyquist.",
            "Zenodo estimator_sign_multiplier=-1 aligns the PALA z-coordinate convention to the Doppler phase convention.",
        ],
        "rows": json_ready(selected),
    }
    (args.output_dir / "external_velocity_accuracy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {args.output_dir / 'pymust_velocity_accuracy_table.tex'}")
    print(f"Wrote {args.output_dir / 'zenodo_velocity_accuracy_table.tex'}")
    print(f"Wrote {args.output_dir / 'external_velocity_accuracy_selected.csv'}")
    print(f"Wrote {args.output_dir / 'external_velocity_accuracy_summary.json'}")


if __name__ == "__main__":
    main()
