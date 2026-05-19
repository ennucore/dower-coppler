#!/usr/bin/env python3
"""Interactive Doppler CNR/gCNR viewer for prepared median images."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pyqtgraph as pg
from scipy import ndimage
from PyQt5 import QtCore, QtWidgets


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "doppler_cnr_gui"
VELOCITY_ALPHA_METRIC = "Phase Velocity mm/s alpha R2xGeo"
VELOCITY_ALPHA_R2_METRIC = "Phase Velocity mm/s alpha R2"
BLAND_ALTMAN_METRIC = "Bland-Altman: phase velocity vs color"


def _cnr_denominator(signal: np.ndarray, background: np.ndarray, noise_mode: str) -> float:
    if noise_mode == "background":
        return float(np.std(background))
    return float(np.sqrt(float(np.var(signal)) + float(np.var(background))))


def compute_cnr(signal_region, background_region, noise_mode: str = "both") -> tuple[float, float]:
    signal = np.asarray(signal_region, dtype=np.float64)
    background = np.asarray(background_region, dtype=np.float64)
    signal = signal[np.isfinite(signal)]
    background = background[np.isfinite(background)]
    if signal.size == 0 or background.size == 0:
        return np.nan, np.nan
    numerator = abs(float(np.mean(signal)) - float(np.mean(background)))
    denominator = _cnr_denominator(signal, background, noise_mode)
    if denominator <= 0:
        return np.nan, np.nan
    cnr = numerator / denominator
    cnr_db = 20.0 * np.log10(cnr) if cnr > 0 else np.nan
    return float(cnr), float(cnr_db)


def compute_cnr_components(
    signal_region, background_region, noise_mode: str = "both"
) -> tuple[float, float, float, float]:
    signal = np.asarray(signal_region, dtype=np.float64)
    background = np.asarray(background_region, dtype=np.float64)
    signal = signal[np.isfinite(signal)]
    background = background[np.isfinite(background)]
    if signal.size == 0 or background.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    numerator = abs(float(np.mean(signal)) - float(np.mean(background)))
    denominator = _cnr_denominator(signal, background, noise_mode)
    cnr = numerator / denominator if denominator > 0 else np.nan
    cnr_db = 20.0 * np.log10(cnr) if cnr > 0 else np.nan
    return float(cnr), float(cnr_db), float(numerator), float(denominator)


def compute_contrast(signal_region, background_region) -> float:
    signal = np.asarray(signal_region, dtype=np.float64)
    background = np.asarray(background_region, dtype=np.float64)
    signal = signal[np.isfinite(signal)]
    background = background[np.isfinite(background)]
    if signal.size == 0 or background.size == 0:
        return np.nan
    mean_background = float(np.mean(background))
    mean_signal = float(np.mean(signal))
    if mean_background == 0:
        return np.nan
    ratio = mean_signal / mean_background
    if not np.isfinite(ratio) or ratio <= 0:
        return np.nan
    return float(10.0 * np.log10(ratio))


def compute_gcnr(signal_region, background_region, bins: int = 256) -> float:
    signal = np.asarray(signal_region, dtype=np.float64).ravel()
    background = np.asarray(background_region, dtype=np.float64).ravel()
    signal = signal[np.isfinite(signal)]
    background = background[np.isfinite(background)]
    if signal.size == 0 or background.size == 0:
        return np.nan
    low = min(float(signal.min()), float(background.min()))
    high = max(float(signal.max()), float(background.max()))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 0.0
    hist_signal, edges = np.histogram(signal, bins=bins, range=(low, high), density=True)
    hist_background, _ = np.histogram(background, bins=edges, range=(low, high), density=True)
    overlap = np.sum(np.minimum(hist_signal, hist_background) * np.diff(edges))
    return float(np.clip(1.0 - overlap, 0.0, 1.0))


def gcnr_db(gcnr: float) -> float:
    if not np.isfinite(gcnr) or gcnr <= 0:
        return np.nan
    return float(10.0 * np.log10(gcnr))


@dataclass
class Dataset:
    name: str
    path: Path
    arrays: Dict[str, np.ndarray]
    extent: np.ndarray | None = None
    window_labels: list[str] | None = None
    selection_window_acqs: int = 1
    acq_indices: np.ndarray | None = None

    @property
    def n_planes(self) -> int:
        first = next(iter(self.arrays.values()))
        if first.ndim == 4:
            return int(first.shape[1])
        return int(first.shape[0]) if first.ndim == 3 else 1

    @property
    def n_windows(self) -> int:
        first = next(iter(self.arrays.values()))
        if first.ndim != 4:
            return 1
        return int(np.ceil(first.shape[0] / max(1, self.selection_window_acqs)))

    @property
    def is_per_acq(self) -> bool:
        return self.acq_indices is not None


def ensure_planes(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim not in {3, 4}:
        raise ValueError(f"Expected 2D, 3D, or 4D image array, got shape {arr.shape}")
    return arr


def robust_signed_normalize(arr: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return np.zeros_like(arr, dtype=np.float32)
    scale = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / scale).astype(np.float32)


def signed_doppler_variant(metric: str) -> bool:
    return metric in {
        "Color Doppler",
        "Dower Coppler",
        "Phase Velocity",
        VELOCITY_ALPHA_METRIC,
        VELOCITY_ALPHA_R2_METRIC,
        "Phase Velocity x R2",
        "Phase Velocity x Geomean |Rk| x R2",
        "Dower x PhaseR2",
        "Signed Scale",
        "Signed Scale x HuberQ",
        "Signed Geomean |Rk|",
        "Signed Geomean |Rk| x HuberQ",
        "Signed Geomean |Rk| x HuberQ x R2",
        "Dower x HuberQ",
        "Phase-sign Dower",
        "Dower-sign PVxR2",
        "Sign-agree Dower",
        "Agree Geomean",
    }


def velocity_alpha_metric(metric: str) -> bool:
    return metric in {VELOCITY_ALPHA_METRIC, VELOCITY_ALPHA_R2_METRIC}


def bland_altman_metric(metric: str) -> bool:
    return metric == BLAND_ALTMAN_METRIC


def raw_iq_trace_metric(metric: str) -> bool:
    return metric.startswith("Raw IQ Middle Element Trace")


def load_dataset(name: str, path: Path) -> Dataset:
    with np.load(path) as z:
        arrays = {}
        if "power_doppler" in z.files:
            arrays["Power Doppler"] = ensure_planes(z["power_doppler"])
        if "color_doppler" in z.files:
            arrays["Color Doppler"] = ensure_planes(z["color_doppler"])
        if "dower_coppler" in z.files:
            arrays["Dower Coppler"] = ensure_planes(z["dower_coppler"])
        if "raw_iq_magnitude" in z.files:
            arrays["Raw Pre-BF IQ Magnitude"] = ensure_planes(z["raw_iq_magnitude"])
        if "raw_iq_middle_element_first_frame" in z.files:
            arrays["Raw IQ Middle Element Trace - first non-noise frame"] = ensure_planes(
                np.asarray(z["raw_iq_middle_element_first_frame"], dtype=np.float32)[:, None]
            )
        if "raw_iq_middle_element_middle_frame" in z.files:
            arrays["Raw IQ Middle Element Trace - middle frame"] = ensure_planes(
                np.asarray(z["raw_iq_middle_element_middle_frame"], dtype=np.float32)[:, None]
            )
        if "phase_velocity" in z.files:
            arrays["Phase Velocity"] = ensure_planes(z["phase_velocity"])
        if "phase_velocity_r2" in z.files:
            arrays["Phase Velocity x R2"] = ensure_planes(z["phase_velocity_r2"])
        if "phase_r2" in z.files:
            arrays["Phase Fit R2"] = ensure_planes(z["phase_r2"])
        if "phase_velocity_r1_to_r5_r2_fit_quality" in z.files:
            arrays["Phase Velocity x |R1..R5| x R2 FitQ"] = ensure_planes(
                z["phase_velocity_r1_to_r5_r2_fit_quality"]
            )
        if "huber_quality" in z.files:
            arrays["Huber Fit Quality"] = ensure_planes(z["huber_quality"])
        if "signed_scale" in z.files:
            arrays["Signed Scale"] = ensure_planes(z["signed_scale"])
        if "signed_scale_huber_quality" in z.files:
            arrays["Signed Scale x HuberQ"] = ensure_planes(z["signed_scale_huber_quality"])
        if "geomean_r" in z.files:
            arrays["Geomean |Rk|"] = ensure_planes(z["geomean_r"])
            pv_geo_r2 = None
            if "Phase Velocity" in arrays and "Phase Fit R2" in arrays:
                arrays[VELOCITY_ALPHA_METRIC] = (arrays["Phase Velocity"] * 1000.0).astype(np.float32)
                arrays[VELOCITY_ALPHA_R2_METRIC] = arrays[VELOCITY_ALPHA_METRIC]
                if "Color Doppler" in arrays:
                    arrays[BLAND_ALTMAN_METRIC] = arrays[VELOCITY_ALPHA_METRIC]
                pv_geo_r2 = (
                    arrays["Phase Velocity"]
                    * arrays["Geomean |Rk|"]
                    * arrays["Phase Fit R2"]
                ).astype(np.float32)
            elif "Phase Velocity x R2" in arrays:
                pv_geo_r2 = (
                    arrays["Phase Velocity x R2"] * arrays["Geomean |Rk|"]
                ).astype(np.float32)
            if pv_geo_r2 is not None:
                arrays["Phase Velocity x Geomean |Rk| x R2"] = pv_geo_r2
                arrays["Dower Coppler"] = pv_geo_r2
        if "signed_geomean_r" in z.files:
            arrays["Signed Geomean |Rk|"] = ensure_planes(z["signed_geomean_r"])
        if "signed_geomean_r_huber_quality" in z.files:
            arrays["Signed Geomean |Rk| x HuberQ"] = ensure_planes(z["signed_geomean_r_huber_quality"])
            if "Phase Fit R2" in arrays:
                arrays["Signed Geomean |Rk| x HuberQ x R2"] = (
                    ensure_planes(z["signed_geomean_r_huber_quality"])
                    * arrays["Phase Fit R2"]
                ).astype(np.float32)
        if "dower_huber_quality" in z.files:
            arrays["Dower x HuberQ"] = ensure_planes(z["dower_huber_quality"])
        if "Dower Coppler" in arrays and "Phase Velocity x R2" in arrays:
            dower = arrays["Dower Coppler"]
            pv_r2 = arrays["Phase Velocity x R2"]
            dower_norm = robust_signed_normalize(dower)
            pv_r2_norm = robust_signed_normalize(pv_r2)
            same_sign = np.sign(dower) == np.sign(pv_r2)
            same_sign &= dower != 0
            same_sign &= pv_r2 != 0
            if "Phase Fit R2" in arrays:
                arrays["Dower x PhaseR2"] = (dower * arrays["Phase Fit R2"]).astype(np.float32)
            arrays["Phase-sign Dower"] = (np.sign(pv_r2) * np.abs(dower)).astype(np.float32)
            arrays["Dower-sign PVxR2"] = (np.sign(dower) * np.abs(pv_r2)).astype(np.float32)
            arrays["Sign-agree Dower"] = np.where(same_sign, dower, 0.0).astype(np.float32)
            arrays["Agree Geomean"] = np.where(
                same_sign,
                np.sign(dower_norm) * np.sqrt(np.abs(dower_norm) * np.abs(pv_r2_norm)),
                0.0,
            ).astype(np.float32)
        if not arrays:
            raise ValueError(f"No displayable arrays found in {path}")
        extent = z["extent"].astype(np.float32) if "extent" in z.files else None
        if extent is not None and max(abs(float(v)) for v in extent) > 30.0:
            extent = (extent / 10.0).astype(np.float32)
        if extent is None and "x_mm" in z.files and "z_mm" in z.files:
            x_mm = np.asarray(z["x_mm"], dtype=np.float32)
            z_mm = np.asarray(z["z_mm"], dtype=np.float32)
            extent = np.array(
                [
                    float(np.nanmin(x_mm)) / 10.0,
                    float(np.nanmax(x_mm)) / 10.0,
                    float(np.nanmax(z_mm)) / 10.0,
                    float(np.nanmin(z_mm)) / 10.0,
                ],
                dtype=np.float32,
            )
        window_labels = [str(v) for v in z["window_labels"]] if "window_labels" in z.files else None
        selection_window_acqs = int(z["selection_window_acqs"]) if "selection_window_acqs" in z.files else 1
        acq_indices = z["acq_indices"].astype(np.int32) if "acq_indices" in z.files else None
    return Dataset(
        name=name,
        path=path,
        arrays=arrays,
        extent=extent,
        window_labels=window_labels,
        selection_window_acqs=selection_window_acqs,
        acq_indices=acq_indices,
    )


class CnrViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Doppler CNR / gCNR Viewer")
        self.resize(1180, 760)

        self.datasets = self._load_available_datasets()
        self.current_dataset = self.datasets[0]
        self.current_window_start = 0
        self.current_window_end = 0
        self.current_bin_acqs = max(1, self.current_dataset.selection_window_acqs)
        self.current_metric = "Signed Scale x HuberQ"
        if self.current_metric not in self.current_dataset.arrays:
            self.current_metric = next(iter(self.current_dataset.arrays))
        self.current_plane = 0
        self.compare_enabled = False
        self.compare_dataset = self.datasets[1] if len(self.datasets) > 1 else self.datasets[0]
        self.current_image = np.zeros((2, 2), dtype=np.float32)
        self.display_image = np.zeros((2, 2), dtype=np.float32)
        self.auto_items: list[pg.GraphicsObject] = []
        self.selections: list[dict] = []
        self.selection_items: list[pg.GraphicsObject] = []
        self.pending_segment: dict | None = None
        self.pending_segment_items: list[pg.GraphicsObject] = []
        self.circle_drag_start: tuple[int, int] | None = None
        self.circle_drag_item: pg.EllipseROI | None = None
        self.background_drag_start: tuple[int, int] | None = None
        self.background_drag_item: pg.RectROI | None = None
        self.next_selection_id = 1
        self.manual_selection_id: int | None = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        layout.addWidget(main_splitter)

        controls_widget = QtWidgets.QWidget()
        controls = QtWidgets.QVBoxLayout(controls_widget)
        controls.setContentsMargins(6, 6, 6, 6)
        controls.setSpacing(4)
        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_widget)
        controls_scroll.setMinimumWidth(240)
        main_splitter.addWidget(controls_scroll)

        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.addItems([d.name for d in self.datasets])
        self.dataset_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.dataset_combo.setMinimumContentsLength(28)
        self.dataset_combo.currentIndexChanged.connect(self._dataset_changed)
        controls.addWidget(QtWidgets.QLabel("Dataset"))
        controls.addWidget(self.dataset_combo)

        self.compare_check = QtWidgets.QCheckBox("Compare side by side")
        self.compare_check.toggled.connect(self._compare_toggled)
        controls.addWidget(self.compare_check)

        self.compare_combo = QtWidgets.QComboBox()
        self.compare_combo.addItems([d.name for d in self.datasets])
        self.compare_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.compare_combo.setMinimumContentsLength(28)
        if len(self.datasets) > 1:
            self.compare_combo.setCurrentIndex(1)
        self.compare_combo.setEnabled(False)
        self.compare_combo.currentIndexChanged.connect(self._compare_dataset_changed)
        controls.addWidget(QtWidgets.QLabel("Comparison dataset"))
        controls.addWidget(self.compare_combo)

        self.bin_acqs_spin = QtWidgets.QSpinBox()
        self.bin_acqs_spin.setRange(1, 999999)
        self.bin_acqs_spin.setValue(self.current_bin_acqs)
        self.bin_acqs_spin.valueChanged.connect(self._bin_acqs_changed)
        controls.addWidget(QtWidgets.QLabel("Bin size [acqs]"))
        controls.addWidget(self.bin_acqs_spin)

        self.window_start_spin = QtWidgets.QSpinBox()
        self.window_start_spin.valueChanged.connect(self._window_start_changed)
        controls.addWidget(QtWidgets.QLabel("Range start"))
        controls.addWidget(self.window_start_spin)

        self.window_end_spin = QtWidgets.QSpinBox()
        self.window_end_spin.valueChanged.connect(self._window_end_changed)
        controls.addWidget(QtWidgets.QLabel("Range end"))
        controls.addWidget(self.window_end_spin)

        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItems(list(self.current_dataset.arrays.keys()))
        self.metric_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.metric_combo.setMinimumContentsLength(22)
        self.metric_combo.setCurrentText(self.current_metric)
        self.metric_combo.currentTextChanged.connect(self._metric_changed)
        controls.addWidget(QtWidgets.QLabel("Metric"))
        controls.addWidget(self.metric_combo)

        self.pd_scale_combo = QtWidgets.QComboBox()
        self.pd_scale_combo.addItems(["PD dB, 15 dB range", "PD linear, vmax p96"])
        self.pd_scale_combo.currentTextChanged.connect(lambda _: self._refresh_all())
        controls.addWidget(QtWidgets.QLabel("Power Doppler display"))
        controls.addWidget(self.pd_scale_combo)

        self.pd_db_range = QtWidgets.QDoubleSpinBox()
        self.pd_db_range.setRange(1.0, 80.0)
        self.pd_db_range.setDecimals(1)
        self.pd_db_range.setSingleStep(1.0)
        self.pd_db_range.setValue(15.0)
        self.pd_db_range.valueChanged.connect(lambda _: self._refresh_all())
        controls.addWidget(QtWidgets.QLabel("PD dB cutoff/range"))
        controls.addWidget(self.pd_db_range)

        self.pd_linear_percentile = QtWidgets.QDoubleSpinBox()
        self.pd_linear_percentile.setRange(1.0, 100.0)
        self.pd_linear_percentile.setDecimals(1)
        self.pd_linear_percentile.setSingleStep(1.0)
        self.pd_linear_percentile.setValue(96.0)
        self.pd_linear_percentile.valueChanged.connect(lambda _: self._refresh_all())
        controls.addWidget(QtWidgets.QLabel("PD linear vmax percentile"))
        controls.addWidget(self.pd_linear_percentile)

        self.cd_abs_percentile = QtWidgets.QDoubleSpinBox()
        self.cd_abs_percentile.setRange(1.0, 100.0)
        self.cd_abs_percentile.setDecimals(1)
        self.cd_abs_percentile.setSingleStep(1.0)
        self.cd_abs_percentile.setValue(99.0)
        self.cd_abs_percentile.valueChanged.connect(lambda _: self._refresh_all())
        controls.addWidget(QtWidgets.QLabel("CD abs limit percentile"))
        controls.addWidget(self.cd_abs_percentile)

        self.dc_abs_percentile = QtWidgets.QDoubleSpinBox()
        self.dc_abs_percentile.setRange(1.0, 100.0)
        self.dc_abs_percentile.setDecimals(1)
        self.dc_abs_percentile.setSingleStep(1.0)
        self.dc_abs_percentile.setValue(99.0)
        self.dc_abs_percentile.valueChanged.connect(lambda _: self._refresh_all())
        controls.addWidget(QtWidgets.QLabel("DC abs limit percentile"))
        controls.addWidget(self.dc_abs_percentile)

        self.cnr_noise_combo = QtWidgets.QComboBox()
        self.cnr_noise_combo.addItems(["Noise: signal + background", "Noise: background only"])
        self.cnr_noise_combo.currentIndexChanged.connect(self._cnr_noise_changed)
        controls.addWidget(QtWidgets.QLabel("CNR noise denominator"))
        controls.addWidget(self.cnr_noise_combo)

        self.plane_spin = QtWidgets.QSpinBox()
        self.plane_spin.setMinimum(0)
        self.plane_spin.valueChanged.connect(self._plane_changed)
        controls.addWidget(QtWidgets.QLabel("Plane"))
        controls.addWidget(self.plane_spin)

        self.auto_radius = QtWidgets.QSpinBox()
        self.auto_radius.setRange(1, 80)
        self.auto_radius.setValue(6)
        controls.addWidget(QtWidgets.QLabel("Auto click signal radius [px]"))
        controls.addWidget(self.auto_radius)

        self.auto_button = QtWidgets.QPushButton("Auto: click image")
        self.auto_button.setCheckable(True)
        self.auto_button.toggled.connect(self._auto_toggled)
        controls.addWidget(self.auto_button)

        self.segment_radius = QtWidgets.QSpinBox()
        self.segment_radius.setRange(3, 200)
        self.segment_radius.setValue(28)
        controls.addWidget(QtWidgets.QLabel("CV segment search radius [px]"))
        controls.addWidget(self.segment_radius)

        self.segment_tolerance = QtWidgets.QDoubleSpinBox()
        self.segment_tolerance.setRange(0.01, 1.0)
        self.segment_tolerance.setDecimals(2)
        self.segment_tolerance.setSingleStep(0.02)
        self.segment_tolerance.setValue(0.16)
        controls.addWidget(QtWidgets.QLabel("CV segment tolerance"))
        controls.addWidget(self.segment_tolerance)

        self.segment_erosion = QtWidgets.QSpinBox()
        self.segment_erosion.setRange(0, 20)
        self.segment_erosion.setValue(1)
        controls.addWidget(QtWidgets.QLabel("CV segment erosion [px]"))
        controls.addWidget(self.segment_erosion)

        self.segment_button = QtWidgets.QPushButton("CV segment: signal then bg rect")
        self.segment_button.setCheckable(True)
        self.segment_button.toggled.connect(self._segment_toggled)
        controls.addWidget(self.segment_button)

        self.circle_button = QtWidgets.QPushButton("Circle: signal then bg rect")
        self.circle_button.setCheckable(True)
        self.circle_button.toggled.connect(self._circle_toggled)
        controls.addWidget(self.circle_button)

        self.manual_button = QtWidgets.QPushButton("Manual: signal/background ROIs")
        self.manual_button.clicked.connect(self._create_manual_rois)
        controls.addWidget(self.manual_button)

        self.save_pdf_button = QtWidgets.QPushButton("Save PDF")
        self.save_pdf_button.clicked.connect(self._save_pdf)
        controls.addWidget(self.save_pdf_button)

        self.remove_latest_button = QtWidgets.QPushButton("Remove latest selection")
        self.remove_latest_button.clicked.connect(self._remove_latest_selection)
        controls.addWidget(self.remove_latest_button)

        self.remove_id_spin = QtWidgets.QSpinBox()
        self.remove_id_spin.setRange(1, 999999)
        controls.addWidget(QtWidgets.QLabel("Selection ID to remove"))
        controls.addWidget(self.remove_id_spin)

        self.remove_id_button = QtWidgets.QPushButton("Remove selection ID")
        self.remove_id_button.clicked.connect(self._remove_selection_id)
        controls.addWidget(self.remove_id_button)

        self.clear_button = QtWidgets.QPushButton("Clear measurements")
        self.clear_button.clicked.connect(self._clear_measurements)
        controls.addWidget(self.clear_button)

        controls.addWidget(QtWidgets.QLabel("Results"))
        self.results = QtWidgets.QPlainTextEdit()
        self.results.setReadOnly(True)
        self.results.setMinimumWidth(290)
        controls.addWidget(self.results, 1)

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setLabel("bottom", "x [cm]")
        self.plot.setLabel("left", "z [cm]")
        self.plot.setMinimumWidth(360)
        main_splitter.addWidget(self.plot)

        self.image_item = pg.ImageItem(axisOrder="col-major")
        self.plot.addItem(self.image_item)
        self.trace_item = self.plot.plot([], [], pen=pg.mkPen("#f59e0b", width=2))
        self.trace_item.hide()
        self.ba_items: list[pg.GraphicsObject] = []

        self.compare_plot = pg.PlotWidget()
        self.compare_plot.setAspectLocked(True)
        self.compare_plot.showGrid(x=True, y=True, alpha=0.18)
        self.compare_plot.setLabel("bottom", "x [cm]")
        self.compare_plot.setLabel("left", "z [cm]")
        self.compare_plot.setMinimumWidth(360)
        self.compare_plot.setXLink(self.plot)
        self.compare_plot.setYLink(self.plot)
        main_splitter.addWidget(self.compare_plot)

        self.compare_image_item = pg.ImageItem(axisOrder="col-major")
        self.compare_plot.addItem(self.compare_image_item)
        self.compare_trace_item = self.compare_plot.plot([], [], pen=pg.mkPen("#f59e0b", width=2))
        self.compare_trace_item.hide()
        self.compare_plot.hide()

        self.hist = pg.HistogramLUTItem(image=self.image_item)
        hist_widget = pg.GraphicsLayoutWidget()
        hist_widget.addItem(self.hist)
        hist_widget.setMinimumWidth(90)
        main_splitter.addWidget(hist_widget)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 1)
        main_splitter.setStretchFactor(3, 0)
        main_splitter.setSizes([330, 560, 560, 130])

        self.signal_roi: pg.RectROI | None = None
        self.background_roi: pg.RectROI | None = None
        self.manual_text: pg.TextItem | None = None

        self.plot.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.plot.viewport().installEventFilter(self)
        self._refresh_all()

    def _load_available_datasets(self) -> list[Dataset]:
        specs = [
            (
                "2026-05-18 BT24480388 tx_el -5 deg H5 row -1 fine x/z, y -6.5 to +0.5 mm, 10 elev, all 10 acqs",
                DATA_DIR
                / "bt24480388_2026-05-18_152812_txel-5_h5_row-1_fine_xz_y-6p5to0p5mm_10elev_all10.npz",
            ),
            (
                "2026-05-18 BT24480388 tx_el 0 deg H5 row -1 fine x/z, y -3.5 to +3.5 mm, 10 elev, all 20 acqs",
                DATA_DIR
                / "bt24480388_2026-05-18_152605_txel0_h5_row-1_fine_xz_y-3p5to3p5mm_10elev_all20.npz",
            ),
            (
                "2026-05-18 BT24480388 tx_el +5 deg H5 row -1 fine x/z, y -0.5 to +6.5 mm, 10 elev, all 9 acqs",
                DATA_DIR
                / "bt24480388_2026-05-18_152924_txel5_h5_row-1_fine_xz_y-0p5to6p5mm_10elev_all9.npz",
            ),
            (
                "2026-05-15 lev_may15_replication H5 row -1, all 47 acqs",
                DATA_DIR / "lev_may15_replication_h5_row-1_all47.npz",
            ),
            (
                "2026-05-15 lev_may15_replication H5 row -1, first half acqs 0-22",
                DATA_DIR / "lev_may15_replication_h5_row-1_first23.npz",
            ),
            (
                "2026-05-15 lev_may15_replication H5 row -1, second half acqs 23-46",
                DATA_DIR / "lev_may15_replication_h5_row-1_last24.npz",
            ),
            (
                "2026-05-15 lev_may15_replication H5 row -1 fine x/z, all 47 acqs",
                DATA_DIR / "lev_may15_replication_h5_row-1_fine_xz_all47.npz",
            ),
            (
                "2026-05-15 lev_may15_replication H5 row -1 fine x/z, first half acqs 0-22",
                DATA_DIR / "lev_may15_replication_h5_row-1_fine_xz_first23.npz",
            ),
            (
                "2026-05-15 lev_may15_replication H5 row -1 fine x/z, second half acqs 23-46",
                DATA_DIR / "lev_may15_replication_h5_row-1_fine_xz_last24.npz",
            ),
            (
                "2026-05-15 lev_may15_replication H5 row -1 fine x/z, y -3.5 to +3.5 mm, 18 elev, all 47 acqs",
                DATA_DIR
                / "lev_may15_replication_h5_fast8_fine_xz_savedgrid_y-3p5to3p5mm_18elev_mean_all47.npz",
            ),
            (
                "2026-05-15 lev_may15_replication H5 row -1 fine x/z, y -3.5 to +3.5 mm, 18 elev, first 23 acqs",
                DATA_DIR
                / "lev_may15_replication_h5_fast8_fine_xz_savedgrid_y-3p5to3p5mm_18elev_mean_first23.npz",
            ),
            (
                "2026-05-15 lev_may15_replication H5 row -1 fine x/z, y -3.5 to +3.5 mm, 18 elev, last 24 acqs",
                DATA_DIR
                / "lev_may15_replication_h5_fast8_fine_xz_savedgrid_y-3p5to3p5mm_18elev_mean_last24.npz",
            ),
            (
                "2026-05-18 BT24480388 tx_el 0 deg H5 row -1 fine x/z, all 20 acqs",
                DATA_DIR / "bt24480388_2026-05-18_152605_txel0_h5_row-1_fine_xz_all20.npz",
            ),
            (
                "2026-05-18 BT24480388 tx_el 0 deg H5 row -1 fine x/z, first 10 acqs",
                DATA_DIR
                / "bt24480388_2026-05-18_152605_txel0_h5_row-1_fine_xz_first10.npz",
            ),
            (
                "2026-05-18 BT24480388 tx_el 0 deg H5 row -1 fine x/z, last 10 acqs",
                DATA_DIR / "bt24480388_2026-05-18_152605_txel0_h5_row-1_fine_xz_last10.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 row -1, 18:16:49, all 21 acqs",
                DATA_DIR / "bt24480388_2026-05-14_181649_h5_gpu_row-1_all21.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 row -1, 18:16:49, first half acqs 0-9",
                DATA_DIR / "bt24480388_2026-05-14_181649_h5_gpu_row-1_first10.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 row -1, 18:16:49, second half acqs 10-20",
                DATA_DIR / "bt24480388_2026-05-14_181649_h5_gpu_row-1_last11.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 row -1, 18:14:29, all 13 acqs",
                DATA_DIR / "bt24480388_2026-05-14_181429_h5_gpu_row-1_all13.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 row -1, 18:14:29, first half acqs 0-5",
                DATA_DIR / "bt24480388_2026-05-14_181429_h5_gpu_row-1_first6.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 row -1, 18:14:29, second half acqs 6-12",
                DATA_DIR / "bt24480388_2026-05-14_181429_h5_gpu_row-1_last7.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 row -1 fine x/z, 18:14:29, all 13 acqs",
                DATA_DIR / "bt24480388_2026-05-14_181429_h5_gpu_row-1_fine_xz_all13.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 row -1 fine x/z, 18:14:29, first half acqs 0-5",
                DATA_DIR / "bt24480388_2026-05-14_181429_h5_gpu_row-1_fine_xz_first6.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 row -1 fine x/z, 18:14:29, second half acqs 6-12",
                DATA_DIR / "bt24480388_2026-05-14_181429_h5_gpu_row-1_fine_xz_last7.npz",
            ),
            (
                "2026-05-14 BT24480388 raw IQ, 18:14:29, middle acq 6, middle row",
                DATA_DIR / "bt24480388_2026-05-14_181429_raw_iq_mid_acq6_midrow.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 GPU row 0, 18:16:49, all 21 acqs",
                DATA_DIR / "bt24480388_2026-05-14_181649_h5_gpu_row0_all21.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 GPU row 0, 18:16:49, first half acqs 0-9",
                DATA_DIR / "bt24480388_2026-05-14_181649_h5_gpu_row0_first10.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 GPU row 0, 18:16:49, second half acqs 10-20",
                DATA_DIR / "bt24480388_2026-05-14_181649_h5_gpu_row0_last11.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 GPU row 0, 18:14:29, all 13 acqs",
                DATA_DIR / "bt24480388_2026-05-14_181429_h5_gpu_row0_all13.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 GPU row 0, 18:14:29, first half acqs 0-5",
                DATA_DIR / "bt24480388_2026-05-14_181429_h5_gpu_row0_first6.npz",
            ),
            (
                "2026-05-14 BT24480388 H5 GPU row 0, 18:14:29, second half acqs 6-12",
                DATA_DIR / "bt24480388_2026-05-14_181429_h5_gpu_row0_last7.npz",
            ),
            (
                "2025-09-18 Head existing compound, all 50 acqs",
                DATA_DIR / "head_2025-09-18_215438_existing_compound_all50.npz",
            ),
            (
                "2025-09-18 Head existing compound, first half acqs 0-24",
                DATA_DIR / "head_2025-09-18_215438_existing_compound_first25.npz",
            ),
            (
                "2025-09-18 Head existing compound, second half acqs 25-49",
                DATA_DIR / "head_2025-09-18_215438_existing_compound_last25.npz",
            ),
            (
                "2025-11-13 BT22041607 existing compound, all 118 acqs",
                DATA_DIR / "bt22041607_2025-11-13_122559_existing_compound_all118.npz",
            ),
            (
                "2025-11-13 BT22041607 existing compound, first half acqs 0-58",
                DATA_DIR / "bt22041607_2025-11-13_122559_existing_compound_first59.npz",
            ),
            (
                "2025-11-13 BT22041607 existing compound, second half acqs 59-117",
                DATA_DIR / "bt22041607_2025-11-13_122559_existing_compound_last59.npz",
            ),
            (
                "2025-11-12 Wilson existing compound, single capture",
                DATA_DIR / "wilson_2025-11-12_153927_existing_compound_all1.npz",
            ),
            (
                "2025-10-14 Wilson existing compound, all 15 acqs",
                DATA_DIR / "wilson_2025-10-14_191927_existing_compound_all15.npz",
            ),
            (
                "2025-10-14 Wilson existing compound, first half acqs 0-6",
                DATA_DIR / "wilson_2025-10-14_191927_existing_compound_first7.npz",
            ),
            (
                "2025-10-14 Wilson existing compound, second half acqs 7-14",
                DATA_DIR / "wilson_2025-10-14_191927_existing_compound_last8.npz",
            ),
            (
                "2026-02-28 Kenny 1-back fus_encoder SVD, acqs 0-7",
                DATA_DIR / "kenny_1back_20260228_fus_svd_acq0000_0007.npz",
            ),
            (
                "2026-04-01 Kenny root fus_encoder SVD, acqs 0-7",
                DATA_DIR / "kenny_root_20260401_fus_svd_acq0000_0007.npz",
            ),
            (
                "2026-05-13 Kenny/Raffi latest zarr GPU beamformed, row 0, all 25 acqs",
                DATA_DIR / "kenny_raffi_2026-05-13_161627_zarr_gpu_row0_all25.npz",
            ),
            (
                "2026-05-13 Kenny/Raffi latest zarr GPU beamformed, row 0, first half acqs 0-11",
                DATA_DIR / "kenny_raffi_2026-05-13_161627_zarr_gpu_row0_first12.npz",
            ),
            (
                "2026-05-13 Kenny/Raffi latest zarr GPU beamformed, row 0, second half acqs 12-24",
                DATA_DIR / "kenny_raffi_2026-05-13_161627_zarr_gpu_row0_last13.npz",
            ),
            (
                "2025-09-21 new H5 recomputed Dower + phase fit, acqs 200-399, middle 8 elev",
                DATA_DIR / "head_2025-09-21_new_h5_recomputed_dower_acq200_399_mid8elev.npz",
            ),
            (
                "2025-09-21 fine x/z rebeamform, y idx 14-15, acqs 200-399",
                DATA_DIR / "head_2025-09-21_fine_xz_yidx14_15_acq200_399.npz",
            ),
            (
                "2025-09-21 full 2D TX fine x/z, y idx 14-15, acqs 200-399",
                DATA_DIR / "head_2025-09-21_full2dtx_fine_xz_yidx14_15_acq200_399.npz",
            ),
            (
                "2025-09-21 full 2D TX fine x/z, y -5 to +5 mm, 10 elev, acqs 200-399",
                DATA_DIR / "head_2025-09-21_full2dtx_fine_xz_y-5to5mm_10elev_acq200_399.npz",
            ),
            (
                "2025-09-21 legacy TX fine x/z, y idx 14-15, acqs 200-209",
                DATA_DIR / "head_2025-09-21_legacytx_fine_xz_yidx14_15_acq200_209.npz",
            ),
            (
                "2025-09-21 full 2D TX fine x/z, y idx 14-15, acqs 200-209",
                DATA_DIR / "head_2025-09-21_full2dtx_fine_xz_yidx14_15_acq200_209.npz",
            ),
            (
                "2025-09-21 full 2D TX fast8 fine x/z, y idx 14-15, acqs 200-209",
                DATA_DIR / "head_2025-09-21_full2dtx_fast8_fine_xz_yidx14_15_acq200_209.npz",
            ),
            (
                "2025-09-21 full 2D TX fine x/z, y -5 to +5 mm, 10 elev, acqs 200-209",
                DATA_DIR / "head_2025-09-21_full2dtx_fine_xz_y-5to5mm_10elev_acq200_209.npz",
            ),
            (
                "2025-09-21 full 2D TX fast8 fine x/z, y -5 to +5 mm, 10 elev, acqs 200-209",
                DATA_DIR / "head_2025-09-21_full2dtx_fast8_fine_xz_y-5to5mm_10elev_acq200_209.npz",
            ),
            (
                "2025-09-21 full 2D TX fast8 fine x/z, y -4 to +4 mm, 10 elev, acqs 200-400 per-acq",
                DATA_DIR / "head_2025-09-21_full2dtx_fast8_fine_xz_y-4to4mm_10elev_acq200_400.npz",
            ),
            (
                "2025-09-21 new H5 recomputed Dower + phase fit, first half acqs 200-299, middle 8 elev",
                DATA_DIR / "head_2025-09-21_new_h5_recomputed_dower_acq200_299_mid8elev.npz",
            ),
            (
                "2025-09-21 new H5 recomputed Dower + phase fit, second half acqs 300-399, middle 8 elev",
                DATA_DIR / "head_2025-09-21_new_h5_recomputed_dower_acq300_399_mid8elev.npz",
            ),
            (
                "2025-09-21 new H5 recomputed Dower + phase fit, acqs 200-239, middle 8 elev",
                DATA_DIR / "head_2025-09-21_new_h5_recomputed_dower_acq200_239_mid8elev.npz",
            ),
            (
                "2025-09-21 new H5 recomputed Dower, acqs 200-239, 22 elev",
                DATA_DIR / "head_2025-09-21_new_h5_recomputed_dower_acq200_239_22elev.npz",
            ),
            (
                "2025-09-21 head acqs 150-250, full post-cutoff SVD, k10",
                DATA_DIR / "head_2025-09-21_per_acq_doppler_full_post_cutoff_k10_acq150_250.npz",
            ),
            (
                "2025-09-21 head per-acq median ranges, full post-cutoff SVD, k10",
                DATA_DIR / "head_2025-09-21_per_acq_doppler_full_post_cutoff_k10.npz",
            ),
            (
                "2025-09-21 head per-acq median ranges, full post-cutoff SVD, k5",
                DATA_DIR / "head_2025-09-21_per_acq_doppler_full_post_cutoff.npz",
            ),
            (
                "2025-09-21 head median, all 480 acqs",
                DATA_DIR / "head_2025-09-21_all480_median.npz",
            ),
            (
                "2025-09-21 head 5-min averages",
                DATA_DIR / "head_2025-09-21_5min_avg_windows_rank64.npz",
            ),
            (
                "2026-04-27 16:42 median, acqs 0-7",
                DATA_DIR / "kenny_2026-04-27_1642_first8_median.npz",
            ),
        ]
        datasets = []
        missing = []
        for name, path in specs:
            if path.exists():
                datasets.append(load_dataset(name, path))
            else:
                missing.append(path.name)
        if not datasets:
            raise FileNotFoundError(f"No Doppler CNR GUI datasets found under {DATA_DIR}")
        if missing:
            print("Skipping missing GUI datasets:", ", ".join(missing), flush=True)
        return datasets

    def _dataset_changed(self, idx: int) -> None:
        self.current_dataset = self.datasets[idx]
        self.current_bin_acqs = max(1, self.current_dataset.selection_window_acqs)
        max_window = max(0, self._n_selection_windows() - 1)
        self.current_window_start = min(self.current_window_start, max_window)
        self.current_window_end = min(self.current_window_end, max_window)
        if self.current_window_end < self.current_window_start:
            self.current_window_end = self.current_window_start
        self.current_plane = min(self.current_plane, self.current_dataset.n_planes - 1)
        current_metric = self.current_metric if self.current_metric in self.current_dataset.arrays else next(iter(self.current_dataset.arrays))
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        self.metric_combo.addItems(list(self.current_dataset.arrays.keys()))
        self.metric_combo.setCurrentText(current_metric)
        self.metric_combo.blockSignals(False)
        self.current_metric = current_metric
        self._refresh_all()

    def _compare_toggled(self, enabled: bool) -> None:
        self.compare_enabled = bool(enabled)
        self.compare_combo.setEnabled(self.compare_enabled)
        self.compare_plot.setVisible(self.compare_enabled)
        self._refresh_all()

    def _compare_dataset_changed(self, idx: int) -> None:
        self.compare_dataset = self.datasets[idx]
        self._refresh_all()

    def _bin_acqs_changed(self, value: int) -> None:
        if not self.current_dataset.is_per_acq:
            self.current_bin_acqs = max(1, self.current_dataset.selection_window_acqs)
            self._refresh_all()
            return
        old_bin = max(1, self.current_bin_acqs)
        sample_count = self._dataset_sample_count()
        old_start = min(self.current_window_start * old_bin, max(0, sample_count - 1))
        old_stop = min((self.current_window_end + 1) * old_bin, sample_count) - 1
        self.current_bin_acqs = max(1, value)
        self.current_window_start = max(0, old_start // self.current_bin_acqs)
        self.current_window_end = max(self.current_window_start, old_stop // self.current_bin_acqs)
        self._refresh_all()

    def _window_start_changed(self, value: int) -> None:
        self.current_window_start = max(0, value)
        if self.current_window_end < self.current_window_start:
            self.window_end_spin.blockSignals(True)
            self.window_end_spin.setValue(self.current_window_start)
            self.window_end_spin.blockSignals(False)
            self.current_window_end = self.current_window_start
        self._refresh_all()

    def _window_end_changed(self, value: int) -> None:
        self.current_window_end = max(0, value)
        if self.current_window_start > self.current_window_end:
            self.window_start_spin.blockSignals(True)
            self.window_start_spin.setValue(self.current_window_end)
            self.window_start_spin.blockSignals(False)
            self.current_window_start = self.current_window_end
        self._refresh_all()

    def _metric_changed(self, metric: str) -> None:
        self.current_metric = metric
        self._refresh_all()

    def _plane_changed(self, plane: int) -> None:
        self.current_plane = plane
        self._refresh_all()

    def _auto_toggled(self, enabled: bool) -> None:
        self.auto_button.setText("Auto: click target" if enabled else "Auto: click image")
        if enabled:
            self.segment_button.setChecked(False)
            self.circle_button.setChecked(False)

    def _segment_toggled(self, enabled: bool) -> None:
        if enabled:
            self.segment_button.setText("CV segment: click signal")
        else:
            self.segment_button.setText("CV segment: signal then bg rect")
            self._clear_pending_segment()
        if enabled:
            self.auto_button.setChecked(False)
            self.circle_button.setChecked(False)

    def _circle_toggled(self, enabled: bool) -> None:
        if enabled:
            self.circle_button.setText("Circle: drag signal")
        else:
            self.circle_button.setText("Circle: signal then bg rect")
            self._clear_pending_segment()
        if enabled:
            self.auto_button.setChecked(False)
            self.segment_button.setChecked(False)

    def _cnr_noise_mode(self) -> str:
        return "background" if self.cnr_noise_combo.currentIndex() == 1 else "both"

    def _cnr_noise_label(self) -> str:
        if self._cnr_noise_mode() == "background":
            return "background std"
        return "sqrt(var(signal) + var(background))"

    def _cnr_noise_changed(self) -> None:
        if self.manual_text is not None and self.signal_roi is not None and self.background_roi is not None:
            self._manual_measure()
            return
        current = self._current_plane_selections()
        if current:
            selection = current[-1]
            self._report_measurement("Selection", selection["signal_mask"], selection["background_mask"])
        self._redraw_selection_overlays()

    def _metric_image(self) -> tuple[np.ndarray, str, tuple[float, float], str]:
        return self._image_for_metric(self.current_metric)

    def _image_for_metric(self, metric: str) -> tuple[np.ndarray, str, tuple[float, float], str]:
        return self._image_for_dataset_metric(self.current_dataset, metric, self.current_plane)

    def _image_for_dataset_metric(
        self, dataset: Dataset, metric: str, plane: int
    ) -> tuple[np.ndarray, str, tuple[float, float], str]:
        raw = self._raw_metric_plane_for(dataset, metric, plane)
        if velocity_alpha_metric(metric):
            velocity = raw.astype(np.float32)
            finite = velocity[np.isfinite(velocity)]
            lim = float(np.percentile(np.abs(finite), float(self.cd_abs_percentile.value()))) if finite.size else 1.0
            alpha_mode = "r2" if metric == VELOCITY_ALPHA_R2_METRIC else "r2_geomean"
            rgba = self._signed_rgba_with_quality_alpha(dataset, plane, velocity, lim, alpha_mode=alpha_mode)
            alpha_label = "R2" if alpha_mode == "r2" else "R2 x geomean"
            return rgba, "rgba", (-lim, lim), f"multi-lag phase velocity [mm/s], alpha = normalized {alpha_label}"
        if bland_altman_metric(metric):
            return raw.astype(np.float32), "gray", (0.0, 1.0), "Bland-Altman: multi-lag phase velocity vs color Doppler [mm/s]"
        if metric == "Power Doppler":
            if self.pd_scale_combo.currentIndex() == 0:
                img = 10.0 * np.log10(np.maximum(raw, 1e-12))
                vmax = float(np.nanmax(img))
                dyn_range = float(self.pd_db_range.value())
                return img.astype(np.float32), "magma", (vmax - dyn_range, vmax), f"PD dB, {dyn_range:g} dB range"
            img = raw.astype(np.float32)
            finite = img[np.isfinite(img)]
            pct = float(self.pd_linear_percentile.value())
            vmax = float(np.percentile(finite, pct)) if finite.size else 1.0
            if vmax <= 0:
                vmax = float(np.nanmax(img)) if finite.size else 1.0
            return img, "magma", (0.0, vmax), f"PD linear, vmax p{pct:g}"
        if metric == "Raw Pre-BF IQ Magnitude":
            img = 20.0 * np.log10(np.maximum(raw, 1e-12))
            vmax = float(np.nanmax(img))
            dyn_range = float(self.pd_db_range.value())
            return img.astype(np.float32), "gray", (vmax - dyn_range, vmax), f"raw IQ magnitude dB, {dyn_range:g} dB range"
        if raw_iq_trace_metric(metric):
            trace = raw.reshape(-1).astype(np.float32)
            return trace[:, None], "gray", (float(np.nanmin(trace)), float(np.nanmax(trace))), "raw IQ magnitude trace"
        if signed_doppler_variant(metric) and metric != "Dower Coppler":
            img = raw.astype(np.float32)
            finite = img[np.isfinite(img)]
            pct = float(self.cd_abs_percentile.value())
            lim = float(np.percentile(np.abs(finite), pct)) if finite.size else 1.0
            if metric == "Agree Geomean":
                label = "normalized blend"
            elif metric in {"Phase Velocity", "Phase Velocity x R2", "Dower-sign PVxR2"}:
                label = "m/s" if metric != "Phase Velocity x R2" else "m/s x R2"
            else:
                label = "a.u."
            return img, "seismic", (-lim, lim), f"{metric} {label}, symmetric p{pct:g}(abs)"
        if metric in {"Phase Fit R2", "Huber Fit Quality"}:
            img = raw.astype(np.float32)
            finite = img[np.isfinite(img)]
            vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
            return img, "viridis", (0.0, max(vmax, 1e-6)), metric
        if metric == "Geomean |Rk|":
            img = raw.astype(np.float32)
            finite = img[np.isfinite(img)]
            pct = float(self.pd_linear_percentile.value())
            vmax = float(np.percentile(finite, pct)) if finite.size else 1.0
            return img, "magma", (0.0, max(vmax, 1e-12)), f"{metric}, vmax p{pct:g}"
        img = raw.astype(np.float32)
        finite = img[np.isfinite(img)]
        pct = float(self.dc_abs_percentile.value())
        lim = float(np.percentile(np.abs(finite), pct)) if finite.size else 1.0
        return img, "seismic", (-lim, lim), f"DC raw, symmetric p{pct:g}(abs)"

    def _measurement_image_for_metric(self, metric: str) -> tuple[np.ndarray, str]:
        if bland_altman_metric(metric):
            return np.zeros_like(self.current_image, dtype=np.float32), metric
        raw = self._raw_metric_plane_for(self.current_dataset, metric, self.current_plane)
        if velocity_alpha_metric(metric):
            return np.abs(raw).astype(np.float32), f"{metric} magnitude for CNR"
        if metric == "Power Doppler":
            return (10.0 * np.log10(np.maximum(raw, 1e-12))).astype(np.float32), "PD full-range dB"
        if metric == "Raw Pre-BF IQ Magnitude":
            return (20.0 * np.log10(np.maximum(raw, 1e-12))).astype(np.float32), "raw IQ magnitude dB"
        if signed_doppler_variant(metric):
            return np.abs(raw).astype(np.float32), f"{metric} magnitude for CNR"
        return raw.astype(np.float32), metric

    def _signed_raw_image_for_metric(self, metric: str) -> np.ndarray:
        return self._raw_metric_plane_for(self.current_dataset, metric, self.current_plane).astype(np.float32)

    def _signed_rgba_with_quality_alpha(
        self,
        dataset: Dataset,
        plane: int,
        velocity_mm_s: np.ndarray,
        lim: float,
        alpha_mode: str = "r2_geomean",
    ) -> np.ndarray:
        norm = np.clip((velocity_mm_s.astype(np.float32) + lim) / max(2.0 * lim, 1e-12), 0.0, 1.0)
        lut = self._lookup_table("seismic")
        idx = np.clip((norm * (len(lut) - 1)).round().astype(np.int32), 0, len(lut) - 1)
        rgb = lut[idx].astype(np.float32)

        if "Phase Fit R2" in dataset.arrays:
            r2 = self._raw_metric_plane_for(dataset, "Phase Fit R2", plane).astype(np.float32)
            quality = np.clip(r2, 0.0, 1.0)
            if alpha_mode == "r2_geomean" and "Geomean |Rk|" in dataset.arrays:
                geo = self._raw_metric_plane_for(dataset, "Geomean |Rk|", plane).astype(np.float32)
                quality = quality * np.maximum(geo, 0.0)
            finite = quality[np.isfinite(quality) & (quality > 0)]
            scale = float(np.percentile(finite, 99.0)) if finite.size else 1.0
            alpha = np.clip(quality / max(scale, 1e-12), 0.0, 1.0)
        else:
            alpha = np.ones_like(velocity_mm_s, dtype=np.float32)

        rgba = np.dstack([rgb, (255.0 * alpha).astype(np.float32)]).astype(np.ubyte)
        rgba[~np.isfinite(velocity_mm_s)] = 0
        return rgba

    def _dataset_sample_count(self) -> int:
        return self._dataset_sample_count_for(self.current_dataset)

    def _dataset_sample_count_for(self, dataset: Dataset) -> int:
        first = next(iter(dataset.arrays.values()))
        return int(first.shape[0]) if first.ndim == 4 else 1

    def _selection_bin_acqs(self) -> int:
        return self._selection_bin_acqs_for(self.current_dataset)

    def _selection_bin_acqs_for(self, dataset: Dataset) -> int:
        if dataset.is_per_acq:
            return max(1, self.current_bin_acqs)
        return max(1, dataset.selection_window_acqs)

    def _n_selection_windows(self) -> int:
        return self._n_selection_windows_for(self.current_dataset)

    def _n_selection_windows_for(self, dataset: Dataset) -> int:
        first = next(iter(dataset.arrays.values()))
        if first.ndim != 4:
            return 1
        if dataset.is_per_acq:
            return int(np.ceil(first.shape[0] / self._selection_bin_acqs_for(dataset)))
        return int(first.shape[0])

    def _selected_sample_bounds(self) -> tuple[int, int]:
        return self._selected_sample_bounds_for(self.current_dataset)

    def _selected_sample_bounds_for(self, dataset: Dataset) -> tuple[int, int]:
        sample_count = self._dataset_sample_count_for(dataset)
        if sample_count <= 1:
            return 0, 1
        bin_acqs = self._selection_bin_acqs_for(dataset)
        start = min(self.current_window_start * bin_acqs, sample_count - 1)
        stop = min((self.current_window_end + 1) * bin_acqs, sample_count)
        return int(start), int(max(start + 1, stop))

    def _raw_metric_plane(self, metric: str) -> np.ndarray:
        return self._raw_metric_plane_for(self.current_dataset, metric, self.current_plane)

    def _raw_metric_plane_for(self, dataset: Dataset, metric: str, plane: int) -> np.ndarray:
        arr = dataset.arrays[metric]
        if arr.ndim == 4:
            plane = min(max(0, int(plane)), arr.shape[1] - 1)
            if dataset.is_per_acq:
                start, stop = self._selected_sample_bounds_for(dataset)
            else:
                start = min(self.current_window_start, arr.shape[0] - 1)
                stop = min(self.current_window_end + 1, arr.shape[0])
            return np.median(arr[start:stop, plane], axis=0)
        plane = min(max(0, int(plane)), arr.shape[0] - 1)
        return arr[plane]

    def _extent_cm(self, shape: tuple[int, int] | None = None) -> tuple[float, float, float, float]:
        """Return image extent as (xmin, xmax, zmax, zmin) in cm."""
        return self._extent_cm_for(self.current_dataset, self.current_metric, shape)

    def _extent_cm_for(
        self, dataset: Dataset, metric: str, shape: tuple[int, int] | None = None
    ) -> tuple[float, float, float, float]:
        """Return image extent as (xmin, xmax, zmax, zmin) in cm."""
        if shape is None:
            shape = self.current_image.shape
        h, w = [int(v) for v in shape]
        if metric == "Raw Pre-BF IQ Magnitude":
            return 0.0, float(w), float(h), 0.0
        extent = dataset.extent
        if extent is not None and len(extent) == 4:
            return tuple(float(v) for v in extent)

        name = f"{dataset.name} {dataset.path.name}"
        if "2025-09-21" in name and (h, w) in {(58, 147), (116, 294)}:
            return -2.7456, 2.7456, 3.7896, 2.1
        if "2025-09-21" in name and (h, w) == (88, 266):
            spacing_cm = (1600.0 / 2_750_000.0 / 4.0) * 100.0
            half_x_cm = spacing_cm * (w - 1) / 2.0
            zmin_cm = 2.1
            return -half_x_cm, half_x_cm, zmin_cm + spacing_cm * (h - 1), zmin_cm
        if (h, w) in {(256, 133), (264, 133)}:
            spacing_cm = 0.0128
            half_x_cm = 1.3728
            zmin_cm = 0.8
            return -half_x_cm, half_x_cm, zmin_cm + spacing_cm * (h - 1), zmin_cm

        return 0.0, float(w), float(h), 0.0

    def _plot_rect(self) -> QtCore.QRectF:
        xmin, xmax, zmax, zmin = self._extent_cm()
        return QtCore.QRectF(xmin, zmin, xmax - xmin, zmax - zmin)

    def _plot_rect_for(self, dataset: Dataset, metric: str, shape: tuple[int, int]) -> QtCore.QRectF:
        xmin, xmax, zmax, zmin = self._extent_cm_for(dataset, metric, shape)
        return QtCore.QRectF(xmin, zmin, xmax - xmin, zmax - zmin)

    def _px_to_plot(self, col: float, row: float) -> tuple[float, float]:
        h, w = self.current_image.shape
        xmin, xmax, zmax, zmin = self._extent_cm()
        x = xmin + (np.asarray(col, dtype=np.float64) / max(1, w)) * (xmax - xmin)
        z = zmin + (np.asarray(row, dtype=np.float64) / max(1, h)) * (zmax - zmin)
        return x, z

    def _plot_to_px(self, x: float, z: float) -> tuple[float, float]:
        h, w = self.current_image.shape
        xmin, xmax, zmax, zmin = self._extent_cm()
        col = (float(x) - xmin) / max(np.finfo(float).eps, xmax - xmin) * w
        row = (float(z) - zmin) / max(np.finfo(float).eps, zmax - zmin) * h
        return col, row

    def _rect_px_to_plot(self, rect: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = rect
        px0, py0 = self._px_to_plot(x0, y0)
        px1, py1 = self._px_to_plot(x1, y1)
        return px0, py0, px1, py1

    def _plot_size_for_pixels(self, width_px: float, height_px: float) -> tuple[float, float]:
        x0, z0 = self._px_to_plot(0, 0)
        x1, z1 = self._px_to_plot(width_px, height_px)
        return abs(x1 - x0), abs(z1 - z0)

    def _window_label(self) -> str:
        return self._window_label_for(self.current_dataset)

    def _window_label_for(self, dataset: Dataset) -> str:
        n_windows = self._n_selection_windows_for(dataset)
        if dataset.is_per_acq:
            start, stop = self._selected_sample_bounds_for(dataset)
            acq_indices = dataset.acq_indices
            first_acq = int(acq_indices[start]) if acq_indices is not None else start
            last_acq = int(acq_indices[stop - 1]) if acq_indices is not None else stop - 1
            bin_acqs = self._selection_bin_acqs_for(dataset)
            if first_acq == last_acq:
                return f"acq {first_acq:03d} (bin {bin_acqs})"
            return f"acqs {first_acq:03d}-{last_acq:03d} median (bin {bin_acqs})"
        if n_windows <= 1:
            return "single image"
        labels = dataset.window_labels or [f"window {idx}" for idx in range(n_windows)]
        start = min(self.current_window_start, len(labels) - 1)
        end = min(self.current_window_end, len(labels) - 1)
        if start == end:
            return labels[start]
        return f"{labels[start]} to {labels[end]} median"

    def _refresh_all(self) -> None:
        self.bin_acqs_spin.blockSignals(True)
        max_bin = max(1, self._dataset_sample_count())
        self.bin_acqs_spin.setRange(1, max_bin)
        self.bin_acqs_spin.setValue(min(self._selection_bin_acqs(), max_bin))
        self.bin_acqs_spin.setEnabled(self.current_dataset.is_per_acq and max_bin > 1)
        self.current_bin_acqs = self.bin_acqs_spin.value()
        self.bin_acqs_spin.blockSignals(False)

        self.window_start_spin.blockSignals(True)
        self.window_end_spin.blockSignals(True)
        max_window = max(0, self._n_selection_windows() - 1)
        self.window_start_spin.setRange(0, max_window)
        self.window_end_spin.setRange(0, max_window)
        self.window_start_spin.setValue(min(self.current_window_start, max_window))
        self.window_end_spin.setValue(min(max(self.current_window_end, self.window_start_spin.value()), max_window))
        self.window_start_spin.setEnabled(self.current_dataset.n_windows > 1)
        self.window_end_spin.setEnabled(self.current_dataset.n_windows > 1)
        self.current_window_start = self.window_start_spin.value()
        self.current_window_end = self.window_end_spin.value()
        self.window_end_spin.blockSignals(False)
        self.window_start_spin.blockSignals(False)

        self.plane_spin.blockSignals(True)
        self.plane_spin.setMaximum(max(0, self.current_dataset.n_planes - 1))
        self.plane_spin.setValue(self.current_plane)
        self.plane_spin.setEnabled(self.current_dataset.n_planes > 1)
        self.plane_spin.blockSignals(False)

        self._clear_bland_altman_items()
        self.display_image, cmap, levels, label = self._metric_image()
        if velocity_alpha_metric(self.current_metric) or bland_altman_metric(self.current_metric):
            self.current_image = self._raw_metric_plane(self.current_metric).astype(np.float32)
        else:
            self.current_image = self.display_image
        if bland_altman_metric(self.current_metric):
            self.image_item.hide()
            self.trace_item.show()
            text = self._show_bland_altman_plot(self.current_dataset, self.current_plane)
            self.plot.setAspectLocked(False)
            self.hist.setEnabled(False)
            label = text
        elif raw_iq_trace_metric(self.current_metric):
            trace = self.display_image.reshape(-1)
            self.image_item.hide()
            self.trace_item.setData(np.arange(trace.size, dtype=np.float32), trace)
            self.trace_item.show()
            self.plot.setAspectLocked(False)
            self.plot.setLabel("bottom", "fast time sample")
            self.plot.setLabel("left", "mean abs IQ")
            self.plot.enableAutoRange()
            self.hist.setEnabled(False)
        else:
            self.trace_item.hide()
            self.image_item.show()
            if self.display_image.ndim == 3:
                self.hist.setEnabled(False)
                self.image_item.setImage(np.transpose(self.display_image, (1, 0, 2)), autoLevels=False)
            else:
                self.hist.setEnabled(True)
                self.image_item.setImage(self.display_image.T, autoLevels=False)
                self.image_item.setLevels(levels)
                self._set_colormap(cmap)
                self.hist.setLevels(*levels)
            self.image_item.setRect(self._plot_rect())
        if bland_altman_metric(self.current_metric):
            self.plot.setAspectLocked(False)
            self.plot.setMouseEnabled(x=True, y=True)
        elif self.current_metric == "Raw Pre-BF IQ Magnitude":
            self.plot.setAspectLocked(False)
            self.plot.setLabel("bottom", "element number")
            self.plot.setLabel("left", "fast time sample")
        elif not raw_iq_trace_metric(self.current_metric):
            self.plot.setAspectLocked(True)
            self.plot.setLabel("bottom", "x [cm]")
            self.plot.setLabel("left", "z [cm]")
        self.plot.setTitle(
            f"{self.current_dataset.name} | {self._window_label()} | {self.current_metric} | plane {self.current_plane}"
        )
        self.results.setPlainText(
            f"Displayed image: {label}\n"
            f"Selected range: {self._window_label()}\n"
            "Measurements use fixed unclipped arrays: PD full-range dB, CD/DC raw signed.\n"
            f"CNR denominator: {self._cnr_noise_label()}.\n"
            "Auto: click a target. Signal is a circle; background is same-row pixels outside it.\n"
            "CV segment: click signal first, then drag a background rectangle.\n"
            "Circle: drag signal circle first, then drag a background rectangle.\n"
            "All measurements report PD, CD, and DC together.\n"
            "Manual: move/resize signal and background rectangles.\n"
            "gCNR = 1 - histogram overlap."
        )
        self._refresh_comparison(cmap, levels)
        self._remove_current_rois()
        self._clear_pending_segment()
        self._redraw_selection_overlays()

    def _set_colormap(self, cmap: str) -> None:
        if cmap == "rgba":
            return
        if cmap in {"magma", "seismic"}:
            self.hist.gradient.setColorMap(pg.colormap.getFromMatplotlib(cmap))
        else:
            self.hist.gradient.loadPreset(cmap)

    def _clear_bland_altman_items(self) -> None:
        for item in self.ba_items:
            self.plot.removeItem(item)
        self.ba_items.clear()

    def _normal_color_doppler_mm_s(
        self, dataset: Dataset, plane: int
    ) -> tuple[np.ndarray | None, str | None]:
        if "Color Doppler" not in dataset.arrays:
            return None, "normal color Doppler unavailable: dataset has no Color Doppler array"
        cd = self._raw_metric_plane_for(dataset, "Color Doppler", plane).astype(np.float32)
        if "Phase Velocity" in dataset.arrays:
            pv = self._raw_metric_plane_for(dataset, "Phase Velocity", plane).astype(np.float32)
            if pv.shape == cd.shape:
                finite = np.isfinite(pv) & np.isfinite(cd)
                if finite.any():
                    max_diff = float(np.nanmax(np.abs(pv[finite] - cd[finite])))
                    scale = float(np.nanmax(np.abs(pv[finite])))
                    tol = max(1e-8, 1e-5 * scale)
                    if max_diff <= tol:
                        return (
                            None,
                            "Bland-Altman unavailable for this dataset: stored color_doppler "
                            "is identical to phase_velocity, so no independent normal color "
                            "Doppler estimate was saved.",
                        )
        return cd * 1000.0, None

    def _show_bland_altman_message(self, message: str) -> None:
        self.trace_item.setData([], [])
        text = pg.TextItem(message, color="#d4d4d4", anchor=(0.5, 0.5))
        text.setPos(0.5, 0.5)
        text.setZValue(10)
        self.plot.addItem(text)
        self.ba_items.append(text)
        self.plot.setXRange(0.0, 1.0, padding=0.0)
        self.plot.setYRange(0.0, 1.0, padding=0.0)
        self.plot.disableAutoRange()

    def _bland_altman_values(
        self, dataset: Dataset, plane: int
    ) -> tuple[np.ndarray, np.ndarray, str | None]:
        pv = self._raw_metric_plane_for(dataset, VELOCITY_ALPHA_METRIC, plane).astype(np.float32)
        cd, reason = self._normal_color_doppler_mm_s(dataset, plane)
        if cd is None:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), reason
        mask = np.isfinite(pv) & np.isfinite(cd)
        if "Phase Fit R2" in dataset.arrays and "Geomean |Rk|" in dataset.arrays:
            r2 = self._raw_metric_plane_for(dataset, "Phase Fit R2", plane).astype(np.float32)
            geo = self._raw_metric_plane_for(dataset, "Geomean |Rk|", plane).astype(np.float32)
            quality = np.clip(r2, 0.0, 1.0) * np.maximum(geo, 0.0)
            mask &= np.isfinite(quality) & (quality > 0)
        pv = pv[mask]
        cd = cd[mask]
        mean = (pv + cd) / 2.0
        diff = pv - cd
        return mean, diff, None

    def _show_bland_altman_plot(self, dataset: Dataset, plane: int) -> str:
        mean, diff, reason = self._bland_altman_values(dataset, plane)
        if reason is not None:
            self._show_bland_altman_message(reason)
            return reason
        if mean.size == 0:
            self._show_bland_altman_message("Bland-Altman unavailable: no finite phase/color velocity pixels")
            return "Bland-Altman unavailable: no finite phase/color velocity pixels"
        max_points = 30000
        if mean.size > max_points:
            idx = np.linspace(0, mean.size - 1, max_points, dtype=np.int64)
            plot_mean = mean[idx]
            plot_diff = diff[idx]
        else:
            plot_mean = mean
            plot_diff = diff
        self.trace_item.setData(
            plot_mean,
            plot_diff,
            pen=None,
            symbol="o",
            symbolSize=3,
            symbolPen=None,
            symbolBrush=pg.mkBrush(80, 160, 255, 70),
        )
        bias = float(np.nanmean(diff))
        sd = float(np.nanstd(diff))
        loa_low = bias - 1.96 * sd
        loa_high = bias + 1.96 * sd
        for y, color, label in [
            (bias, "#f59e0b", "bias"),
            (loa_low, "#ef4444", "-1.96 SD"),
            (loa_high, "#ef4444", "+1.96 SD"),
        ]:
            line = pg.InfiniteLine(pos=y, angle=0, pen=pg.mkPen(color, width=1.5, style=QtCore.Qt.DashLine), label=label)
            self.plot.addItem(line)
            self.ba_items.append(line)
        self.plot.setLabel("bottom", "mean velocity [mm/s]")
        self.plot.setLabel("left", "phase velocity - color Doppler [mm/s]")
        finite_mean = mean[np.isfinite(mean)]
        finite_diff = diff[np.isfinite(diff)]
        if finite_mean.size:
            x0, x1 = np.percentile(finite_mean, [0.5, 99.5])
            xpad = max(float(x1 - x0) * 0.08, 1e-6)
            self.plot.setXRange(float(x0 - xpad), float(x1 + xpad), padding=0.0)
        if finite_diff.size:
            y0, y1 = np.percentile(finite_diff, [0.5, 99.5])
            y0 = min(float(y0), loa_low)
            y1 = max(float(y1), loa_high)
            ypad = max(float(y1 - y0) * 0.12, 1e-6)
            self.plot.setYRange(float(y0 - ypad), float(y1 + ypad), padding=0.0)
        self.plot.disableAutoRange()
        return (
            f"Bland-Altman phase velocity vs color Doppler [mm/s]\n"
            f"Pixels: {mean.size}; bias: {bias:.3g} mm/s; SD: {sd:.3g} mm/s; "
            f"limits: [{loa_low:.3g}, {loa_high:.3g}] mm/s"
        )

    def _lookup_table(self, cmap: str) -> np.ndarray:
        color_map = None
        if cmap in {"magma", "seismic", "viridis", "gray"}:
            try:
                color_map = pg.colormap.getFromMatplotlib(cmap)
            except Exception:
                color_map = None
        if color_map is None:
            try:
                color_map = pg.colormap.get(cmap)
            except Exception:
                gray = np.linspace(0, 255, 256, dtype=np.ubyte)
                return np.column_stack([gray, gray, gray])
        return color_map.getLookupTable(0.0, 1.0, 256)

    def _refresh_comparison(self, cmap: str, levels: tuple[float, float]) -> None:
        if not self.compare_enabled:
            self.compare_plot.hide()
            return

        self.compare_plot.show()
        metric = self.current_metric
        if bland_altman_metric(metric):
            self.compare_image_item.hide()
            self.compare_trace_item.hide()
            self.compare_plot.setTitle("Bland-Altman comparison is shown in the main panel")
            return
        dataset = self.compare_dataset
        if metric not in dataset.arrays:
            self.compare_image_item.hide()
            self.compare_trace_item.hide()
            self.compare_plot.setTitle(f"{dataset.name} | missing metric: {metric}")
            return

        plane = min(self.current_plane, dataset.n_planes - 1)
        img, _, _, _ = self._image_for_dataset_metric(dataset, metric, plane)
        if raw_iq_trace_metric(metric):
            trace = img.reshape(-1)
            self.compare_image_item.hide()
            self.compare_trace_item.setData(np.arange(trace.size, dtype=np.float32), trace)
            self.compare_trace_item.show()
            self.compare_plot.setAspectLocked(False)
            self.compare_plot.setLabel("bottom", "fast time sample")
            self.compare_plot.setLabel("left", "mean abs IQ")
            self.compare_plot.enableAutoRange()
        else:
            self.compare_trace_item.hide()
            self.compare_image_item.show()
            if img.ndim == 3:
                self.compare_image_item.setImage(np.transpose(img, (1, 0, 2)), autoLevels=False)
            else:
                self.compare_image_item.setImage(img.T, autoLevels=False)
                self.compare_image_item.setLevels(levels)
                self.compare_image_item.setLookupTable(self._lookup_table(cmap))
            self.compare_image_item.setRect(self._plot_rect_for(dataset, metric, img.shape[:2]))
            if metric == "Raw Pre-BF IQ Magnitude":
                self.compare_plot.setAspectLocked(False)
                self.compare_plot.setLabel("bottom", "element number")
                self.compare_plot.setLabel("left", "fast time sample")
            else:
                self.compare_plot.setAspectLocked(True)
                self.compare_plot.setLabel("bottom", "x [cm]")
                self.compare_plot.setLabel("left", "z [cm]")

        self.compare_plot.setTitle(
            f"{dataset.name} | {self._window_label_for(dataset)} | {metric} | plane {plane}"
        )

    def _remove_latest_selection(self) -> None:
        current = self._current_plane_selections()
        if not current:
            self.results.setPlainText("No current-plane selections to remove.")
            return
        self._remove_selection(current[-1]["id"])

    def _remove_selection_id(self) -> None:
        self._remove_selection(int(self.remove_id_spin.value()))

    def _remove_selection(self, selection_id: int) -> None:
        before = len(self.selections)
        self.selections = [s for s in self.selections if s["id"] != selection_id]
        if self.manual_selection_id == selection_id:
            self.manual_selection_id = None
            self._remove_current_rois()
        self._redraw_selection_overlays()
        removed = before - len(self.selections)
        self.results.setPlainText(
            f"Removed selection #{selection_id}." if removed else f"Selection #{selection_id} not found."
        )

    def _clear_measurements(self, clear_text: bool = True) -> None:
        self._clear_selection_items()
        self.selections.clear()
        self.manual_selection_id = None
        self._remove_current_rois()
        self._clear_pending_segment()
        if clear_text:
            self.results.clear()

    def _remove_current_rois(self) -> None:
        for item in self.auto_items:
            self.plot.removeItem(item)
        self.auto_items.clear()
        for roi in (self.signal_roi, self.background_roi):
            if roi is not None:
                self.plot.removeItem(roi)
        self.signal_roi = None
        self.background_roi = None
        if self.manual_text is not None:
            self.plot.removeItem(self.manual_text)
            self.manual_text = None

    def _clear_selection_items(self) -> None:
        for item in self.selection_items:
            self.plot.removeItem(item)
        self.selection_items.clear()

    def _clear_pending_segment_items(self) -> None:
        for item in self.pending_segment_items:
            self.plot.removeItem(item)
        self.pending_segment_items.clear()

    def _clear_pending_segment(self) -> None:
        self._clear_pending_segment_items()
        self.pending_segment = None
        self.circle_drag_start = None
        self.background_drag_start = None
        if self.circle_drag_item is not None:
            self.plot.removeItem(self.circle_drag_item)
            self.circle_drag_item = None
        if self.background_drag_item is not None:
            self.plot.removeItem(self.background_drag_item)
            self.background_drag_item = None

    def eventFilter(self, obj, event) -> bool:
        if obj is self.plot.viewport() and self.circle_button.isChecked() and self.pending_segment is None:
            if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                point = self._image_point_from_viewport_event(event)
                if point is None:
                    return False
                self.circle_drag_start = point
                self._update_circle_drag_preview(point, point)
                return True
            if event.type() == QtCore.QEvent.MouseMove and self.circle_drag_start is not None:
                point = self._image_point_from_viewport_event(event)
                if point is not None:
                    self._update_circle_drag_preview(self.circle_drag_start, point)
                return True
            if event.type() == QtCore.QEvent.MouseButtonRelease and self.circle_drag_start is not None:
                point = self._image_point_from_viewport_event(event)
                if point is not None:
                    self._finalize_circle_signal(self.circle_drag_start, point)
                return True

        bg_drag_active = self.segment_button.isChecked() or self.circle_button.isChecked()
        if obj is self.plot.viewport() and bg_drag_active and self.pending_segment is not None:
            if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                point = self._image_point_from_viewport_event(event)
                if point is None:
                    return False
                self.background_drag_start = point
                self._update_background_drag_preview(point, point)
                return True
            if event.type() == QtCore.QEvent.MouseMove and self.background_drag_start is not None:
                point = self._image_point_from_viewport_event(event)
                if point is not None:
                    self._update_background_drag_preview(self.background_drag_start, point)
                return True
            if event.type() == QtCore.QEvent.MouseButtonRelease and self.background_drag_start is not None:
                point = self._image_point_from_viewport_event(event)
                if point is not None:
                    self._finalize_segment_with_background_rect(self.background_drag_start, point)
                return True
        return super().eventFilter(obj, event)

    def _image_point_from_viewport_event(self, event) -> tuple[int, int] | None:
        scene_pos = self.plot.mapToScene(event.pos())
        view_pos = self.plot.plotItem.vb.mapSceneToView(scene_pos)
        col_f, row_f = self._plot_to_px(view_pos.x(), view_pos.y())
        col = int(round(col_f))
        row = int(round(row_f))
        h, w = self.current_image.shape
        if row < 0 or row >= h or col < 0 or col >= w:
            return None
        return row, col

    def _mouse_clicked(self, event) -> None:
        interactive_mode = (
            self.auto_button.isChecked()
            or self.segment_button.isChecked()
        )
        if not interactive_mode:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self.segment_button.isChecked() and self.pending_segment is not None:
            return
        view_pos = self.plot.plotItem.vb.mapSceneToView(event.scenePos())
        col_f, row_f = self._plot_to_px(view_pos.x(), view_pos.y())
        col = int(round(col_f))
        row = int(round(row_f))
        if self.segment_button.isChecked():
            self._segment_measure(row, col)
        else:
            self._auto_measure(row, col)
            self.auto_button.setChecked(False)

    def _auto_measure(self, row: int, col: int) -> None:
        img = self.current_image
        h, w = img.shape
        if row < 0 or row >= h or col < 0 or col >= w:
            return
        self._clear_auto_items()
        radius = int(self.auto_radius.value())
        rect = (
            max(0, col - radius),
            max(0, row - radius),
            min(w, col + radius),
            min(h, row + radius),
        )
        signal_mask = self._ellipse_mask_from_rect(rect)
        background_mask, row0, row1 = self._same_row_background(signal_mask)
        selection = {
            "id": self.next_selection_id,
            "dataset_index": self.dataset_combo.currentIndex(),
            "window_start": self.current_window_start,
            "window_end": self.current_window_end,
            "bin_acqs": self._selection_bin_acqs(),
            "plane": self.current_plane,
            "kind": "auto",
            "signal_mask": signal_mask,
            "background_mask": background_mask,
            "geometry": {
                "rect": tuple(int(v) for v in rect),
                "row0": int(row0),
                "row1": int(row1),
                "width": int(w),
            },
        }
        self.next_selection_id += 1
        self.selections.append(selection)
        self._report_measurement("Auto", signal_mask, background_mask)
        self._redraw_selection_overlays()

    def _ellipse_mask_from_rect(self, rect: tuple[int, int, int, int]) -> np.ndarray:
        h, w = self.current_image.shape
        x0, y0, x1, y1 = [int(v) for v in rect]
        x0, x1 = np.clip([x0, x1], 0, w)
        y0, y1 = np.clip([y0, y1], 0, h)
        mask = np.zeros((h, w), dtype=bool)
        if x1 <= x0 or y1 <= y0:
            return mask
        yy, xx = np.ogrid[:h, :w]
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        rx = max((x1 - x0) / 2.0, 0.5)
        ry = max((y1 - y0) / 2.0, 0.5)
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        return mask

    def _same_row_background(self, signal_mask: np.ndarray) -> tuple[np.ndarray, int, int]:
        rows = np.where(signal_mask.any(axis=1))[0]
        background_mask = np.zeros_like(signal_mask, dtype=bool)
        if rows.size == 0:
            return background_mask, 0, 0
        background_mask[rows, :] = True
        background_mask &= ~signal_mask
        return background_mask, int(rows.min()), int(rows.max()) + 1

    def _update_auto_selection_from_roi(self, selection_id: int, roi: pg.ROI) -> None:
        rect = self._roi_rect(roi)
        signal_mask = self._ellipse_mask_from_rect(rect)
        background_mask, row0, row1 = self._same_row_background(signal_mask)
        for selection in self.selections:
            if selection["id"] == selection_id:
                selection["signal_mask"] = signal_mask.copy()
                selection["background_mask"] = background_mask.copy()
                selection["geometry"] = {
                    "rect": rect,
                    "row0": row0,
                    "row1": row1,
                    "width": int(self.current_image.shape[1]),
                }
                self._report_measurement("Auto", signal_mask, background_mask)
                break
        self._redraw_selection_overlays()

    def _circle_rect_from_points(self, start: tuple[int, int], end: tuple[int, int]) -> tuple[tuple[int, int, int, int], int]:
        h, w = self.current_image.shape
        row0, col0 = start
        row1, col1 = end
        radius = int(np.ceil(np.hypot(row1 - row0, col1 - col0)))
        radius = max(1, radius)
        rect = (
            max(0, col0 - radius),
            max(0, row0 - radius),
            min(w, col0 + radius + 1),
            min(h, row0 + radius + 1),
        )
        return rect, radius

    def _update_circle_drag_preview(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        rect, _ = self._circle_rect_from_points(start, end)
        if self.circle_drag_item is not None:
            self.plot.removeItem(self.circle_drag_item)
        px0, py0, px1, py1 = self._rect_px_to_plot(rect)
        self.circle_drag_item = pg.EllipseROI(
            [px0, py0],
            [max(1e-6, px1 - px0), max(1e-6, py1 - py0)],
            pen=pg.mkPen("c", width=2),
            movable=False,
        )
        self.circle_drag_item.setZValue(8)
        self.plot.addItem(self.circle_drag_item)

    def _finalize_circle_signal(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        if self.pending_segment is not None:
            return
        rect, radius = self._circle_rect_from_points(start, end)
        row, col = start
        mask = self._ellipse_mask_from_rect(rect)
        if int(mask.sum()) < 3:
            self.results.setPlainText("Circular signal ROI is too small.")
            self.circle_drag_start = None
            return
        if self.circle_drag_item is not None:
            self.plot.removeItem(self.circle_drag_item)
            self.circle_drag_item = None
        self.pending_segment = {
            "kind": "circle",
            "usable_mask": mask,
            "raw_mask": mask.copy(),
            "centroid_row": float(row),
            "centroid_col": float(col),
            "signal_rect": tuple(int(v) for v in rect),
            "signal_radius_px": int(radius),
        }
        self._draw_pending_segment(mask)
        self.circle_button.setText("Circle: drag bg rect")
        self.circle_drag_start = None
        self.results.setPlainText("Circular signal ROI selected. Drag a background rectangle to finish the measurement.")

    def _segment_measure(self, row: int, col: int) -> None:
        img = self.current_image
        h, w = img.shape
        if row < 0 or row >= h or col < 0 or col >= w:
            return
        if self.pending_segment is not None:
            return
        mask = self._local_cv_segment(row, col)
        if mask is None or int(mask.sum()) < 3:
            self.results.setPlainText("CV segmentation did not find a usable region.")
            return

        erosion = int(self.segment_erosion.value())
        usable_mask = mask.copy()
        if erosion > 0:
            eroded = ndimage.binary_erosion(mask, iterations=erosion)
            if eroded.any():
                usable_mask = eroded

        ys, xs = np.where(mask)
        if self.pending_segment is None:
            self.pending_segment = {
                "raw_mask": mask.copy(),
                "usable_mask": usable_mask.copy(),
                "centroid_row": float(np.mean(ys)),
                "centroid_col": float(np.mean(xs)),
            }
            self._draw_pending_segment(mask)
            self.segment_button.setText("CV segment: drag bg rect")
            self.results.setPlainText("Signal segmented. Drag a background rectangle to finish the measurement.")
            return

    def _background_rect_from_points(self, start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int, int, int]:
        h, w = self.current_image.shape
        row0, col0 = start
        row1, col1 = end
        x0 = int(np.clip(min(col0, col1), 0, w))
        x1 = int(np.clip(max(col0, col1) + 1, 0, w))
        y0 = int(np.clip(min(row0, row1), 0, h))
        y1 = int(np.clip(max(row0, row1) + 1, 0, h))
        return x0, y0, x1, y1

    def _update_background_drag_preview(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        rect = self._background_rect_from_points(start, end)
        x0, y0, x1, y1 = rect
        if self.background_drag_item is not None:
            self.plot.removeItem(self.background_drag_item)
        px0, py0, px1, py1 = self._rect_px_to_plot(rect)
        self.background_drag_item = pg.RectROI(
            [px0, py0],
            [max(1e-6, px1 - px0), max(1e-6, py1 - py0)],
            pen=pg.mkPen((255, 127, 80), width=2),
            movable=False,
        )
        self.background_drag_item.setZValue(8)
        self.plot.addItem(self.background_drag_item)

    def _finalize_segment_with_background_rect(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        if self.pending_segment is None:
            return
        rect = self._background_rect_from_points(start, end)
        background_mask = self._rect_mask(rect)
        signal_mask = self.pending_segment["usable_mask"]
        signal_raw_mask = self.pending_segment["raw_mask"]
        if np.logical_and(signal_mask, background_mask).any():
            background_mask = np.logical_and(background_mask, ~signal_mask)
        if int(background_mask.sum()) < 3:
            self.results.setPlainText("Background rectangle is too small or overlaps the signal; drag a larger background rectangle.")
            self.background_drag_start = None
            return
        ys, xs = np.where(background_mask)

        mode_kind = str(self.pending_segment.get("kind", "segmented"))
        mode_label = "Circle" if mode_kind == "circle" else "CV segment"
        geometry = {
            "signal_mask": signal_raw_mask,
            "background_rect": rect,
            "centroid_row": self.pending_segment["centroid_row"],
            "centroid_col": self.pending_segment["centroid_col"],
            "background_centroid_row": float(np.mean(ys)),
            "background_centroid_col": float(np.mean(xs)),
        }
        if mode_kind == "circle":
            geometry["signal_rect"] = self.pending_segment.get("signal_rect")
            geometry["signal_radius_px"] = self.pending_segment.get("signal_radius_px")
        selection = {
            "id": self.next_selection_id,
            "dataset_index": self.dataset_combo.currentIndex(),
            "window_start": self.current_window_start,
            "window_end": self.current_window_end,
            "bin_acqs": self._selection_bin_acqs(),
            "plane": self.current_plane,
            "kind": mode_kind,
            "signal_mask": signal_mask,
            "background_mask": background_mask,
            "geometry": geometry,
        }
        self.next_selection_id += 1
        self.selections.append(selection)
        self._clear_pending_segment()
        self._report_measurement(mode_label, signal_mask, background_mask)
        self._redraw_selection_overlays()
        if mode_kind == "circle":
            self.circle_button.setChecked(False)
        else:
            self.segment_button.setChecked(False)

    def _rect_mask(self, rect: tuple[int, int, int, int]) -> np.ndarray:
        h, w = self.current_image.shape
        x0, y0, x1, y1 = rect
        x0, x1 = np.clip([x0, x1], 0, w)
        y0, y1 = np.clip([y0, y1], 0, h)
        mask = np.zeros((h, w), dtype=bool)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
        return mask

    def _draw_pending_segment(self, mask: np.ndarray) -> None:
        self._clear_pending_segment_items()
        boundary = ndimage.binary_dilation(mask) & ~mask
        ys, xs = np.where(boundary)
        plot_x, plot_y = self._px_to_plot(xs + 0.5, ys + 0.5)
        boundary_item = pg.ScatterPlotItem(
            x=plot_x,
            y=plot_y,
            size=2,
            pen=pg.mkPen("c"),
            brush=pg.mkBrush("c"),
        )
        self.plot.addItem(boundary_item)
        self.pending_segment_items.append(boundary_item)

    def _local_cv_segment(self, row: int, col: int) -> np.ndarray | None:
        raw_img = np.asarray(self.current_image, dtype=np.float32)
        sign_mask = None
        if signed_doppler_variant(self.current_metric):
            seed_raw = float(raw_img[row, col])
            if seed_raw > 0:
                sign_mask = raw_img > 0
            elif seed_raw < 0:
                sign_mask = raw_img < 0
            else:
                self.results.setPlainText(
                    "Clicked seed is near zero on a signed image; choose a positive or negative region."
                )
                return None
        img = raw_img
        if signed_doppler_variant(self.current_metric):
            img = np.abs(img)
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            return None
        lo = float(np.percentile(finite, 1))
        hi = float(np.percentile(finite, 99))
        if hi <= lo:
            hi = float(np.max(finite))
            lo = float(np.min(finite))
        if hi <= lo:
            return None
        norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)

        h, w = norm.shape
        radius = int(self.segment_radius.value())
        y0 = max(0, row - radius)
        y1 = min(h, row + radius + 1)
        x0 = max(0, col - radius)
        x1 = min(w, col + radius + 1)
        local = norm[y0:y1, x0:x1]
        sy = row - y0
        sx = col - x0
        seed = float(local[sy, sx])
        tol = float(self.segment_tolerance.value())

        try:
            from skimage.segmentation import flood

            local_mask = flood(local, (sy, sx), tolerance=tol)
        except Exception:
            local_mask = np.abs(local - seed) <= tol

        yy, xx = np.ogrid[y0:y1, x0:x1]
        local_mask &= (yy - row) ** 2 + (xx - col) ** 2 <= radius**2
        if sign_mask is not None:
            local_sign = sign_mask[y0:y1, x0:x1]
            local_mask &= local_sign

        max_area = max(10, int(0.45 * local.size))
        if local_mask.sum() < 5 or local_mask.sum() > max_area:
            threshold = max(float(np.percentile(local, 75)), seed - tol)
            candidate = local >= threshold
            if sign_mask is not None:
                candidate &= sign_mask[y0:y1, x0:x1]
            labels, n_labels = ndimage.label(candidate)
            label = labels[sy, sx]
            if label == 0 and n_labels > 0:
                # If the exact seed is below threshold, take the closest labeled pixel.
                ys, xs = np.where(labels > 0)
                nearest = np.argmin((ys - sy) ** 2 + (xs - sx) ** 2)
                label = labels[ys[nearest], xs[nearest]]
            local_mask = labels == label if label else candidate

        local_mask = ndimage.binary_fill_holes(local_mask)
        local_mask = ndimage.binary_opening(local_mask, iterations=1)
        local_mask = ndimage.binary_closing(local_mask, iterations=1)
        if sign_mask is not None:
            local_mask &= sign_mask[y0:y1, x0:x1]
        if not local_mask[sy, sx]:
            local_mask[sy, sx] = True

        mask = np.zeros_like(norm, dtype=bool)
        mask[y0:y1, x0:x1] = local_mask
        return mask

    def _clear_auto_items(self) -> None:
        for item in self.auto_items:
            self.plot.removeItem(item)
        self.auto_items.clear()

    def _create_manual_rois(self) -> None:
        self._remove_current_rois()
        self.manual_selection_id = None
        h, w = self.current_image.shape
        sig_w, sig_h = max(4, w // 8), max(4, h // 8)
        bg_w, bg_h = sig_w, sig_h
        sig_x, sig_y = self._px_to_plot(w * 0.35, h * 0.35)
        bg_x, bg_y = self._px_to_plot(w * 0.58, h * 0.35)
        sig_wc, sig_hc = self._plot_size_for_pixels(sig_w, sig_h)
        bg_wc, bg_hc = self._plot_size_for_pixels(bg_w, bg_h)
        self.signal_roi = pg.RectROI([sig_x, sig_y], [sig_wc, sig_hc], pen=pg.mkPen("c", width=2))
        self.background_roi = pg.RectROI([bg_x, bg_y], [bg_wc, bg_hc], pen=pg.mkPen((255, 127, 80), width=2))
        for roi in (self.signal_roi, self.background_roi):
            roi.addScaleHandle([1, 1], [0, 0])
            roi.addScaleHandle([0, 0], [1, 1])
            self.plot.addItem(roi)
            roi.sigRegionChanged.connect(self._manual_measure)
        self.manual_text = pg.TextItem(color="w", fill=(0, 0, 0, 180), anchor=(0, 1))
        self.plot.addItem(self.manual_text)
        self._manual_measure()

    def _manual_measure(self) -> None:
        if self.signal_roi is None or self.background_roi is None:
            return
        signal_mask = self._roi_mask(self.signal_roi)
        background_mask = self._roi_mask(self.background_roi)
        self._upsert_manual_selection(signal_mask, background_mask)
        self._report_measurement("Manual", signal_mask, background_mask)
        text = self._manual_selection_label()
        if self.manual_text is not None:
            pos = self.signal_roi.pos()
            size = self.signal_roi.size()
            self.manual_text.setText(text)
            dx, _ = self._plot_size_for_pixels(4, 0)
            self.manual_text.setPos(pos.x() + size.x() + dx, pos.y())
        self._redraw_selection_overlays(include_manual=False)

    def _manual_selection_label(self) -> str:
        if self.manual_selection_id is None:
            return ""
        for selection in self.selections:
            if selection["id"] == self.manual_selection_id:
                return self._selection_label(selection)
        return ""

    def _upsert_manual_selection(self, signal_mask: np.ndarray, background_mask: np.ndarray) -> None:
        if self.signal_roi is None or self.background_roi is None:
            return
        signal_rect = self._roi_rect(self.signal_roi)
        background_rect = self._roi_rect(self.background_roi)
        if self.manual_selection_id is None:
            self.manual_selection_id = self.next_selection_id
            self.next_selection_id += 1
            self.selections.append(
                {
                    "id": self.manual_selection_id,
                    "dataset_index": self.dataset_combo.currentIndex(),
                    "window_start": self.current_window_start,
                    "window_end": self.current_window_end,
                    "bin_acqs": self._selection_bin_acqs(),
                    "plane": self.current_plane,
                    "kind": "manual",
                    "signal_mask": signal_mask.copy(),
                    "background_mask": background_mask.copy(),
                    "geometry": {
                        "signal_rect": signal_rect,
                        "background_rect": background_rect,
                    },
                }
            )
            return
        for selection in self.selections:
            if selection["id"] == self.manual_selection_id:
                selection["signal_mask"] = signal_mask.copy()
                selection["background_mask"] = background_mask.copy()
                selection["geometry"] = {
                    "signal_rect": signal_rect,
                    "background_rect": background_rect,
                }
                selection["dataset_index"] = self.dataset_combo.currentIndex()
                selection["window_start"] = self.current_window_start
                selection["window_end"] = self.current_window_end
                selection["bin_acqs"] = self._selection_bin_acqs()
                selection["plane"] = self.current_plane
                break

    def _roi_mask(self, roi: pg.RectROI) -> np.ndarray:
        x0, y0, x1, y1 = self._roi_rect(roi)
        h, w = self.current_image.shape
        mask = np.zeros((h, w), dtype=bool)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
        return mask

    def _roi_rect(self, roi: pg.RectROI) -> tuple[int, int, int, int]:
        h, w = self.current_image.shape
        pos = roi.pos()
        size = roi.size()
        col0, row0 = self._plot_to_px(pos.x(), pos.y())
        col1, row1 = self._plot_to_px(pos.x() + size.x(), pos.y() + size.y())
        x0 = int(np.floor(min(col0, col1)))
        x1 = int(np.ceil(max(col0, col1)))
        y0 = int(np.floor(min(row0, row1)))
        y1 = int(np.ceil(max(row0, row1)))
        x0, x1 = np.clip([x0, x1], 0, w)
        y0, y1 = np.clip([y0, y1], 0, h)
        return int(x0), int(y0), int(x1), int(y1)

    def _report_measurement(self, mode: str, signal_mask: np.ndarray, background_mask: np.ndarray) -> str:
        lines = [
            f"{mode} measurement",
            f"Dataset: {self.current_dataset.name}",
            f"Selected range: {self._window_label()}",
            f"Plane: {self.current_plane}",
            f"Signal pixels: {int(signal_mask.sum())}, background pixels: {int(background_mask.sum())}",
            f"CNR denominator: {self._cnr_noise_label()}",
            "",
        ]
        current_short = ""
        for metric in self.current_dataset.arrays:
            img, label = self._measurement_image_for_metric(metric)
            signal = img[signal_mask]
            background = img[background_mask]
            cnr, cnr_db, cnr_numerator, cnr_denominator = compute_cnr_components(
                signal, background, self._cnr_noise_mode()
            )
            gcnr = compute_gcnr(signal, background)
            contrast_db = compute_contrast(signal, background)
            signal_mean = float(np.nanmean(signal)) if signal.size else np.nan
            background_mean = float(np.nanmean(background)) if background.size else np.nan
            signal_std = float(np.nanstd(signal)) if signal.size else np.nan
            background_std = float(np.nanstd(background)) if background.size else np.nan
            signed_note = ""
            if signed_doppler_variant(metric):
                raw_img = self._signed_raw_image_for_metric(metric)
                raw_signal = raw_img[signal_mask]
                raw_background = raw_img[background_mask]
                raw_signal_mean = float(np.nanmean(raw_signal)) if raw_signal.size else np.nan
                raw_background_mean = float(np.nanmean(raw_background)) if raw_background.size else np.nan
                signed_note = (
                    f"\n  Signed raw signal/background mean: "
                    f"{raw_signal_mean:.4g} / {raw_background_mean:.4g}"
                )
            metric_text = (
                f"{metric} ({label})\n"
                f"  CNR: {cnr:.4g}\n"
                f"  CNR dB (20log10): {cnr_db:.3g}\n"
                f"  |mean diff| / denominator: {cnr_numerator:.4g} / {cnr_denominator:.4g}\n"
                f"  gCNR: {gcnr:.4f}\n"
                f"  Signal mean/std: {signal_mean:.4g} / {signal_std:.4g}\n"
                f"  Background mean/std: {background_mean:.4g} / {background_std:.4g}"
                f"{signed_note}\n"
                f"  Contrast dB: {contrast_db:.3g}"
            )
            lines.append(metric_text)
            lines.append("")
            if metric == self.current_metric:
                current_short = f"{metric}\nCNR {cnr_db:.3g} dB"
        text = "\n".join(lines).rstrip()
        self.results.setPlainText(text)
        return current_short

    def _metric_short_names(self):
        labels = {
            "Power Doppler": "PD",
            "Raw Pre-BF IQ Magnitude": "IQ",
            "Color Doppler": "CD",
            "Dower Coppler": "DC",
            "Phase Velocity": "PV",
            VELOCITY_ALPHA_METRIC: "PVmmAlpha",
            VELOCITY_ALPHA_R2_METRIC: "PVmmA-R2",
            "Phase Velocity x R2": "PVxR2",
            "Phase Velocity x Geomean |Rk| x R2": "PVGeoR2",
            "Phase Fit R2": "R2",
            "Huber Fit Quality": "HuberQ",
            "Signed Scale": "Scale",
            "Signed Scale x HuberQ": "ScaleQ",
            "Geomean |Rk|": "GeoR",
            "Signed Geomean |Rk|": "SGeoR",
            "Signed Geomean |Rk| x HuberQ": "SGeoRQ",
            "Signed Geomean |Rk| x HuberQ x R2": "SGeoRQR2",
            "Dower x HuberQ": "DCHQ",
            "Dower x PhaseR2": "DCxR2",
            "Phase-sign Dower": "PVsignDC",
            "Dower-sign PVxR2": "DCsignPV",
            "Sign-agree Dower": "AgreeDC",
            "Agree Geomean": "AgreeGM",
        }
        return [
            (labels.get(metric, metric[:6]), metric)
            for metric in self.current_dataset.arrays
            if not bland_altman_metric(metric)
        ]

    def _selection_label(self, selection: dict) -> str:
        lines = [f"#{selection['id']}"]
        signal_mask = selection["signal_mask"]
        background_mask = selection["background_mask"]
        for short, metric in self._metric_short_names():
            img, _ = self._measurement_image_for_metric(metric)
            signal = img[signal_mask]
            background = img[background_mask]
            cnr, cnr_db = compute_cnr(signal, background, self._cnr_noise_mode())
            gcnr = compute_gcnr(signal, background)
            lines.append(f"{short} CNR {cnr:.2g} ({cnr_db:.2g} dB), gCNR {gcnr:.2f}")
        return "\n".join(lines)

    def _current_plane_selections(self) -> list[dict]:
        dataset_index = self.dataset_combo.currentIndex()
        current_bin = self._selection_bin_acqs()
        default_bin = max(1, self.current_dataset.selection_window_acqs if self.current_dataset.is_per_acq else 1)
        return [
            selection
            for selection in self.selections
            if selection["dataset_index"] == dataset_index
            and selection.get("window_start", 0) == self.current_window_start
            and selection.get("window_end", 0) == self.current_window_end
            and selection.get("bin_acqs", default_bin) == current_bin
            and selection["plane"] == self.current_plane
            and selection["signal_mask"].shape == self.current_image.shape
        ]

    def _redraw_selection_overlays(self, include_manual: bool = True) -> None:
        self._clear_selection_items()
        for selection in self._current_plane_selections():
            if not include_manual and selection["kind"] == "manual" and selection["id"] == self.manual_selection_id:
                continue
            self._draw_selection(selection)

    def _draw_selection(self, selection: dict) -> None:
        geom = selection["geometry"]
        if selection["kind"] == "auto":
            x0, y0, x1, y1 = geom["rect"]
            px0, py0, px1, py1 = self._rect_px_to_plot((x0, y0, x1, y1))
            signal_item = pg.EllipseROI(
                [px0, py0],
                [max(1e-6, px1 - px0), max(1e-6, py1 - py0)],
                pen=pg.mkPen("c", width=2),
                movable=True,
            )
            signal_item.addScaleHandle([1, 1], [0, 0])
            signal_item.addScaleHandle([0, 0], [1, 1])
            signal_item.sigRegionChangeFinished.connect(
                lambda *args, sid=selection["id"], roi=signal_item: self._update_auto_selection_from_roi(sid, roi)
            )
            self.plot.addItem(signal_item)
            self.selection_items.append(signal_item)
            bx0, by0, bx1, by1 = self._rect_px_to_plot((0, geom["row0"], geom["width"], geom["row1"]))
            bg_item = pg.RectROI(
                [bx0, by0],
                [max(1e-6, bx1 - bx0), max(1e-6, by1 - by0)],
                pen=pg.mkPen((255, 127, 80), width=1.5),
                movable=False,
            )
            bg_item.setZValue(5)
            self.plot.addItem(bg_item)
            self.selection_items.append(bg_item)
            label_x, label_y = self._px_to_plot(x1 + 4, (y0 + y1) / 2.0)
        elif selection["kind"] == "manual":
            sx0, sy0, sx1, sy1 = geom["signal_rect"]
            bx0, by0, bx1, by1 = geom["background_rect"]
            psx0, psy0, psx1, psy1 = self._rect_px_to_plot((sx0, sy0, sx1, sy1))
            pbx0, pby0, pbx1, pby1 = self._rect_px_to_plot((bx0, by0, bx1, by1))
            signal_item = pg.RectROI(
                [psx0, psy0],
                [max(1e-6, psx1 - psx0), max(1e-6, psy1 - psy0)],
                pen=pg.mkPen("c", width=2),
                movable=False,
            )
            bg_item = pg.RectROI(
                [pbx0, pby0],
                [max(1e-6, pbx1 - pbx0), max(1e-6, pby1 - pby0)],
                pen=pg.mkPen((255, 127, 80), width=2),
                movable=False,
            )
            self.plot.addItem(signal_item)
            self.plot.addItem(bg_item)
            self.selection_items.extend([signal_item, bg_item])
            label_x, label_y = self._px_to_plot(sx1 + 4, sy0)
        else:
            signal_boundary = ndimage.binary_dilation(geom["signal_mask"]) & ~geom["signal_mask"]
            sig_ys, sig_xs = np.where(signal_boundary)
            bx0, by0, bx1, by1 = geom["background_rect"]
            plot_x, plot_y = self._px_to_plot(sig_xs + 0.5, sig_ys + 0.5)
            pbx0, pby0, pbx1, pby1 = self._rect_px_to_plot((bx0, by0, bx1, by1))
            signal_item = pg.ScatterPlotItem(
                x=plot_x,
                y=plot_y,
                size=2,
                pen=pg.mkPen("c"),
                brush=pg.mkBrush("c"),
            )
            bg_item = pg.RectROI(
                [pbx0, pby0],
                [max(1e-6, pbx1 - pbx0), max(1e-6, pby1 - pby0)],
                pen=pg.mkPen((255, 127, 80), width=2),
                movable=False,
            )
            self.plot.addItem(signal_item)
            self.plot.addItem(bg_item)
            self.selection_items.extend([signal_item, bg_item])
            label_x, label_y = self._px_to_plot(geom["centroid_col"] + 6, geom["centroid_row"])
            bg_label = pg.TextItem(
                f"#{selection['id']} bg",
                color=(255, 255, 255),
                anchor=(0, 0.5),
                fill=(0, 0, 0, 150),
            )
            bg_x, bg_y = self._px_to_plot(geom["background_centroid_col"] + 6, geom["background_centroid_row"])
            bg_label.setPos(bg_x, bg_y)
            bg_label.setZValue(10)
            self.plot.addItem(bg_label)
            self.selection_items.append(bg_label)

        text = pg.TextItem(
            self._selection_label(selection),
            color=(255, 255, 255),
            anchor=(0, 0.5),
            fill=(0, 0, 0, 190),
        )
        text.setPos(label_x, label_y)
        text.setZValue(10)
        self.plot.addItem(text)
        self.selection_items.append(text)

    def _save_pdf(self) -> None:
        selections = self._current_plane_selections()
        if self.manual_text is not None and self.signal_roi is not None and self.background_roi is not None:
            self._manual_measure()
            selections = self._current_plane_selections()
        if not selections:
            self.results.setPlainText("No current-plane measurements to save.")
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = DATA_DIR / f"cnr_measurement_{timestamp}.pdf"

        metrics = list(self.current_dataset.arrays.keys())
        fig, axes = plt.subplots(1, len(metrics), figsize=(5.7 * len(metrics), 6), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, metric in zip(axes, metrics):
            img, cmap, levels, label = self._image_for_metric(metric)
            if bland_altman_metric(metric):
                mean, diff, reason = self._bland_altman_values(self.current_dataset, self.current_plane)
                if mean.size:
                    ax.scatter(mean, diff, s=2, alpha=0.25)
                    bias = float(np.nanmean(diff))
                    sd = float(np.nanstd(diff))
                    ax.axhline(bias, color="#f59e0b", linestyle="--", linewidth=1.0)
                    ax.axhline(bias - 1.96 * sd, color="#ef4444", linestyle="--", linewidth=1.0)
                    ax.axhline(bias + 1.96 * sd, color="#ef4444", linestyle="--", linewidth=1.0)
                elif reason is not None:
                    ax.text(0.5, 0.5, reason, transform=ax.transAxes, ha="center", va="center", wrap=True)
                ax.set_xlabel("mean velocity [mm/s]")
                ax.set_ylabel("phase velocity - color Doppler [mm/s]")
            else:
                xmin, xmax, zmax, zmin = self._extent_cm(img.shape[:2])
                if img.ndim == 3:
                    ax.imshow(img, origin="lower", aspect="auto", extent=[xmin, xmax, zmin, zmax])
                else:
                    ax.imshow(
                        img,
                        cmap=cmap,
                        vmin=levels[0],
                        vmax=levels[1],
                        origin="lower",
                        aspect="auto",
                        extent=[xmin, xmax, zmin, zmax],
                    )
                ax.set_xlabel("x [cm]")
                ax.set_ylabel("z [cm]")
                for selection in selections:
                    self._draw_selection_matplotlib(ax, selection)
            ax.set_title(f"{metric}\n{label}")
        fig.suptitle(f"{self.current_dataset.name} | {self._window_label()} | plane {self.current_plane}")
        fig.savefig(path)
        plt.close(fig)
        shape_json, shape_npz = self._save_measurement_shapes(path, selections)
        self.results.appendPlainText(
            f"\nSaved PDF: {path}"
            f"\nSaved region shapes JSON: {shape_json}"
            f"\nSaved region masks NPZ: {shape_npz}"
        )

    def _mask_bounds(self, mask: np.ndarray) -> dict:
        rows, cols = np.where(mask)
        if rows.size == 0:
            return {"row_min": None, "row_max": None, "col_min": None, "col_max": None}
        return {
            "row_min": int(rows.min()),
            "row_max": int(rows.max()),
            "col_min": int(cols.min()),
            "col_max": int(cols.max()),
        }

    def _jsonable_geometry(self, value):
        if isinstance(value, dict):
            return {str(k): self._jsonable_geometry(v) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            if value.dtype == bool:
                rows, cols = np.where(value)
                return {
                    "shape": [int(v) for v in value.shape],
                    "true_pixels_rc": [[int(r), int(c)] for r, c in zip(rows.tolist(), cols.tolist())],
                }
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [self._jsonable_geometry(v) for v in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    def _selection_shape_record(self, selection: dict, index: int) -> dict:
        signal_mask = np.asarray(selection["signal_mask"], dtype=bool)
        background_mask = np.asarray(selection["background_mask"], dtype=bool)
        sig_rows, sig_cols = np.where(signal_mask)
        bg_rows, bg_cols = np.where(background_mask)
        return {
            "index": int(index),
            "id": int(selection["id"]),
            "kind": str(selection["kind"]),
            "dataset": self.current_dataset.name,
            "dataset_path": str(self.current_dataset.path),
            "window_start": int(selection.get("window_start", self.current_window_start)),
            "window_end": int(selection.get("window_end", self.current_window_end)),
            "bin_acqs": int(selection.get("bin_acqs", self._selection_bin_acqs())),
            "plane": int(selection["plane"]),
            "image_shape_rc": [int(v) for v in signal_mask.shape],
            "signal_pixels": int(signal_mask.sum()),
            "background_pixels": int(background_mask.sum()),
            "signal_bounds": self._mask_bounds(signal_mask),
            "background_bounds": self._mask_bounds(background_mask),
            "signal_pixels_rc": [[int(r), int(c)] for r, c in zip(sig_rows.tolist(), sig_cols.tolist())],
            "background_pixels_rc": [[int(r), int(c)] for r, c in zip(bg_rows.tolist(), bg_cols.tolist())],
            "geometry": self._jsonable_geometry(selection.get("geometry", {})),
        }

    def _save_measurement_shapes(self, pdf_path: Path, selections: list[dict]) -> tuple[Path, Path]:
        json_path = pdf_path.with_suffix(".regions.json")
        npz_path = pdf_path.with_suffix(".regions.npz")
        records = [self._selection_shape_record(selection, i) for i, selection in enumerate(selections)]
        payload = {
            "dataset": self.current_dataset.name,
            "dataset_path": str(self.current_dataset.path),
            "selected_range": self._window_label(),
            "plane": int(self.current_plane),
            "metric_displayed": self.current_metric,
            "image_shape_rc": [int(v) for v in self.current_image.shape],
            "selections": records,
        }
        import json

        json_path.write_text(json.dumps(payload, indent=2))
        arrays = {}
        for i, selection in enumerate(selections):
            arrays[f"selection_{i:02d}_signal_mask"] = np.asarray(selection["signal_mask"], dtype=bool)
            arrays[f"selection_{i:02d}_background_mask"] = np.asarray(selection["background_mask"], dtype=bool)
        np.savez_compressed(npz_path, **arrays)
        return json_path, npz_path

    def _draw_selection_matplotlib(self, ax, selection: dict) -> None:
        from matplotlib.patches import Rectangle

        geom = selection["geometry"]
        if selection["kind"] == "auto":
            from matplotlib.patches import Ellipse

            x0, y0, x1, y1 = geom["rect"]
            px0, py0, px1, py1 = self._rect_px_to_plot((x0, y0, x1, y1))
            ax.add_patch(
                Ellipse(
                    ((px0 + px1) / 2.0, (py0 + py1) / 2.0),
                    px1 - px0,
                    py1 - py0,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.5,
                )
            )
            bx0, by0, bx1, by1 = self._rect_px_to_plot((0, geom["row0"], geom["width"], geom["row1"]))
            ax.add_patch(
                Rectangle(
                    (bx0, by0),
                    bx1 - bx0,
                    by1 - by0,
                    fill=False,
                    edgecolor="coral",
                    linewidth=1.2,
                )
            )
            label_x, label_y = self._px_to_plot(x1 + 4, (y0 + y1) / 2.0)
        elif selection["kind"] == "manual":
            sx0, sy0, sx1, sy1 = geom["signal_rect"]
            bx0, by0, bx1, by1 = geom["background_rect"]
            psx0, psy0, psx1, psy1 = self._rect_px_to_plot((sx0, sy0, sx1, sy1))
            pbx0, pby0, pbx1, pby1 = self._rect_px_to_plot((bx0, by0, bx1, by1))
            ax.add_patch(Rectangle((psx0, psy0), psx1 - psx0, psy1 - psy0, fill=False, edgecolor="cyan", linewidth=1.5))
            ax.add_patch(Rectangle((pbx0, pby0), pbx1 - pbx0, pby1 - pby0, fill=False, edgecolor="coral", linewidth=1.5))
            label_x, label_y = self._px_to_plot(sx1 + 4, sy0)
        else:
            xmin, xmax, zmax, zmin = self._extent_cm(geom["signal_mask"].shape)
            ax.contour(
                geom["signal_mask"].astype(float),
                levels=[0.5],
                colors=["cyan"],
                linewidths=1.3,
                origin="lower",
                extent=[xmin, xmax, zmin, zmax],
            )
            bx0, by0, bx1, by1 = geom["background_rect"]
            pbx0, pby0, pbx1, pby1 = self._rect_px_to_plot((bx0, by0, bx1, by1))
            ax.add_patch(Rectangle((pbx0, pby0), pbx1 - pbx0, pby1 - pby0, fill=False, edgecolor="coral", linewidth=1.5))
            label_x, label_y = self._px_to_plot(geom["centroid_col"] + 6, geom["centroid_row"])
            bg_x, bg_y = self._px_to_plot(geom["background_centroid_col"] + 6, geom["background_centroid_row"])
            ax.text(
                bg_x,
                bg_y,
                f"#{selection['id']} bg",
                color="white",
                fontsize=7,
                va="center",
                bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none", "pad": 2},
            )
        ax.text(
            label_x,
            label_y,
            self._selection_label(selection),
            color="white",
            fontsize=7,
            va="center",
            bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 2},
        )


def main() -> int:
    app = QtWidgets.QApplication([])
    pg.setConfigOptions(imageAxisOrder="col-major", antialias=True)
    viewer = CnrViewer()
    viewer.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
