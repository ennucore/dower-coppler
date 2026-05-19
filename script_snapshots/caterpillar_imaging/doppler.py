from __future__ import annotations
import torch
from typing import Literal, Optional
import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter


_COLOR_DOPPLER_TYPE_ALIASES = {
    "dower coppler": "phase_velocity_geomean_r2",
}


def canonical_color_doppler_type(cd_type: str) -> str:
    key = str(cd_type).strip()
    return _COLOR_DOPPLER_TYPE_ALIASES.get(key.lower(), key)


def normalize_compound_frames(compound_images: torch.Tensor) -> torch.Tensor:
    return compound_images / torch.from_numpy(
        gaussian_filter(compound_images.abs().cpu(), sigma=5.0)
    ).to(compound_images.device)


# svd clutter filtering
@torch.no_grad()
def svd_filter_fast(
    compound_images: torch.Tensor,
    low_cutoff: float = 0.1,
    high_cutoff: float = 1.0,
) -> torch.Tensor:
    """
    Apply SVD filter to compound images. Uses a trick to reduce the matrix size by computing the spectrum of A^T A instead of A.
    But it's not as numerically stable as the full SVD method, since the condition number of A^T A is the square of the condition number of A.

    Args:
        compound_images: Compound images (frames, rows, cols) or (frames, elev_planes, rows, cols)
        low_cutoff: Remove the first low_cutoff singular values (0-1)
        high_cutoff: Keep the last (1-high_cutoff) fraction of singular values (0-1)

    Returns:
        Filtered compound images (frames, rows, cols) or (frames, elev_planes, rows, cols)
    """
    # MPS complex matmul and linalg.eigh are silently wrong for large complex
    # tensors. Route through CPU and move result back.
    src_device = compound_images.device
    if src_device.type == "mps":
        compound_images = compound_images.cpu()
    X = compound_images.reshape(compound_images.shape[0], -1)
    C = X @ X.conj().T
    evals, U = torch.linalg.eigh(C)
    U = U[:, torch.argsort(evals, descending=True)]
    n = C.shape[0]
    a = round(low_cutoff * n)
    b = round((1 - high_cutoff) * n)
    if a + b >= n:
        raise ValueError(
            "You're removing all the components. Adjust the cutoff parameters."
        )
    Uc = U[:, a : n - b]
    result = (Uc @ Uc.conj().T @ X).reshape_as(compound_images)
    return result.to(src_device)


@torch.no_grad()
def svd_filter_full(
    compound_images: torch.Tensor,
    low_cutoff: float = 0.1,
    high_cutoff: float = 1.0,
) -> torch.Tensor:
    """
    This method has better numerical stability than the fast method.
    """
    src_device = compound_images.device
    if src_device.type == "mps":
        compound_images = compound_images.cpu()

    X = (
        compound_images.permute(1, 2, 0)
        if compound_images.ndim == 3
        else compound_images.permute(1, 2, 3, 0)
    )
    if compound_images.ndim == 3:
        rows, cols, frames = X.shape
        elev_planes = 1
    else:
        elev_planes, rows, cols, frames = X.shape
    Xf = X.reshape(rows * cols * elev_planes, frames)
    U, S, Vh = torch.linalg.svd(Xf, full_matrices=False)
    n_low = round(low_cutoff * frames)
    n_high_remove = max(0, min(frames, round((1.0 - float(high_cutoff)) * frames)))
    if n_low + n_high_remove >= frames:
        raise ValueError(
            "You're removing all the components. Adjust the cutoff parameters."
        )
    tissue = (U[:, :n_low] * S[:n_low].unsqueeze(0)) @ Vh[:n_low, :] if n_low > 0 else 0
    filt = Xf - tissue
    if n_high_remove > 0:
        start = frames - n_high_remove
        filt = filt - (U[:, start:] * S[start:].unsqueeze(0)) @ Vh[start:, :]
    if compound_images.ndim == 3:
        result = filt.reshape(rows, cols, frames).permute(2, 0, 1)
    else:
        result = filt.reshape(elev_planes, rows, cols, frames).permute(3, 0, 1, 2)
    return result.to(src_device)


@torch.no_grad()
def power_doppler(
    compound_images: torch.Tensor,
    low_cutoff: float = 0.1,
    high_cutoff: float = 1.0,
    mean_subtract: bool = True,
    first_frame_subtract: bool = False,
    method: Literal["fast", "full"] = "fast",
    separate_3d_svd: bool = True,
    skip_first_frames: int = 5,
    use_percentile: float | None = None,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Calculate power doppler from compound images. Use the SVD method.

    Args:
        compound_images: Compound images (frames, rows, cols) or (frames, elev_planes, rows, cols)
        runtime_metadata: Runtime metadata
        cutoff_n: Number of singular values to keep
        skip_first_frames: Number of frames to skip at the beginning
        method: Method to use for SVD filtering
        separate_3d_svd: Whether to separate the SVD filtering by elev_planes
        mean_subtract: Whether to subtract the mean of the compound images
        first_frame_subtract: Whether to subtract the first frame of the compound images

    Returns:
        Power doppler image (rows, cols) or (elev_planes, rows, cols)
    """
    if first_frame_subtract:
        compound_images = compound_images[1:] - compound_images[0]
    if mean_subtract:
        compound_images = compound_images - compound_images.mean(dim=0, keepdim=True)
    if normalize:
        compound_images = normalize_compound_frames(compound_images)

    if separate_3d_svd and compound_images.ndim == 4 and compound_images.shape[1] > 1:
        # Do it separately by elevational planes for true 3D stacks:
        # (frames, elev_planes, rows, cols). A 2D stack is (frames, rows, cols)
        # and must be filtered as one image, not row-by-row.
        separated_by_elev_planes = [
            power_doppler(
                p,
                low_cutoff,
                high_cutoff,
                mean_subtract,
                first_frame_subtract,
                method,
                separate_3d_svd=False,
            )[1][:, 0]
            for p in compound_images.split(1, dim=1)
        ]
        combined = torch.stack(separated_by_elev_planes, dim=1)
        return combined.abs().square().sum(dim=0), combined

    if method == "fast":
        filtered = svd_filter_fast(compound_images, low_cutoff, high_cutoff)
    elif method == "full":
        filtered = svd_filter_full(compound_images, low_cutoff, high_cutoff)
    else:
        raise ValueError(f"Invalid method: {method}")
    abs_sq = filtered[skip_first_frames:].abs().square()
    return abs_sq.sum(dim=0), filtered
    # return (
    #     abs_sq.quantile(dim=0, q=use_percentile)
    #     if use_percentile is not None
    #     else abs_sq.sum(dim=0)
    # ), filtered


def _velocity_to_frequency(
    velocity: float, tx_freq: float, sound_speed: float = 1540.0
) -> float:
    """Convert velocity to Doppler cutoff frequency in Hz."""
    return 2 * velocity * tx_freq / sound_speed


def _scipy_highpass_filter(
    sig: torch.Tensor, sample_rate: float, cutoff_freq: float, order: int = 4
) -> torch.Tensor:
    """Apply Butterworth high-pass filter using scipy.signal."""
    # Design the filter
    nyquist = sample_rate / 2
    normal_cutoff = cutoff_freq / nyquist
    sos = signal.butter(order, normal_cutoff, btype="high", output="sos")

    # Convert to numpy for filtering
    sig_np = sig.cpu().numpy() if isinstance(sig, torch.Tensor) else sig

    # Apply filter (scipy.signal.sosfilt handles complex arrays directly)
    filtered_np = signal.sosfilt(sos, sig_np, axis=-1)

    # Convert back to torch tensor
    return torch.from_numpy(filtered_np).to(sig.device)


@torch.no_grad()
def high_pass_filter_compound_images(
    compound_images: torch.Tensor,
    frame_rate: float,
    cutoff_frequency: Optional[float] = None,
    cutoff_velocity: Optional[float] = None,
    tx_freq: Optional[float] = None,
    sound_speed: float = 1540.0,
    order: int = 4,
) -> torch.Tensor:
    """
    Apply high-pass filter to compound images along the frame dimension.

    Args:
        compound_images: Compound images (frames, rows, cols) or (frames, elev_planes, rows, cols)
        frame_rate: Frame rate in Hz
        cutoff_frequency: Cutoff frequency in Hz for high-pass filtering
        cutoff_velocity: Optional velocity cutoff in m/s for high-pass filtering (alternative to cutoff_frequency)
        tx_freq: Transmit frequency in Hz (required if cutoff_velocity is used)
        sound_speed: Speed of sound in m/s (default: 1540.0)
        order: Filter order (default: 4)

    Returns:
        High-pass filtered compound images with the same shape as input
    """
    if cutoff_velocity is not None:
        if tx_freq is None:
            raise ValueError("tx_freq must be provided when using cutoff_velocity")
        cutoff_frequency = _velocity_to_frequency(cutoff_velocity, tx_freq, sound_speed)
    elif cutoff_frequency is None:
        raise ValueError("Either cutoff_frequency or cutoff_velocity must be provided")

    # Design the filter
    nyquist = frame_rate / 2
    normal_cutoff = cutoff_frequency / nyquist
    sos = signal.butter(order, normal_cutoff, btype="high", output="sos")

    # Convert to numpy for filtering
    compound_np = (
        compound_images.cpu().numpy()
        if isinstance(compound_images, torch.Tensor)
        else compound_images
    )

    # Reshape to (frames, spatial_dims) for filtering along frame dimension
    original_shape = compound_np.shape
    n_frames = original_shape[0]
    spatial_dims = compound_np.shape[1:]
    compound_reshaped = compound_np.reshape(n_frames, -1)

    # Apply filter along frame dimension (axis=0)
    filtered_np = signal.sosfilt(sos, compound_reshaped, axis=0)

    # Reshape back to original shape
    filtered_np = filtered_np.reshape(original_shape)

    # Convert back to torch tensor
    if isinstance(compound_images, torch.Tensor):
        return torch.from_numpy(filtered_np).to(compound_images.device)
    else:
        return filtered_np


def color_doppler(
    sig: torch.Tensor,
    frame_rate: float,
    high_pass_velocity: Optional[float] = None,
    tx_freq: Optional[float] = None,
    sound_speed: float = 1540.0,
    blur_sigma: Optional[float] = None,
):
    """
    Compute color Doppler velocity image from beamformed signals.

    This function implements the Kasai autocorrelation method for Doppler frequency estimation.
    It computes the lag-1 autocorrelation (r1) of the complex signal and extracts the phase
    to estimate the Doppler frequency, which is then converted to velocity.

    Args:
        beamformed_signals: Complex torch.Tensor of shape (frames, depth, width)
        frame_rate: Frame rate in Hz
        high_pass_velocity: Optional velocity cutoff in m/s for high-pass filtering
        tx_freq: Transmit frequency in Hz (required if high_pass_velocity is used)
        sound_speed: Speed of sound in m/s (default: 1540.0)

    Returns:
        torch.Tensor: Color Doppler velocity image of shape (depth, width) in m/s
    """
    if high_pass_velocity is not None:
        if tx_freq is None:
            raise ValueError("tx_freq must be provided for high-pass filtering")
        fc = _velocity_to_frequency(high_pass_velocity, tx_freq, sound_speed)
        sig = _scipy_highpass_filter(sig, frame_rate, fc)

    # Compute r1 data: lag-1 autocorrelation s_n * conj(s_{n-1})
    r1 = sig[1:] * torch.conj(sig[:-1])

    # Kasai method using complex operations
    # Sum r1 over time dimension to get mean autocorrelation
    r1_mean = r1.mean(dim=0)
    # Compute phase angle of r1_mean
    color_doppler = torch.angle(r1_mean)
    # Convert to velocity: f = angle(r1) / (2π * T) where T = 1/frame_rate
    color_doppler = color_doppler * frame_rate / (2 * np.pi)
    color_doppler = color_doppler * sound_speed / (2 * tx_freq)

    if blur_sigma:
        if isinstance(color_doppler, torch.Tensor):
            color_doppler_np = color_doppler.cpu().numpy()
        else:
            color_doppler_np = color_doppler

        color_doppler_np = gaussian_filter(color_doppler_np, sigma=blur_sigma)

        if isinstance(color_doppler, torch.Tensor):
            color_doppler = torch.from_numpy(color_doppler_np).to(color_doppler.device)
        else:
            color_doppler = color_doppler_np

    return color_doppler


def kasai_autocorr_matrix(
    sig: torch.Tensor,
    frame_rate: float,
    tx_freq: float,
    sound_speed: float = 1540.0,
    max_lag: int | None = None,
) -> torch.Tensor:
    """Multi-lag Kasai estimate from the slow-time autocorrelation matrix.

    For each pixel, the implicit matrix is R[i, j] = s[i] * conj(s[j]).
    Averaging each off-diagonal gives lag-k autocorrelation estimates; combining
    their phase/k terms yields a velocity-like signed map.
    """
    n_frames = sig.shape[0]
    if n_frames < 2:
        return torch.zeros_like(sig[0].real)

    lag_count = min(max_lag or 8, n_frames - 1)
    phases = []
    weights = []
    for lag in range(1, lag_count + 1):
        rk = (sig[lag:] * torch.conj(sig[:-lag])).mean(dim=0)
        phases.append(torch.angle(rk) / lag)
        weights.append(rk.abs())

    phase_stack = torch.stack(phases, dim=0)
    weight_stack = torch.stack(weights, dim=0)
    denom = weight_stack.sum(dim=0).clamp_min(1e-12)
    phase_per_frame = torch.atan2(
        (weight_stack * torch.sin(phase_stack)).sum(dim=0) / denom,
        (weight_stack * torch.cos(phase_stack)).sum(dim=0) / denom,
    )

    frequency_hz = phase_per_frame * frame_rate / (2 * np.pi)
    return frequency_hz * sound_speed / (2 * tx_freq)


# ── Helper ──
def _fftfreq(T, frame_rate, device, dtype=torch.float32):
    return torch.fft.fftfreq(T, d=1.0 / frame_rate, device=device).to(dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 7 proven winners (kept)
# ═══════════════════════════════════════════════════════════════════════════

# Peak Doppler Frequency — dominant velocity component
def peak_frequency(sig, frame_rate):
    T = sig.shape[0]
    P = torch.fft.fft(sig, dim=0).abs() ** 2
    freqs = _fftfreq(T, frame_rate, sig.device)
    idx = P.argmax(dim=0)
    freqs_view = freqs.view(T, *([1] * (sig.ndim - 1))).expand_as(P)
    return freqs_view.gather(0, idx.unsqueeze(0)).squeeze(0)


def _tmas_lags(sig, max_lag=3, lags=None):
    if lags is None:
        max_lag = max(1, int(max_lag))
        lags = range(1, min(max_lag, sig.shape[0] - 1) + 1)
    else:
        lags = [int(k) for k in lags if 0 < int(k) < sig.shape[0]]
    return list(lags)


def _lag_autocorrelations(sig, max_lag=3, lags=None):
    lags = _tmas_lags(sig, max_lag=max_lag, lags=lags)
    if not lags:
        return lags, None
    rk = [((sig[k:] * torch.conj(sig[:-k])).mean(dim=0)) for k in lags]
    return lags, torch.stack(rk, dim=0)


def _unwrap_lag_phases(phases: torch.Tensor) -> torch.Tensor:
    if phases.shape[0] <= 1:
        return phases
    two_pi = 2.0 * np.pi
    diffs = phases[1:] - phases[:-1]
    wrapped = torch.remainder(diffs + np.pi, two_pi) - np.pi
    corrections = torch.cumsum(wrapped - diffs, dim=0)
    return torch.cat((phases[:1], phases[1:] + corrections), dim=0)


def lag_phase_linear_fit(
    sig,
    max_lag=3,
    lags=None,
    weighted: bool = True,
    fit_intercept: bool = False,
):
    """Fit autocorrelation phase versus lag.

    Returns ``(slope, fit_quality)`` where slope is radians/frame.  The default
    fit is constrained through lag 0, phase 0; setting ``fit_intercept`` allows a
    constant phase offset.  ``fit_quality`` is a weighted R^2-like map that is
    useful for rejecting pixels whose lag phases are not close to linear.
    """
    lags, rk = _lag_autocorrelations(sig, max_lag=max_lag, lags=lags)
    if rk is None:
        zeros = torch.zeros_like(sig[0].real)
        return zeros, zeros

    phases = _unwrap_lag_phases(torch.angle(rk))
    weights = rk.abs() if weighted else torch.ones_like(phases)
    lag_values = torch.as_tensor(lags, device=sig.device, dtype=phases.dtype)
    lag_values = lag_values.view(-1, *([1] * (sig.ndim - 1)))
    weight_sum = weights.sum(dim=0)

    if fit_intercept:
        safe_weight_sum = weight_sum.clamp_min(1e-12)
        lag_mean = (weights * lag_values).sum(dim=0) / safe_weight_sum
        phase_mean = (weights * phases).sum(dim=0) / safe_weight_sum
        lag_centered = lag_values - lag_mean
        phase_centered = phases - phase_mean
        numerator = (weights * lag_centered * phase_centered).sum(dim=0)
        denominator = (weights * lag_centered.square()).sum(dim=0).clamp_min(1e-12)
        slope = numerator / denominator
        intercept = phase_mean - slope * lag_mean
        predicted = lag_values * slope + intercept
    else:
        numerator = (weights * lag_values * phases).sum(dim=0)
        denominator = (weights * lag_values.square()).sum(dim=0).clamp_min(1e-12)
        slope = numerator / denominator
        predicted = lag_values * slope

    if len(lags) <= 1:
        quality = torch.ones_like(slope)
    else:
        safe_weight_sum = weight_sum.clamp_min(1e-12)
        phase_mean = (weights * phases).sum(dim=0) / safe_weight_sum
        residual = phases - predicted
        sse = (weights * residual.square()).sum(dim=0)
        sst = (weights * (phases - phase_mean).square()).sum(dim=0)
        quality = (1.0 - sse / sst.clamp_min(1e-12)).clamp(0.0, 1.0)
    quality = torch.where(weight_sum > 1e-12, quality, torch.zeros_like(quality))
    return slope, quality


def weighted_lag_phase_slope(sig, max_lag=3, lags=None, fit_intercept: bool = False):
    """Weighted least-squares slow-time phase slope from multi-lag autocorrelations."""
    slope, _ = lag_phase_linear_fit(
        sig,
        max_lag=max_lag,
        lags=lags,
        weighted=True,
        fit_intercept=fit_intercept,
    )
    return slope


def lag_phase_fit_quality(sig, max_lag=3, lags=None, fit_intercept: bool = False):
    """Weighted fit quality for the lag-phase linearity model."""
    _, quality = lag_phase_linear_fit(
        sig,
        max_lag=max_lag,
        lags=lags,
        weighted=True,
        fit_intercept=fit_intercept,
    )
    return quality


def weighted_lag_ls_doppler(
    sig,
    frame_rate: float,
    tx_freq: float,
    sound_speed: float = 1540.0,
    max_lag=3,
    lags=None,
):
    phase_per_frame = weighted_lag_phase_slope(sig, max_lag=max_lag, lags=lags)
    frequency_hz = phase_per_frame * frame_rate / (2 * np.pi)
    return frequency_hz * sound_speed / (2 * tx_freq)


# TMAS — multi-lag coherence product
def tmas_coherence(sig, lags=None, max_lag=3):
    lags = _tmas_lags(sig, max_lag=max_lag, lags=lags)
    if not lags:
        return torch.zeros_like(sig[0].real)
    rk_abs = []
    for k in lags:
        rk_abs.append((sig[k:] * torch.conj(sig[:-k])).mean(dim=0).abs())
    return torch.stack(rk_abs, dim=0).prod(dim=0)


# Signed power Doppler composite
def signed_power_composite(sig, alpha=1.0):
    T = sig.shape[0]
    P = torch.fft.fft(sig, dim=0).abs() ** 2
    Pp = P[1 : T // 2].sum(dim=0)
    Pn = P[T // 2 + 1 :].sum(dim=0)
    D = (Pp - Pn) / (Pp + Pn).clamp_min(1e-12)
    pd = (sig.abs() ** 2).sum(dim=0)
    return D * torch.log1p(alpha * pd)


# Spectral skewness (third moment)
def spectral_skewness(sig, frame_rate):
    T = sig.shape[0]
    P = torch.fft.fft(sig, dim=0).abs() ** 2
    freqs = _fftfreq(T, frame_rate, sig.device).view(T, *([1] * (sig.ndim - 1)))
    P_sum = P.sum(dim=0).clamp_min(1e-10)
    centroid = (freqs * P).sum(dim=0) / P_sum
    diff = freqs - centroid.unsqueeze(0)
    bw2 = (diff ** 2 * P).sum(dim=0) / P_sum
    bw = torch.sqrt(bw2.clamp_min(1e-20))
    m3 = (diff ** 3 * P).sum(dim=0) / P_sum
    return m3 / (bw ** 3 + 1e-12)


# Directional power ratio
def directional_power_ratio(sig):
    T = sig.shape[0]
    P = torch.fft.fft(sig, dim=0).abs() ** 2
    Pp = P[1 : T // 2].sum(dim=0)
    Pn = P[T // 2 + 1 :].sum(dim=0)
    return (Pp - Pn) / (Pp + Pn).clamp_min(1e-12)


# Circular phase variance
def circular_phase_variance(sig):
    dphi = torch.angle(sig[1:] * torch.conj(sig[:-1]))
    mean_cos = torch.cos(dphi).mean(dim=0)
    mean_sin = torch.sin(dphi).mean(dim=0)
    return 1.0 - torch.sqrt(mean_cos ** 2 + mean_sin ** 2)


# Axial autocorrelation proxy (Loupas-style)
def axial_autocorr_proxy(sig):
    RT = (sig[1:] * torch.conj(sig[:-1])).mean(dim=0)
    if sig.ndim == 3:
        RZ = (sig[:, 1:, :] * torch.conj(sig[:, :-1, :])).mean(dim=0)
        RZ_padded = torch.nn.functional.pad(RZ, (0, 0, 0, 1), value=0)
    elif sig.ndim == 4:
        RZ = (sig[:, :, 1:, :] * torch.conj(sig[:, :, :-1, :])).mean(dim=0)
        RZ_padded = torch.nn.functional.pad(RZ, (0, 0, 0, 1), value=0)
    else:
        return torch.angle(RT)
    angle_RZ_abs = torch.angle(RZ_padded).abs().clamp_min(1e-10)
    return torch.angle(RT) / angle_RZ_abs


# ═══════════════════════════════════════════════════════════════════════════
# 9 additional kept metrics
# ═══════════════════════════════════════════════════════════════════════════

# Velocity-weighted directional power ratio (emphasises fast flow)
def weighted_dir_power(sig, frame_rate):
    """Dir power ratio weighted by |f| — fast flow emphasised."""
    T = sig.shape[0]
    P = torch.fft.fft(sig, dim=0).abs() ** 2
    freqs = _fftfreq(T, frame_rate, sig.device)
    w = freqs.abs().view(T, *([1] * (sig.ndim - 1))).expand_as(P)
    wP = w * P
    Pp = wP[1 : T // 2].sum(dim=0)
    Pn = wP[T // 2 + 1 :].sum(dim=0)
    return (Pp - Pn) / (Pp + Pn).clamp_min(1e-12)


# 3. Hann-windowed peak frequency (reduced spectral leakage)
def hann_peak_freq(sig, frame_rate):
    """Peak frequency with Hann window — less spectral leakage."""
    T = sig.shape[0]
    w = torch.hann_window(T, device=sig.device).view(T, *([1] * (sig.ndim - 1)))
    P = torch.fft.fft(sig * w, dim=0).abs() ** 2
    freqs = _fftfreq(T, frame_rate, sig.device)
    idx = P.argmax(dim=0)
    freqs_view = freqs.view(T, *([1] * (sig.ndim - 1))).expand_as(P)
    return freqs_view.gather(0, idx.unsqueeze(0)).squeeze(0)


# 4. Signed TMAS: TMAS magnitude × sign of directional ratio
def signed_tmas(sig, max_lag=3):
    """TMAS × sign(dir_power_ratio) — denoised signed flow."""
    T = sig.shape[0]
    P = torch.fft.fft(sig, dim=0).abs() ** 2
    Pp = P[1 : T // 2].sum(dim=0)
    Pn = P[T // 2 + 1 :].sum(dim=0)
    D = (Pp - Pn) / (Pp + Pn).clamp_min(1e-12)
    lags = _tmas_lags(sig, max_lag=max_lag)
    if not lags:
        return torch.zeros_like(sig[0].real)
    rk_abs = [((sig[k:] * torch.conj(sig[:-k])).mean(0)).abs() for k in lags]
    tm = torch.stack(rk_abs, 0).prod(0)
    return D * tm


def signed_tmas_wls(sig, max_lag=3):
    """TMAS weighted by a multi-lag autocorrelation WLS phase slope."""
    lags, rk = _lag_autocorrelations(sig, max_lag=max_lag)
    if rk is None:
        return torch.zeros_like(sig[0].real)

    tm = rk.abs().prod(dim=0)
    phase_per_frame = weighted_lag_phase_slope(sig, max_lag=max_lag, lags=lags)
    signed_scale = (phase_per_frame / np.pi).clamp(-1.0, 1.0)
    return signed_scale * tm


def _wls_origin_phase_slope(
    phases: torch.Tensor,
    weights: torch.Tensor,
    lag_values: torch.Tensor,
) -> torch.Tensor:
    numerator = (weights * lag_values * phases).sum(dim=0)
    denominator = (weights * lag_values.square()).sum(dim=0).clamp_min(1e-12)
    return numerator / denominator


def huber_lag_phase_slope(
    sig,
    max_lag=3,
    lags=None,
    delta: float = 0.7,
    iterations: int = 5,
):
    """Robust WLS slow-time phase slope from multi-lag autocorrelations."""
    lags, rk = _lag_autocorrelations(sig, max_lag=max_lag, lags=lags)
    if rk is None:
        return torch.zeros_like(sig[0].real)

    phases = _unwrap_lag_phases(torch.angle(rk))
    weights = rk.abs()
    lag_values = torch.as_tensor(lags, device=sig.device, dtype=phases.dtype)
    lag_values = lag_values.view(-1, *([1] * (sig.ndim - 1)))

    slope = _wls_origin_phase_slope(phases, weights, lag_values)
    delta = max(float(delta), 1e-6)
    for _ in range(max(0, int(iterations))):
        residual = phases - lag_values * slope
        robust = torch.clamp(delta / residual.abs().clamp_min(1e-6), max=1.0)
        slope = _wls_origin_phase_slope(phases, weights * robust, lag_values)
    return slope


def signed_tmas_wls_huber(
    sig,
    max_lag=5,
    delta: float = 0.7,
    iterations: int = 5,
):
    """TMAS weighted by a robust Huber-WLS multi-lag phase slope."""
    lags, rk = _lag_autocorrelations(sig, max_lag=max_lag)
    if rk is None:
        return torch.zeros_like(sig[0].real)

    tm = rk.abs().prod(dim=0)
    phase_per_frame = huber_lag_phase_slope(
        sig,
        max_lag=max_lag,
        lags=lags,
        delta=delta,
        iterations=iterations,
    )
    signed_scale = (phase_per_frame / np.pi).clamp(-1.0, 1.0)
    return signed_scale * tm


def phase_velocity_geomean_r2(
    sig,
    frame_rate: float,
    tx_freq: float,
    sound_speed: float = 1540.0,
    max_lag=5,
    lags=None,
):
    """Dower Coppler: phase velocity x geomean(|R_k|) x phase-fit R^2."""
    lags, rk = _lag_autocorrelations(sig, max_lag=max_lag, lags=lags)
    if rk is None:
        return torch.zeros_like(sig[0].real)

    phase_per_frame, quality = lag_phase_linear_fit(
        sig,
        max_lag=max_lag,
        lags=lags,
        weighted=True,
        fit_intercept=False,
    )
    frequency_hz = phase_per_frame * frame_rate / (2 * np.pi)
    phase_velocity = frequency_hz * sound_speed / (2 * tx_freq)
    geomean_r = torch.exp(torch.log(rk.abs().clamp_min(1e-30)).mean(dim=0))
    return phase_velocity * geomean_r * quality


def signed_tmas_phase_ls(sig, max_lag=3, fit_intercept: bool = False):
    """Signed multi-lag phase-LS map without multiplying all lag magnitudes.

    The phase slope is fit from ``angle(R_k) ~= omega * k`` with weights
    ``|R_k|``.  Magnitude is a weighted mean autocorrelation strength rather
    than the product used by TMAS, so weak higher lags do not collapse the map.
    The returned map is additionally scaled by fit quality, which emphasizes
    pixels whose phase really changes linearly with lag.
    """
    lags, rk = _lag_autocorrelations(sig, max_lag=max_lag)
    if rk is None:
        return torch.zeros_like(sig[0].real)

    phase_per_frame, quality = lag_phase_linear_fit(
        sig,
        max_lag=max_lag,
        lags=lags,
        weighted=True,
        fit_intercept=fit_intercept,
    )
    mag = rk.abs()
    weighted_mag = (mag.square().sum(dim=0) / mag.sum(dim=0).clamp_min(1e-12))
    signed_scale = (phase_per_frame / np.pi).clamp(-1.0, 1.0)
    return signed_scale * weighted_mag * quality


# Complex autocorrelation product (preserves direction through product)
def complex_autocorr_product(sig, lags=None):
    """Product of complex R_k — preserves directional information."""
    if lags is None:
        lags = [1, 2, 3]
    prod = None
    for k in lags:
        rk = (sig[k:] * torch.conj(sig[:-k])).mean(dim=0)
        prod = rk if prod is None else prod * rk
    return torch.angle(prod) / sum(lags)


# 9. Lateral autocorrelation proxy (like axial but lateral neighbours)
def lateral_autocorr_proxy(sig):
    """Phase compensated by lateral correlation — complement to axial."""
    RT = (sig[1:] * torch.conj(sig[:-1])).mean(dim=0)
    if sig.ndim == 3:
        RX = (sig[:, :, 1:] * torch.conj(sig[:, :, :-1])).mean(dim=0)
        RX_padded = torch.nn.functional.pad(RX, (0, 1), value=0)
    elif sig.ndim == 4:
        RX = (sig[:, :, :, 1:] * torch.conj(sig[:, :, :, :-1])).mean(dim=0)
        RX_padded = torch.nn.functional.pad(RX, (0, 1), value=0)
    else:
        return torch.angle(RT)
    angle_RX_abs = torch.angle(RX_padded).abs().clamp_min(1e-10)
    return torch.angle(RT) / angle_RX_abs


# 10. Spatiotemporal phase gradient magnitude
def spatiotemporal_gradient(sig):
    """sqrt(|dφ/dt|² + |dφ/dz|²) — combined phase gradient energy."""
    dphi_t = torch.angle(sig[1:] * torch.conj(sig[:-1]))
    grad_t2 = (dphi_t ** 2).mean(dim=0)
    if sig.ndim == 3:
        dphi_z = torch.angle(sig[:, 1:, :] * torch.conj(sig[:, :-1, :]))
        grad_z2 = (dphi_z ** 2).mean(dim=0)
        grad_z2 = torch.nn.functional.pad(grad_z2, (0, 0, 0, 1), value=0)
    elif sig.ndim == 4:
        dphi_z = torch.angle(sig[:, :, 1:, :] * torch.conj(sig[:, :, :-1, :]))
        grad_z2 = (dphi_z ** 2).mean(dim=0)
        grad_z2 = torch.nn.functional.pad(grad_z2, (0, 0, 0, 1), value=0)
    else:
        return torch.sqrt(grad_t2)
    return torch.sqrt(grad_t2 + grad_z2)


# Directional power ratio — low-frequency band only
def dir_ratio_low_band(sig):
    """Dir power ratio in inner-quarter frequencies — slow flow direction."""
    T = sig.shape[0]
    P = torch.fft.fftshift(torch.fft.fft(sig, dim=0).abs() ** 2, dim=0)
    mid = T // 2
    q = T // 8
    Pp = P[mid + 1 : mid + q].sum(dim=0)
    Pn = P[mid - q : mid].sum(dim=0)
    return (Pp - Pn) / (Pp + Pn).clamp_min(1e-12)


# Signed TMAS-PD: direction × log(1 + TMAS)
def signed_tmas_pd(sig, max_lag=3):
    """Direction × log(1+TMAS) — signed denoised flow map."""
    T = sig.shape[0]
    P = torch.fft.fft(sig, dim=0).abs() ** 2
    Pp = P[1 : T // 2].sum(dim=0)
    Pn = P[T // 2 + 1 :].sum(dim=0)
    D = (Pp - Pn) / (Pp + Pn).clamp_min(1e-12)
    lags = _tmas_lags(sig, max_lag=max_lag)
    if not lags:
        return torch.zeros_like(sig[0].real)
    rk_abs = [((sig[k:] * torch.conj(sig[:-k])).mean(0)).abs() for k in lags]
    tm = torch.stack(rk_abs, 0).prod(0)
    return D * torch.log1p(tm)


# Flow composite index — multiplicative meta-metric of winners
def flow_composite_index(sig, max_lag=3):
    """tmas^(1/3) × |dir_power_ratio|^(1/2) × circ_phase_var — meta-metric."""
    # TMAS
    lags = _tmas_lags(sig, max_lag=max_lag)
    if not lags:
        return torch.zeros_like(sig[0].real)
    rk_abs = [((sig[k:] * torch.conj(sig[:-k])).mean(0)).abs() for k in lags]
    tm = torch.stack(rk_abs, 0).prod(0)
    # Dir power ratio
    T = sig.shape[0]
    P = torch.fft.fft(sig, dim=0).abs() ** 2
    Pp = P[1 : T // 2].sum(dim=0)
    Pn = P[T // 2 + 1 :].sum(dim=0)
    D = (Pp - Pn) / (Pp + Pn).clamp_min(1e-12)
    # Circular phase variance
    dphi = torch.angle(sig[1:] * torch.conj(sig[:-1]))
    mc = torch.cos(dphi).mean(dim=0)
    ms = torch.sin(dphi).mean(dim=0)
    cv = 1.0 - torch.sqrt(mc ** 2 + ms ** 2)
    return tm.pow(1.0 / 3) * D.abs().pow(0.5) * cv


"""
More dopplers:
    
Alternative Blood-Flow Metrics for Transcranial Doppler Imaging from Complex Beamformed Ensembles
Executive summary
You already have a working color Doppler velocity estimator based on the Kasai lag‑1 autocorrelation and (elsewhere in the codebase) a power Doppler pipeline built around SVD clutter filtering. In the Alegria Neurotech repo alegria-neurotech/caterpillar, caterpillar/imaging/doppler.py implements (i) SVD clutter filtering (fast and full variants), (ii) a power Doppler image as the summed post‑filter energy, and (iii) a Kasai-style color Doppler velocity map computed from the phase of the mean lag‑1 correlation r1 = s[n]·conj(s[n−1]) with optional high‑pass filtering.

For transcranial brain vasculature, where skull attenuation/aberration and motion/reverberation can crush effective SNR, a good “flow metric” often needs to do at least one of: (a) exploit phase evolution (Doppler), (b) exploit decorrelation (moving scatterers lose coherence), (c) exploit spectral structure (moments/entropy of Doppler spectra), or (d) exploit subspace structure (blood vs tissue separation in spatiotemporal modes). Skull effects (attenuation, aberration, refraction, mode conversion) are widely recognized as major limiting factors in transcranial ultrasound image quality and Doppler sensitivity. 

Below are 20 alternative candidate metrics you can compute from your existing complex slow‑time tensor sig[T, P] (time × pixels, complex). They are not all “physically exact velocity,” but each can plausibly highlight blood flow under different failure modes. For each metric, you get: a concise definition/pseudocode; the intuition; expected advantages and failure modes in transcranial conditions; compute cost; normalization/visualization guidance (including when “velocity-like units” are possible); and PyTorch-leaning implementation notes consistent with complex beamformed frames.

Connector and repository findings
The only enabled connector is github (per your instruction to list all enabled connectors).

In alegria-neurotech/caterpillar, Doppler-related code and UI panels appear in at least:

caterpillar/imaging/doppler.py:

svd_filter_fast (eigendecomposition of (X X^H)) and svd_filter_full (full SVD on reshaped space×time matrix).
power_doppler: power image computed as (\sum_t | \text{SVD-filtered}(s_t) |^2) (with options like mean subtraction / skip-first-frames / optional normalization).
color_doppler: a Kasai lag‑1 autocorrelation estimator with optional high‑pass filtering and Gaussian blur, mapping phase→frequency→velocity using frame_rate, tx_freq, sound_speed parameters.
caterpillar/acquire/acquisition.py: the Acquisition object calls power_doppler(...) and then derives color_doppler_frames by slicing the Doppler signal into ensembles and passing frame_rate = empirical_pulse_repetition_rate_hz and tx_freq = config.tx_freq_hz.

UI panels demonstrate how results are visualized/scaled:

realtime/panels/color_doppler.py uses percentile clipping and a diverging colormap for velocity-like signed output.
realtime/panels/doppler_svd.py shows multiple SVD cutoff settings and optional dB display.
realtime/panels/doppler_diff.py displays signed differences vs a session mean (useful for change detection).
realtime/panels/doppler_signal_intensity.py plots mean (|\text{signal}|) over time (global or ROI).
Visualization helpers:

caterpillar/utils/plotting.py uses log compression to_db = 20*log10(|x|) with max subtraction.
Data examples:

The repo contains scripts for batch processing .h5 acquisitions (beamform + Doppler), but no obvious bundled datasets surfaced in code search; data paths appear external/environment-specific.
There are notebooks (e.g., ULM-related) that assume local acquisition files and demonstrate SVD filtering and interactive exploration; they function as usage examples, not embedded datasets.
Framing the problem for transcranial flow metrics
Classic color Doppler “velocity” estimation relies on the Doppler shift relation and sampling constraints. The core Doppler relation (including the angle factor) is typically written (v = \frac{f_D c}{2 f_0 \cos\theta}); in practice, many imaging implementations estimate the beam‑projected velocity by assuming (\cos\theta \approx 1) unless a vessel angle is known. 

Two transcranial realities shape what works:

Skull-induced degradation: attenuation and phase aberration reduce coherent energy and can distort phase-based estimators; transcranial imaging literature explicitly calls out attenuation, aberration, refraction, and mode conversion as key degraders of image quality. 

Clutter vs slow flow ambiguity: Doppler “wall filters” are high‑pass filters that suppress low-frequency, high-amplitude tissue motion, but they can also suppress genuine slow blood flow because the filter separates by frequency alone. 

Because of those, it’s often wise to treat “flow representation” as a family of outputs: some velocity-like (signed), some intensity-like (blood volume / moving scatterer energy), some quality-like (coherence or confidence), and some hemodynamic (pulsatility indices over time windows).

Candidate flow metrics
Shared notation and minimal helper conventions
Let sig be complex (s[t,p]) with shape ((T, P)). In your codebase, analogous tensors are complex beamformed/compounded frames over time.

Common optional preprocessing you may apply before any metric (depending on which failure mode dominates):

DC / mean removal in slow-time: (s[t] \leftarrow s[t] - \mathbb{E}_t[s[t]]) (removes stationary components).
High‑pass (wall) filtering in slow-time (Butterworth or FIR) to reject tissue clutter.
SVD clutter filtering (global or local). SVD-based spatiotemporal filtering is widely used in ultrafast Doppler / microvessel imaging and can outperform purely temporal high-pass filters in the presence of tissue motion. 
Below, each metric definition assumes you’ve decided what (if any) filtering to apply first.

Metric A — Lag‑k autocorrelation phase (Kasai generalization)
Definition: for integer lag (k \ge 1), [ r_k(p)=\frac{1}{T-k}\sum_{t=k}^{T-1} s[t,p];\overline{s[t-k,p]},\quad \hat f_D(p)=\frac{\angle r_k(p)}{2\pi}\cdot\frac{f_s}{k}. ] Kasai’s original method is the (k=1) case and is foundational for real-time color flow mapping. 

Intuition: larger lags average over longer time baselines; phase is still a frequency proxy if the signal remains coherent.
Transcranial pros/failures: can be more stable than lag‑1 when lag‑1 is dominated by phase noise, but more fragile to decorrelation from fast flow, motion, or aberration (phase coherence drops with increasing lag). Skull-driven phase instability can make higher lags noisier rather than better. 

Cost: (O(TP)).
Normalization / “velocity-like units”: (\hat v_\text{axial} = \hat f_D \cdot \frac{c}{2 f_0}) (beam-projected; (\theta) unspecified). 

PyTorch notes: use complex multiply + torch.angle, divide by (k); consider masking low-magnitude (r_k) to avoid random phase.

python
Copy
# A: lag-k phase estimator
k = 2
rk = (sig[k:] * sig[:-k].conj()).mean(dim=0)          # [P] complex
fd = torch.angle(rk) * (fs / (2*torch.pi*k))          # [P] Hz
v_ax = fd * (c / (2*f0))                              # units require fs, c, f0
Metric B — Multi‑lag phase‑slope regression (robust Doppler frequency)
Definition: compute (r_k) for (k=1..K), unwrap phases (\phi_k=\text{unwrap}(\angle r_k)), then fit [ \phi_k \approx 2\pi f_D \cdot k / f_s + \phi_0 ] by weighted least squares (weights (w_k = |r_k|) or SNR proxy).
Intuition: instead of trusting a single lag, you estimate the phase slope vs lag, which reduces variance when some lags are noisy. This is conceptually aligned with “use more than one lag” Doppler estimators, including 2D autocorrelation extensions. 

Transcranial pros/failures: more robust to lag‑1 dropouts; still vulnerable if phase unwrapping fails under low SNR or if decorrelation makes (r_k) unreliable for larger (k).
Cost: (O(KTP)) with small K (e.g., 2–8).
Normalization: same as Metric A for velocity-like mapping. 

PyTorch notes: avoid explicit unwrap per pixel if too heavy; you can fit using complex log slope approximation for small phase changes, or unwrap with torch.cumsum of wrapped diffs.

python
Copy
# B: multi-lag regression (sketch)
K = 4
rk = torch.stack([(sig[k:] * sig[:-k].conj()).mean(0) for k in range(1, K+1)], 0)  # [K,P]
phi = torch.angle(rk)  # wrapped; optionally unwrap across k
w = rk.abs().clamp_min(1e-8)
k = torch.arange(1, K+1, device=sig.device).float().view(K,1)
# slope ~= argmin_w ||phi - (a*k + b)|| ; closed-form weighted regression
Metric C — Loupas-style “2D autocorrelation” proxy using axial neighborhoods
Background: the 2D autocorrelation approach (Loupas–Powers–Gill) evaluates Doppler using correlations across both slow-time and samples within a range gate, improving robustness under some conditions. 

Definition (image-proxy): treat nearby pixels along depth (axial) as the “range-gate samples.” For each pixel (p), compute:

slow-time lag: (R_T = \sum_{t} s[t]\overline{s[t-1]}) (as usual),
axial lag: (R_Z = \sum_{z} s[z]\overline{s[z-1]}) using a small axial window around (p), then form a compensated estimate like ( \hat f_D \propto \arg(R_T) / \arg(R_Z)) or other stable ratio forms (you’ll tune the exact form empirically).
Intuition: skull aberrations and speckle can induce range-dependent phase; adding an axial correlation term can partially normalize local carrier/phase effects.
Transcranial pros/failures: could help when per-pixel phase is unstable but locally smooth in depth; may fail near strong reverberation layers where axial phase is corrupted (reverberation and multipath are common ultrasound artifacts). 

Cost: (O(TP)) plus small axial convolution.
Normalization: still yields a frequency-like quantity if tuned to match known Doppler scaling; otherwise treat as a “relative velocity index.”
PyTorch notes: requires reshaping P back to (Z,X) (and maybe Y) to do axial neighbor ops.
python
Copy
# C: axial-neighborhood proxy (sketch, assuming sig[T,Z,X])
RT = (sig[1:] * sig[:-1].conj()).sum(0)          # [Z,X]
RZ = (sig[:,1:,:] * sig[:,:-1,:].conj()).sum(0)  # [Z-1,X]
# combine RT and local RZ window around each pixel (implementation choice)
Metric D — Normalized lag‑1 coherence magnitude (“blood coherence”)
Definition: [ \rho_1(p) = \frac{\left|\mathbb{E}[s[t]\overline{s[t-1]}]\right|}{\mathbb{E}[|s[t]|^2] + \epsilon}. ] Optionally use (1-\rho_1) as a decorrelation metric.
Intuition: stationary tissue tends to be highly coherent; moving blood often decorrelates faster (especially with multiple scatterers and/or complex flow), so coherence magnitude can separate regimes even when phase is unreliable.
Transcranial pros/failures: more robust than phase when skull aberration makes phase erratic; however, probe motion and bulk tissue motion also reduce coherence, producing false positives unless motion correction / clutter filtering is good. Wall filters cannot separate slow blood from tissue just by frequency either. 

Cost: (O(TP)).
Normalization: map (\rho_1\in[0,1]) to intensity color map; if you want “velocity-like,” calibrate monotonic mapping (v_\text{pseudo} = a(1-\rho_1)^b) on phantom or by matching quantiles to a trusted velocity estimate.
PyTorch notes: compute r1_mean and p0_mean cheaply; clamp epsilon.

python
Copy
# D: normalized coherence magnitude
r1 = (sig[1:] * sig[:-1].conj()).mean(0)
p0 = (sig.abs()**2).mean(0)
rho1 = r1.abs() / p0.clamp_min(1e-8)
metric = 1.0 - rho1
Metric E — Instantaneous frequency via phase differences (robust median/MAD)
Definition: per time step, [ \Delta\phi_t(p)=\angle\left(s[t]\overline{s[t-1]}\right),\quad \hat f_D(p)=\text{median}_t(\Delta\phi_t)\cdot \frac{f_s}{2\pi}. ] Also compute dispersion: (\text{MAD}(\Delta\phi_t)) as a “turbulence/noise” map.
Intuition: Kasai averages complex vectors then takes an angle; this alternative takes all instantaneous phase steps and uses robust statistics.
Transcranial pros/failures: median is resistant to occasional phase jumps from low SNR; still fails if most (\Delta\phi) are random (very low coherence) or if phase wraps dominate (aliasing). Nyquist constraints and aliasing depend on PRF/slow-time sampling. 

Cost: near (O(TP)), but median is typically (O(T\log T)) per pixel if implemented naively; approximate with trimmed mean for speed.
Normalization: velocity-like via Doppler relation. 

PyTorch notes: torch.median is implemented but can be heavy; consider computing per chunk of pixels.

python
Copy
# E: instantaneous phase-step median
dphi = torch.angle(sig[1:] * sig[:-1].conj())          # [T-1,P]
fd = torch.median(dphi, dim=0).values * (fs/(2*torch.pi))
Metric F — Phase variance / jitter map
Definition: [ \text{PV}(p)=\text{Var}_t(\Delta\phi_t(p)) \quad \text{or} \quad \text{Var}_t(\angle r_1(t,p)). ] Clinically, “variance mapping” exists in color flow imaging, though it does not necessarily measure what people intuitively call “disturbed flow” and can correlate strongly with aliasing regions. 

Intuition: broad distributions of instantaneous Doppler phase shifts suggest mixed velocities, turbulence, or low SNR.
Transcranial pros/failures: can highlight vessel jets or complex flow, but also highlights anything that breaks coherence (skull-induced aberration, reverberation, motion). Reverberation is a classic artifact source. 

Cost: (O(TP)).
Normalization: unitless; visualize with log or percentile clipping. A “velocity-like” mapping is not physically meaningful, but you can scale PV to [0,1] and treat as a confidence/turbulence overlay on a signed velocity map.
PyTorch notes: compute circular variance for phase (use complex mean magnitude: (1-|E[e^{j\Delta\phi}]|)) to avoid wrap issues.

python
Copy
# F: circular phase variance
dphi = torch.angle(sig[1:] * sig[:-1].conj())
circ_var = 1.0 - torch.abs(torch.mean(torch.exp(1j*dphi), dim=0))
Metric G — Power Doppler (total Doppler power / moving scatterer energy)
Definition (post clutter-filter):
[ \text{PD}(p)=\sum_{t=t_0}^{T-1} |s_f[t,p]|^2 ] where (s_f) is a clutter-suppressed signal (SVD, wall filter, etc.). Power Doppler displays integrated Doppler power rather than mean frequency. 

Intuition: PD tracks the amount of moving blood scatterers (often closer to “blood volume / perfusion” than to velocity). In the repo, PD is implemented after SVD filtering.
Transcranial pros/failures: typically more robust than signed velocity under poor angle knowledge and low coherence; but it is sensitive to noise floor and residual clutter (motion). “Noise equalization” and SVD variants exist specifically because deep-region Doppler can saturate with noise. 

Cost: metric itself (O(TP)); total pipeline can be dominated by SVD cost unless you use fast/approximate variants. 

Normalization: log compression (dB) is common; your repo uses to_db for other displays.
Velocity-like mapping: generally not valid; if you must, treat (\sqrt{\text{PD}}) as a “magnitude proxy” and calibrate.
PyTorch notes: standard sum(abs(sig_f)**2, dim=0); consider subtracting estimated noise power.

python
Copy
# G: power Doppler after filtering sig_f
pd = (sig_f.abs()**2).sum(dim=0)   # [P]
pd_db = 10*torch.log10(pd.clamp_min(1e-12))
Metric H — Fractional Moving Blood Volume (FMBV‑style normalization)
Definition: a normalized PD: [ \text{FMBV}(p)=\frac{\text{PD}(p)}{\text{PD}\text{ref}} ] where ( \text{PD}\text{ref} ) is a reference (e.g., max in a vessel ROI or a calibration flow tube). “Fractional moving blood volume” was proposed as PD normalized to a maximum reference, aiming for depth normalization and a more continuous measure. 

Intuition: transcranial settings vary gain/TGC and attenuation; relative normalization can make comparisons more stable than raw PD.
Transcranial pros/failures: helps mitigate depth-dependent attenuation variability, but still inherits PD’s sensitivity to clutter/noise and depends on stable reference selection (which can be hard transcranially if the “best vessel” changes). Skull window absence is common in a sizeable minority of subjects, complicating “reference vessel” assumptions. 

Cost: (O(TP)).
Normalization: typically clamp FMBV to [0,1] and display with “hot” map; optionally use log(1+α·FMBV).
PyTorch notes: compute global or ROI-based reference per frame; beware division by near-zero.

python
Copy
# H: FMBV-style normalized PD
pd = (sig_f.abs()**2).sum(0)
ref = torch.quantile(pd, 0.995)   # robust "max" proxy
fmbv = (pd / ref.clamp_min(1e-12)).clamp(0, 1)
Metric I — Directional power ratio (signed flow without phase)
Definition: compute Doppler spectrum (P(f)) via FFT along time; integrate positive and negative frequency halves: [ P_+(p)=\sum_{f>0} P(f,p),; P_-(p)=\sum_{f<0} P(f,p),; D(p)=\frac{P_+ - P_-}{P_+ + P_- + \epsilon}. ] Intuition: direction comes from which side of the spectrum holds more energy, avoiding fragile single-angle phase at one lag.
Transcranial pros/failures: robust when phase is noisy but the spectrum is still asymmetric; fails if clutter dominates or if aliasing folds energy (Nyquist/aliasing depends on PRF). 

Cost: (O(PT\log T)).
Normalization / velocity-like mapping: (D\in[-1,1]) is direction confidence; combine with PD magnitude (Metric G) to form signed intensity (D\cdot \log(1+\text{PD})). Velocity mapping would require a frequency estimate (see centroid/peak, Metrics J/K).
PyTorch notes: torch.fft.fft then power; beware DC bin and windowing.

python
Copy
# I: directional power ratio
S = torch.fft.fft(sig_f, dim=0)          # [T,P]
P = (S.abs()**2)
Pp = P[1:T//2].sum(0)
Pn = P[T//2+1:].sum(0)
D = (Pp - Pn) / (Pp + Pn).clamp_min(1e-12)
Metric J — Spectral centroid (mean Doppler frequency)
Definition: [ \bar f(p)=\frac{\sum_f f,P(f,p)}{\sum_f P(f,p)+\epsilon}. ] Color flow mapping is often described as estimating mean Doppler frequency/velocity, historically linked to autocorrelation-based estimators. 

Intuition: centroid uses all frequencies, often smoothing noise and capturing mean flow.
Transcranial pros/failures: can outperform peak picking under noise; but broad clutter leakage biases the centroid toward 0 unless filtering is strong (SVD/wall). Wall filters can also remove true slow flow. 

Cost: FFT (O(PT\log T)) + moment sums.
Normalization: velocity-like via (v_\text{axial} = \bar f \cdot \frac{c}{2 f_0}) (angle unspecified). 

PyTorch notes: use fftshift frequencies or construct frequency bins consistent with your sampling rate.

python
Copy
# J: spectral centroid
S = torch.fft.fft(sig_f, dim=0)
P = S.abs()**2
freq = torch.fft.fftfreq(T, d=1/fs).to(sig.device).view(T,1)
fbar = (freq * P).sum(0) / P.sum(0).clamp_min(1e-12)
Metric K — Peak / ML frequency from periodogram
Definition: (f_\text{peak}(p)=\arg\max_f P(f,p)).
Intuition: when flow is laminar and SNR is decent, the dominant frequency can track centerline velocity components.
Transcranial pros/failures: brittle under low SNR or multi-component flow; also sensitive to aliasing and windowing. Nyquist/aliasing constraints are particularly salient in pulsed Doppler and color Doppler. 

Cost: (O(PT\log T)).
Normalization: velocity-like via Doppler relation. 

PyTorch notes: apply a taper (Hann) to reduce spectral leakage; for sub-bin precision, quadratic interpolation around the peak.

python
Copy
# K: peak frequency bin
w = torch.hann_window(T, device=sig.device).view(T,1)
S = torch.fft.fft(sig_f*w, dim=0)
P = S.abs()**2
idx = torch.argmax(P, dim=0)
fpeak = torch.fft.fftfreq(T, d=1/fs).to(sig.device)[idx]
Metric L — Spectral bandwidth (second central moment)
Definition: [ \sigma_f^2(p)=\frac{\sum_f (f-\bar f)^2 P(f,p)}{\sum_f P(f,p)+\epsilon}. ] Variance maps in clinical color flow exist, but empirical work shows they can correlate with aliasing rather than true “disturbed flow.” 

Intuition: broader spectra can indicate turbulent/mixed velocities, or simply noise and decorrelation.
Transcranial pros/failures: can highlight jets or complex flow if the signal is real; will also spike in low SNR regions behind skull or in reverberation zones. 

Cost: (O(PT\log T)).
Normalization: unit is Hz²; map (\sqrt{\sigma_f^2}) to “Hz bandwidth” then (optionally) to “velocity spread” using Doppler scaling (angle unknown). 

PyTorch notes: compute centroid first, then bandwidth.

python
Copy
# L: spectral bandwidth
fbar = (freq*P).sum(0) / P.sum(0).clamp_min(1e-12)
bw2 = ((freq - fbar)**2 * P).sum(0) / P.sum(0).clamp_min(1e-12)
bw = torch.sqrt(bw2.clamp_min(0.0))
Metric M — Spectral skewness (third moment)
Definition: [ \gamma_1(p)=\frac{\sum_f (f-\bar f)^3 P(f,p)}{(\sigma_f^2)^{3/2}\sum_f P(f,p)+\epsilon}. ] Intuition: detects asymmetric spectra beyond what the centroid captures; might separate arterial-like waveforms (more high-frequency content) from venous-like (lower) in some regimes. Ultrafast Doppler literature notes arterial/venous differentiation from hemodynamics in brain imaging contexts. 

Transcranial pros/failures: sensitive to bias from clutter leakage; requires decent spectral SNR.
Cost: (O(PT\log T)).
Normalization: dimensionless; visualize with diverging colormap; optionally gate by PD (Metric G) so background noise doesn’t dominate.
PyTorch notes: compute using existing fbar and bw.

python
Copy
# M: spectral skewness
m3 = ((freq - fbar)**3 * P).sum(0) / P.sum(0).clamp_min(1e-12)
skew = m3 / (bw**3 + 1e-12)
Metric N — Spectral kurtosis (fourth moment)
Definition: [ \gamma_2(p)=\frac{\sum_f (f-\bar f)^4 P(f,p)}{(\sigma_f^2)^2\sum_f P(f,p)+\epsilon}. ] Intuition: peaky vs flat spectra; could highlight narrowband laminar flow vs broadband noisy/turbulent regions.
Transcranial pros/failures: like skewness, needs sufficient Doppler SNR; can strongly react to windowing and residual noise floor.
Cost: (O(PT\log T)).
Normalization: dimensionless; clip extreme values (kurtosis can blow up when bandwidth is tiny).
PyTorch notes: add clamps on bw.

python
Copy
# N: spectral kurtosis
m4 = ((freq - fbar)**4 * P).sum(0) / P.sum(0).clamp_min(1e-12)
kurt = m4 / (bw**4 + 1e-12)
Metric O — Spectral entropy (“how spread out is Doppler power?”)
Definition: normalize spectrum as (p(f)=P(f)/\sum P), then [ H(p)=-\sum_f p(f)\log p(f). ] Intuition: distinguishes narrowband coherent flow (lower entropy) from broadband/noisy/mixed motion (higher entropy).
Transcranial pros/failures: can serve as a quality/confidence map; but entropy also rises with motion artifact and deep noise saturation (a known issue in ultrafast microvessel imaging without careful noise handling). 

Cost: (O(PT\log T)).
Normalization: divide by (\log N_f) to get [0,1] entropy; overlay with PD to suppress background-only regions.
PyTorch notes: clamp probabilities to avoid log(0).

python
Copy
# O: spectral entropy
p = P / P.sum(0).clamp_min(1e-12)
H = -(p * torch.log(p.clamp_min(1e-12))).sum(0)
Hn = H / torch.log(torch.tensor(float(T), device=sig.device))
Metric P — Band-limited Doppler energy (select a velocity window)
Definition: choose a frequency band ([f_1,f_2]) (or multiple bands) and compute [ E_{[f_1,f_2]}(p)=\sum_{f\in[f_1,f_2]} P(f,p). ] Intuition: can suppress near‑DC residual clutter and suppress high-frequency bins dominated by noise or artifacts in your system.
Transcranial pros/failures: useful when you empirically know where clutter lives (near 0) and where noise dominates; but if PRF is low or aliasing occurs, true flow power may fold into unexpected bins. 

Cost: FFT (O(PT\log T)).
Normalization: energy → log compression; optionally compute directional band energy difference like Metric I but within a band.
PyTorch notes: precompute freq and boolean mask; use symmetric bands if you want direction-invariant energy.

python
Copy
# P: band-limited energy
mask = (freq.abs() >= f1) & (freq.abs() <= f2)
E = P[mask.squeeze(1)].sum(0)
Metric Q — High‑lag autocorrelation “temporal multiply-and-sum” (TMAS) / coherence-boosted PD
Background: recent work describes coherence-based power Doppler estimation using high-lag autocorrelation and nonlinear compounding approaches to suppress uncorrelated noise, trading off against flow decorrelation. 

Definition (one workable form): for chosen lags (k\in\mathcal{K}), [ \text{TMAS}(p)=\left|\sum_{k\in\mathcal{K}} \mathbb{E}t\left[s[t]\overline{s[t-k]}\right]\right|^\alpha ] or multiply magnitudes across lags to suppress uncorrelated components: [ \text{TMAS}\times(p)=\prod_{k\in\mathcal{K}} \left|\mathbb{E}_t[s[t]\overline{s[t-k]}]\right|. ] Intuition: correlated blood signal persists across multiple lags more than random noise; multiplying/summing across lags can be a strong denoiser.
Transcranial pros/failures: can be excellent where noise dominates; but fast flow and motion cause decorrelation across lags (signal loss), especially problematic if skull aberration already reduces coherence. 

Cost: (O(|\mathcal{K}|TP)).
Normalization: intensity-like map; log compress; optionally sign with Metric I (direction power ratio).
PyTorch notes: keep (|\mathcal{K}|) small (e.g., 2–6) and tune.

python
Copy
# Q: TMAS-style multi-lag coherence product
lags = [1, 2, 3]
rk_abs = [ (sig[k:] * sig[:-k].conj()).mean(0).abs() for k in lags ]
tmas = torch.stack(rk_abs, 0).prod(0)
Metric R — Temporal total-variation energy (complex difference energy)
Definition: [ \text{TVE}(p)=\sum_{t=1}^{T-1} |s[t,p]-s[t-1,p]|^2. ] Intuition: fast-changing signals yield high TV energy; can highlight moving scatterers in blood after clutter suppression.
Transcranial pros/failures: catches motion and flow; but without good clutter removal, bulk tissue/probe motion dominates. Reverberation can also create rapid changes unrelated to flow. 

Cost: (O(TP)).
Normalization: intensity-like; log compress; often combine with a clutter-suppression pre-step (SVD/wall filter).
PyTorch notes: simple and fast; good baseline metric when you want a “motion energy” proxy.

python
Copy
# R: temporal difference energy
ds = sig_f[1:] - sig_f[:-1]
tve = (ds.abs()**2).sum(0)
Metric S — Speckle variance (magnitude or power variance over time)
Definition:
[ \text{SV}(p)=\text{Var}_t(|s[t,p]|) \quad \text{or} \quad \text{Var}_t(|s[t,p]|^2). ] Intuition: moving scatterers change speckle statistics across frames; variance can reveal perfused vessels even when phase is unreliable. Similar “speckle variance” ideas are well known in other coherent imaging modalities for microvasculature detection (e.g., OCT), where it is noted to be relatively angle-insensitive in certain regimes. 

Transcranial pros/failures: can be robust to phase aberration; but strongly sensitive to any global motion and to changing noise floor with depth/gain. Wall filters and gain settings affect detectability of low-velocity flow vs clutter. 

Cost: (O(TP)).
Normalization: standardize by mean power: (\text{SV}/(\mathbb{E}[|s|^2]+\epsilon)) to reduce depth biases; visualize with log and percentile clipping.
PyTorch notes: do variance in float32/float64 for stability.

python
Copy
# S: speckle variance of power
pwr = sig.abs()**2
sv = pwr.var(dim=0)
sv_norm = sv / pwr.mean(0).clamp_min(1e-12)
Metric T — Spatiotemporal neighbor coherence (vesselness-by-similarity)
Definition: for each pixel (p), compute average complex correlation with its neighbors (q\in\mathcal{N}(p)): [ C(p)=\frac{1}{|\mathcal{N}|}\sum_{q\in\mathcal{N}} \frac{\left|\mathbb{E}_t[s[t,p]\overline{s[t,q]}]\right|}{\sqrt{\mathbb{E}|s[t,p]|^2;\mathbb{E}|s[t,q]|^2}+\epsilon}. ] Intuition: a vessel often occupies multiple adjacent pixels; true flow signals can show locally consistent temporal structure, whereas random noise is less spatially coherent. Spatiotemporal coherence has been explicitly exploited in SVD-based clutter filtering and related coherence metrics in microvasculature imaging quality assessment. 

Transcranial pros/failures: improves robustness when per-pixel SNR is low but spatial support exists; fails on thin vessels at resolution limit and can be confused by coherent reverberation patterns. 

Cost: (O(TP|\mathcal{N}|)) with small neighbor sets (4–8).
Normalization: (C\in[0,1]); use as confidence mask or multiply with PD for a coherence-weighted PD map.
PyTorch notes: easiest if sig is reshaped to (T,H,W) to shift and correlate.

python
Copy
# T: neighbor coherence (sketch; assuming sig[T,H,W])
p0 = (sig.abs()**2).mean(0)
corr_right = (sig[:,:,1:] * sig[:,:,:-1].conj()).mean(0).abs()
den = (p0[:,:,1:] * p0[:,:,:-1]).sqrt().clamp_min(1e-12)
C = corr_right / den   # repeat for other neighbor directions and average
Metric U — Local patch PCA/SVD “blood energy fraction”
Background: SVD-based clutter filtering is widely used in ultrafast Doppler because tissue tends to lie in a highly coherent low-rank subspace while blood occupies different components; the approach substantially increases Doppler sensitivity in ultrafast settings. 

Definition (local): for each spatial patch around pixel (p), form matrix (X\in\mathbb{C}^{T\times N}) where (N) is #pixels in patch. Let singular values be (\sigma_1\ge\dots\ge\sigma_r). Define: [ \text{BEF}(p)=\frac{\sum_{i=i_1}^{i_2}\sigma_i^2}{\sum_{i}\sigma_i^2} ] for mid-order indices ([i_1,i_2]) intended to capture blood (tissue often dominates top singular values).
Intuition: this yields a local version of “blood subspace energy,” less prone to global motion dominating everything.
Transcranial pros/failures: can be very robust to spatially varying clutter (a transcranial reality), but heavier compute; also selection of ([i_1,i_2]) is data-dependent and may drift with motion. Adaptive boundary selection has been studied (globally) for ultrafast Doppler. 

Cost: high: for each patch, SVD (O(\min(T,N),T,N)); mitigate by downsampling patches or using approximate SVD. 

Normalization: BEF (\in[0,1]); display as heatmap or multiply by PD.
PyTorch notes: implement patch extraction with unfold; batch SVD on many patches can be GPU-heavy.

python
Copy
# U: local patch SVD energy fraction (conceptual sketch)
# X: [T, Npatch] complex per patch; do svdvals, then ratio of selected energies
Metric V — Low-rank residual energy per pixel (local robust “moving energy”)
Definition: in a patch, fit rank-(r) low-rank model (X_r) (tissue) and compute residual (R=X-X_r). Define pixel metric as residual energy at that pixel aggregated over time: [ \text{RRE}(p)=\sum_t |R[t,p]|^2. ] Intuition: explicit “what is not explainable by tissue subspace.” This matches how SVD clutter filtering conceptually operates, but done locally. 

Transcranial pros/failures: more accurate separation under spatially varying clutter; but fails if blood becomes coherent enough to leak into the low-rank model (e.g., large vessels with strong coherent flow) or if motion makes tissue not low-rank locally.
Cost: high (local low-rank).
Normalization: intensity-like; log compress; optionally normalize by patch mean to reduce depth bias.
PyTorch notes: if you already compute local SVD once, you get both U and V “for free.”

python
Copy
# V: residual energy (sketch)
# after computing low-rank Xr, residual R = X - Xr, then sum |R|^2 per pixel
Metric W — Tensor (HOSVD) energy split across slow-time and spatial modes
Background: higher-order SVD (HOSVD) extensions have been explored for ultrafast Doppler clutter filtering under non-negligible tissue motion. 

Definition: reshape your data into a tensor ( \mathcal{X}\in\mathbb{C}^{T\times H\times W}) (or (T\times E\times Z\times X) if you have elevational planes) and compute a low-rank factorization (HOSVD/Tucker). Define a flow metric as energy in selected temporal factors or residual energy after removing a low-rank “tissue” Tucker component.
Intuition: transcranial clutter may be structured across multiple spatial dimensions; tensor decompositions can model that structure more flexibly than matrix SVD.
Transcranial pros/failures: potentially powerful for wide-FOV with structured reverberation, but computationally heavy and sensitive to rank choices; might be best as an offline baseline.
Cost: very high relative to others (typically superlinear in tensor dimensions). 

Normalization: intensity-like; log compress; compare across acquisitions via robust quantiles.
PyTorch notes: use libraries or implement alternating least squares; batch sizes matter.

python
Copy
# W: HOSVD/Tucker (high-level idea)
# X = sig.reshape(T,H,W) -> compute tucker ranks -> residual energy map
Metric X — Wavelet-domain clutter-suppressed energy (flow energy in wavelet bands)
Background: wavelet methods have been proposed for Doppler wall/clutter suppression to preserve low-velocity flow better than simple high-pass filtering. 

Definition: apply a wavelet transform along slow-time (t) per pixel; compute energy in selected detail bands as a flow metric: [ E_\text{wav}(p)=\sum_{\ell\in\mathcal{L}}|d_\ell(p)|_2^2. ] Intuition: wavelets can separate transient tissue motion artifacts and persistent flow components across scales.
Transcranial pros/failures: can be robust to nonstationary motion; but tuning scales is nontrivial, and strong reverberation may occupy similar bands.
Cost: typically (O(TP)) for fast wavelets; constants can be higher than simple filtering.
Normalization: energy → log compress; optionally depth-normalize by estimated noise.
PyTorch notes: implement via convolution filterbanks; keep it differentiable if needed.

python
Copy
# X: wavelet-band energy (sketch)
# coeffs = wavelet_transform(sig.real or sig) along time; energy in chosen bands
Metric Y — Empirical Mode Decomposition (EMD) residual energy (slow-flow preserving)
Background: EMD has been proposed to remove wall components without wiping out low-velocity blood as aggressively as high-pass filters. 

Definition: decompose each pixel time series into intrinsic mode functions (IMFs); discard low-frequency IMFs as clutter estimate; metric is energy of remaining IMFs.
Intuition: adaptive decomposition can outperform fixed cutoffs when clutter frequency varies over time (motion).
Transcranial pros/failures: potentially good under variable motion; heavy compute and less GPU-friendly; uncertain behavior under very low SNR.
Cost: high (iterative per pixel).
Normalization: intensity-like; log compress.
PyTorch notes: often implemented on CPU; consider it an offline reference.

python
Copy
# Y: EMD residual (conceptual)
# imfs = emd(sig[:,p]) ; blood = sum(imfs[k:]) ; metric = sum |blood|^2
Metric Z — Short-time Doppler spectrum + resistivity/pulsatility indices
Background: ultrafast Doppler has enabled pixelwise Doppler spectra and derived hemodynamic indices (e.g., resistivity index maps) in brain imaging contexts. 

Definition: compute a time-varying velocity estimate (v(t,p)) from short-time spectra (STFT centroid or peak), then compute:

Resistivity Index: (RI(p) = \frac{V_\text{syst}(p)-V_\text{diast}(p)}{V_\text{syst}(p)+\epsilon}),
Pulsatility Index: (PI(p) = \frac{V_\text{syst}-V_\text{diast}}{V_\text{mean}+\epsilon}). Intuition: these indices can highlight arteries vs veins and vascular state beyond a static flow snapshot.
Transcranial pros/failures: needs enough time coverage (ideally spanning cardiac cycles) and stable motion; in transcranial settings, window quality varies and can fail in some subjects. 

Cost: high: multiple FFTs per pixel (STFT).
Normalization: outputs are dimensionless indices; display with perceptual colormap and mask where confidence is low (e.g., low PD or high entropy).
PyTorch notes: batch STFT along time; keep windows and hop sizes as parameters (all unspecified).
python
Copy
# Z: STFT->v(t)->RI/PI (sketch)
# Vt = stft_based_velocity(sig_f, fs, f0, c)   # [Tw,P]
# RI = (Vt.max(0).values - Vt.min(0).values) / Vt.max(0).values.clamp_min(1e-12)
Metric AA — Coherence-weighted power Doppler (PD × coherence confidence)
Background: coherence factor ideas and coherence-based metrics are used to improve contrast and interpretability in ultrasound imaging, and are known to be SNR-sensitive. 

Definition: combine a PD-like magnitude with a coherence/confidence term: [ \text{CwPD}(p)=\text{PD}(p)\cdot \rho_1(p)^\beta ] or (\text{PD}\cdot C(p)) with neighbor coherence (Metric T).
Intuition: suppress regions where “flow energy” is likely noise because coherence is low.
Transcranial pros/failures: often helpful because skull attenuation raises deep noise; but if real flow is decorrelated (fast/complex flow), coherence weighting can suppress it too.
Cost: low/medium (PD (O(TP)) + coherence (O(TP))).
Normalization: log compress; tune (\beta) and clip quantiles.
PyTorch notes: easy composition.

python
Copy
# AA: coherence-weighted PD
pd = (sig_f.abs()**2).sum(0)
rho1 = (sig_f[1:] * sig_f[:-1].conj()).mean(0).abs() / (sig_f.abs()**2).mean(0).clamp_min(1e-12)
cw_pd = pd * rho1.pow(1.0)
Metric AB — Signed “power × direction” composite (pseudo-velocity visualization)
Definition: combine directional ratio (D) (Metric I) and magnitude PD (Metric G): [ \text{SPD}(p) = D(p)\cdot \log(1+\alpha,\text{PD}(p)). ] Intuition: yields a signed map that “looks like” color Doppler (direction) but uses robust power for magnitude.
Transcranial pros/failures: often visually stable under low coherence; can still be wrong under aliasing and clutter leakage. 

Cost: (O(PT\log T)).
Velocity-like mapping: not physically velocity; you can calibrate α so typical vessel values match the dynamic range of a velocity map, then label it as “signed flow index” rather than m/s.
PyTorch notes: trivial once you have D and pd.

python
Copy
# AB: signed power Doppler composite
spd = D * torch.log1p(alpha * pd)
Metric AC — Frame-to-frame normalized cross-correlation on magnitude (SIV-inspired flow proxy)
Background: cross-correlation of speckle patterns has been used in ultrasound speckle image velocimetry (SIV) to infer flow; it is improved by preprocessing due to heterogeneous speckles. 

Definition (simple local): in a small window around each pixel, compute normalized cross-correlation between (|s[t]|) and (|s[t-1]|); use the reduction in NCC or the displacement that maximizes NCC as a flow proxy.
Intuition: moving scatterers shift speckle; correlation drops or shifts peak.
Transcranial pros/failures: can work even if phase is unreliable; but transcranial images often have low speckle contrast and motion, making registration difficult.
Cost: medium/high depending on search window (classic PIV is heavy).
Normalization: output can be “decorrelation” or displacement (pixels/frame). Converting to m/s requires spatial calibration and frame rate (both unspecified).
PyTorch notes: implement with conv2d for NCC; restrict to small displacements.

python
Copy
# AC: NCC drop (very simplified)
a = sig.abs()[1:]       # [T-1,P] or reshape to [T-1,1,H,W]
b = sig.abs()[:-1]
ncc_drop = 1.0 - cosine_similarity(a, b)   # proxy; true NCC is windowed
Metric AD — Clutter-subtracted “difference from session mean” (change detection)
Repo note: a session-mean difference display already exists for power Doppler in the UI layer.
Definition: for any scalar flow map (M(p)) (PD, SPD, centroid velocity, etc.), compute (\Delta M = M_\text{current}-\mathbb{E}[M_\text{history}]).
Intuition: transcranial signals drift (probe pressure/angle, window changes); differencing can highlight relative changes (task-evoked perfusion changes, etc.) even when absolute scale is unstable.
Transcranial pros/failures: useful for within-session comparisons; not an absolute flow measure; can be confounded by slow motion drift.
Cost: negligible beyond metric cost.
Normalization: robust symmetric scaling (percentile of |Δ|) matches your panel design pattern.
PyTorch notes: maintain running mean/variance online.

python
Copy
# AD: delta-from-running-mean
delta = metric - running_mean
running_mean = running_mean + eta*(metric - running_mean)
Metric AE — Ensemble coherence quality score (per-pixel or global) Background: “ensemble coherence” metrics have been proposed to assess/quantify Doppler ensemble quality and its improvement under motion correction in contrast-free microvasculature imaging. 

Definition: define a quality score such as (\rho_1(p)) (Metric D), neighbor coherence (C(p)) (Metric T), or a combined score: [ Q(p)=\rho_1(p)\cdot C(p). ] Intuition: treat coherence as a confidence map to gate any flow estimate.
Transcranial pros/failures: extremely useful when skull window quality varies: you can suppress “flow” visuals in incoherent regions (avoid hallucinating vessels). But coherence drops both for true fast flow and for motion; gating can hide true positives.
Cost: low/medium.
Normalization: map to [0,1] and use as alpha channel mask.
PyTorch notes: use Q to weight displays, not necessarily as the primary metric.

python
Copy
# AE: combined quality
Q = (rho1 * C).clamp(0, 1)
Comparison and trade-offs
Summary table of the 20 metrics
Key: interpretability (“does it correspond to a familiar physical quantity?”), sensitivities (slow/fast flow), robustness (noise + motion + skull-induced artifacts), cost, and implementation ease for your data format (sig[T,P] complex).

ID	Metric (short name)	Physical interpretability	Slow flow sensitivity	Fast flow sensitivity	Robustness to noise/motion	Computational cost	Ease (PyTorch)
A	Lag‑k phase	High (velocity-like)	Med	Med→Low (decorrelation)	Med	Low	Easy
B	Multi‑lag phase regression	High	Med	Med	Med→High	Low→Med	Med
C	2D autocorr proxy (axial neighborhood)	Med→High	Med	Med	Med	Low→Med	Med
D	Coherence magnitude (ρ₁)	Med (confidence/flow)	High	Med	High (phase-robust)	Low	Easy
E	Robust phase-step median	High	Med	Med	Med→High	Med	Med
F	Phase variance / circular variance	Med (turbulence/confidence)	Med	Med	Low→Med (false positives)	Low	Easy
G	Power Doppler (PD)	Med (flow energy)	High	High	Med (needs filtering)	Low*	Easy
H	FMBV (normalized PD)	Med	High	High	Med	Low*	Easy
I	Directional power ratio	Med (direction)	Med	High	Med	Med	Med
J	Spectral centroid	High (mean freq/vel)	Med	High	Med	Med	Med
K	Peak frequency	High	Low→Med	High	Low→Med	Med	Med
L	Spectral bandwidth	Med (turbulence proxy)	Med	Med	Low→Med	Med	Med
M	Spectral skewness	Low→Med	Low→Med	Med	Low	Med	Med
N	Spectral kurtosis	Low→Med	Low→Med	Med	Low	Med	Med
O	Spectral entropy	Low→Med (quality proxy)	Med	Med	Med	Med	Med
P	Band-limited energy	Med	High†	High†	Med	Med	Med
Q	TMAS / high-lag coherence PD	Med	High	Med→Low (decorrelation)	High (noise)	Med	Med
R	Temporal difference energy (TVE)	Low→Med	High	High	Low→Med (motion confound)	Low	Easy
S	Speckle variance	Low→Med	High	Med	Low→Med (motion confound)	Low	Easy
T	Neighbor spatiotemporal coherence	Med (confidence/vesselness)	High	Med	Med→High	Med	Med
U	Local patch SVD blood-energy fraction	Med	High	High	High	High	Hard
V	Local low-rank residual energy	Med	High	High	High	High	Hard
W	Tensor/HOSVD energy split	Med	High	High	High	Very High	Hard
X	Wavelet band energy	Med	High	Med	Med	Med	Med
Y	EMD residual energy	Med	High	Med	Med	High	Hard
Z	STFT → RI/PI indices	High (hemodynamic indices)	Med	Med	Med (needs long stable data)	High	Hard
AA	Coherence-weighted PD	Med	High	High	Med→High	Low→Med	Easy
AB	Signed PD composite	Low→Med	High	High	Med	Med	Med
AC	NCC/SIV-like proxy	Med (displacement proxy)	Med	Med	Low→Med	High	Hard
AD	Δ from mean (change map)	Low→Med	High	High	Med	+Negligible	Easy
AE	Coherence quality score	Med	—	—	High	Low→Med	Easy

* PD itself is low-cost, but the total pipeline cost depends on the clutter filter (SVD can dominate). 

† Depends on your choice of band; if you exclude too much low-frequency content, you can erase slow flow, consistent with wall filter limitations. 

A compact “tradeoff chart” (robustness vs compute)
This is a qualitative grouping (not a benchmark), meant to guide what to try first on transcranial data.

Lower compute	Higher compute
More robust to noise/motion	D, G/H, Q, T, AA, AE	U/V, W, Z (if motion stable)
Less robust / more fragile	A (high-lag), F, R, S	C (if reverberation heavy), K/M/N, AC

When transcranial SNR is poor, “robustness-first” metrics like PD variants + coherence gating (G/H/AA/AE) and multi-lag coherence (Q) often beat pure velocity estimators visually, while at least giving you meaningful vessel maps—especially if you already rely on SVD-based clutter suppression as in ultrafast Doppler/microvessel imaging practice. 

Implementation notes and processing pipelines
Practical pipeline templates
Your repo already reflects a common Doppler pipeline: beamform → clutter filter → integrate to PD; and for color Doppler, filtered signal → lag‑1 phase → velocity scaling.

Two pipeline patterns cover most metrics above:

mermaid
Copy
flowchart TD
  A[Complex beamformed ensemble sig[T,P]] --> B{Clutter suppression?}
  B -->|None / mean subtract| C[Optional mean/DC removal]
  B -->|Wall filter (HPF)| D[Temporal HPF]
  B -->|SVD / subspace| E[SVD or local low-rank]
  C --> F[Metric computation]
  D --> F
  E --> F
  F --> G[Normalization: log/percentile/zscore]
  G --> H[Visualization: colormap + confidence mask]
mermaid
Copy
flowchart TD
  A[Complex sig[T,P]] --> B[FFT/STFT along time]
  B --> C[Power spectrum P(f)]
  C --> D[Moments: centroid/bandwidth/skew/kurtosis]
  C --> E[Directional energy P+ vs P-]
  C --> F[Entropy / band energy]
  D --> G[Velocity-like scale if desired]
  E --> H[Signed intensity composite]
  F --> I[Confidence/quality overlays]
Normalization patterns that work well in practice
Percentile clipping (e.g., 99–99.9%) is often more stable than max-scaling under rare outliers (your color Doppler panel uses robust percentiles for symmetric limits).
Log compression for intensity-like metrics (PD, band energy): (10\log_{10}(\cdot)) or (\log(1+\alpha x)). Log compression is consistent with your existing to_db helper for amplitude-like displays.
Coherence as alpha mask: display a velocity-like hue map (A/B/J/K) but modulate transparency by a confidence (Q) (AE). This is particularly important transcranially to avoid strong-looking artifacts in incoherent regions. 
Mapping to “velocity-like units” without assuming parameters
If you want a numerical velocity scale (m/s) for metrics that estimate Doppler frequency (f_D) (A/B/E/J/K), the common mapping is:

[ v_\text{axial} = \frac{c}{2 f_0} f_D \quad (\text{beam-projected; }\cos\theta\ \text{unspecified}) ] or, if you have angle information, (v = \frac{c}{2 f_0 \cos\theta} f_D). 

For intensity-like metrics (PD, entropy, coherence), do not label as m/s; instead:

label as “flow power,” “flow index,” or “coherence,” and
if you absolutely need a velocity-like display, calibrate a monotonic mapping by matching percentiles to a trusted estimator in a high-quality ROI (e.g., map a vessel’s 10–90% range to the 10–90% of a velocity-like estimator).
Compute considerations for real-time use
FFT-based per-pixel spectra (J–P) are typically manageable when (T) is modest, but become heavy if you do long STFTs (Z).
SVD/HOSVD (U–W) can dominate runtime; many papers focus specifically on accelerating SVD clutter filtering (randomized SVD, simplified forms) because of this bottleneck. 
Coherence-based improvements (Q, AA, AE) tend to be cheap relative to SVD and can give large usability gains when deep noise is the limiting factor. 
Transcranial-specific failure mode checklist for evaluating any metric
Even if you don’t “solve” these in the metric itself, it helps to know what you’re measuring:

Skull attenuation / aberration lowers coherence and can bias phase-based estimators. 
Reverberation creates structured false patterns and can inflate variance/TV-like metrics. 
Motion (probe/head/brain pulsation) leaks into “flow” unless clutter suppression and/or motion correction are strong; wall filters remove low-frequency motion but also remove low-velocity flow. 
Aliasing / Nyquist: any frequency/velocity estimator (especially peak/centroid) can fold under insufficient PRF. 
If you want a tight “try-first” shortlist (opinionated): in transcranial work I’d usually start with G/H + AE (coherence gating), then Q (TMAS multi-lag coherence), then B (multi-lag phase slope), and only then invest in heavier local SVD or STFT hemodynamic indices if you have stable long recordings. This ordering matches what tends to survive low SNR, skull variability, and motion better than a single-lag “pure velocity” map.
"""
