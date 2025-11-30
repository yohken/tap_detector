"""
tap_utils.py
Basic DSP utilities for tap detection.

- Butterworth HPF + filtfilt (zero-phase)
- Hilbert envelope calculation
- Envelope smoothing (ms単位)
"""

from __future__ import annotations
import numpy as np
import scipy.signal


# ============================================================
# High-pass filter (Butterworth + filtfilt, zero-phase)
# ============================================================

def butter_highpass_zero_phase(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply Butterworth high-pass filter with zero-phase (filtfilt).
    
    Args:
        y: audio signal
        sr: sampling rate
        cutoff_hz: HPF cutoff frequency
        order: butterworth filter order

    Returns:
        y_filt: filtered signal (zero-phase)
    """
    if cutoff_hz <= 0:
        return y.astype(float)

    nyq = sr / 2.0
    norm = cutoff_hz / nyq
    b, a = scipy.signal.butter(order, norm, btype="high")
    y_filt = scipy.signal.filtfilt(b, a, y).astype(float)
    return y_filt


# ============================================================
# Hilbert envelope + smoothing
# ============================================================

def hilbert_envelope(y: np.ndarray, sr: int, smooth_ms: float = 0.3) -> np.ndarray:
    """
    Compute Hilbert envelope with smoothing.
    
    Args:
        y: filtered audio
        sr: sampling rate
        smooth_ms: smoothing window (ms)
    
    Returns:
        env_smooth: smoothed envelope
    """
    analytic = scipy.signal.hilbert(y)
    env = np.abs(analytic)

    if smooth_ms <= 0:
        return env.astype(float)

    win_len = max(1, int(smooth_ms * 1e-3 * sr))
    window = np.ones(win_len) / win_len
    env_smooth = np.convolve(env, window, mode="same")
    return env_smooth.astype(float)
