"""
tap_core.py
Core logic for tap onset detection (Fujii 10% method + amplitude filtering).

- Butterworth HPF + filtfilt (zero-phase)
- Hilbert envelope with configurable smoothing window (ms)
- Envelope-peak based filtering (threshold_ratio)
- Waveform-amplitude based filtering (amp_threshold_ratio)
- Fujii backward 10% for tap_start
"""

from __future__ import annotations
from typing import List, Dict

import numpy as np
import soundfile as sf
import scipy.signal

from .tap_utils import butter_highpass_zero_phase, hilbert_envelope


FUJII_RATIO = 0.1  # 各ピークに対する Fujii backward 10%


def detect_tap_onsets_and_peaks(
    y: np.ndarray,
    sr: int,
    *,
    hp_cutoff: float = 300.0,
    threshold_ratio: float = 0.1,      # envelope フィルタ (env >= threshold_ratio * env_max)
    amp_threshold_ratio: float = 0.03, # waveform フィルタ (abs(y) >= amp_threshold_ratio * max_abs)
    min_distance_ms: float = 100.0,
    smooth_ms: float = 0.3,            # Hilbert envelope smoothing window [ms]
) -> List[Dict[str, float]]:
    """
    Tap detection using Fujii 10% method with two-stage filtering.

    Args:
        y: waveform (mono)
        sr: sampling rate
        hp_cutoff: HPF cutoff [Hz]
        threshold_ratio: envelope peak filter (ratio of global env_max)
        amp_threshold_ratio: waveform amplitude filter (ratio of max_abs)
        min_distance_ms: minimum distance between peaks [ms]
        smooth_ms: smoothing window for Hilbert envelope [ms]

    Returns:
        List of dict with keys:
            tap_start, tap_peak, hp_cutoff, threshold, amp_threshold, smooth_ms
    """
    # HPF
    y_filt = butter_highpass_zero_phase(y, sr, hp_cutoff)

    # Envelope
    env = hilbert_envelope(y_filt, sr, smooth_ms=smooth_ms)

    if len(env) == 0:
        return []

    env_max = float(np.max(env))
    max_amp = float(np.max(np.abs(y_filt)))
    if env_max <= 0.0 or max_amp <= 0.0:
        return []

    # Envelope-based peak filtering
    env_min_height = threshold_ratio * env_max
    min_distance = int(min_distance_ms * 1e-3 * sr)

    peak_idx, _ = scipy.signal.find_peaks(
        env,
        distance=min_distance,
        height=env_min_height,
    )

    results: List[Dict[str, float]] = []

    for p in peak_idx:
        # Waveform amplitude filter
        if abs(y_filt[p]) < amp_threshold_ratio * max_amp:
            continue

        peak_val = env[p]
        if peak_val <= 0.0:
            continue

        # Fujii backward 10% threshold
        thr_fujii = FUJII_RATIO * peak_val

        # Backward search
        i = p
        while i > 0 and env[i] >= thr_fujii:
            i -= 1
        if i <= 0:
            continue

        # Linear interpolation for crossing
        x1, y1 = i, env[i]
        x2, y2 = i + 1, env[i + 1]
        if y2 == y1:
            tap_start = i / sr
        else:
            tap_start = (x1 + (thr_fujii - y1) / (y2 - y1)) / sr

        tap_peak = p / sr

        results.append(
            {
                "tap_start": float(tap_start),
                "tap_peak": float(tap_peak),
                "hp_cutoff": float(hp_cutoff),
                "threshold": float(threshold_ratio),
                "amp_threshold": float(amp_threshold_ratio),
                "smooth_ms": float(smooth_ms),
            }
        )

    return results


def detect_taps_from_wav(
    wav_path: str,
    *,
    hp_cutoff: float = 300.0,
    threshold_ratio: float = 0.1,
    amp_threshold_ratio: float = 0.03,
    min_distance_ms: float = 100.0,
    smooth_ms: float = 0.3,
) -> List[Dict[str, float]]:
    """
    Main API: detect taps from a WAV file.

    Returns:
        List of dict with keys:
            file_name, tap_start, tap_peak, hp_cutoff, threshold, amp_threshold, smooth_ms
    """
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    results = detect_tap_onsets_and_peaks(
        y,
        sr,
        hp_cutoff=hp_cutoff,
        threshold_ratio=threshold_ratio,
        amp_threshold_ratio=amp_threshold_ratio,
        min_distance_ms=min_distance_ms,
        smooth_ms=smooth_ms,
    )

    for r in results:
        r["file_name"] = wav_path

    return results
