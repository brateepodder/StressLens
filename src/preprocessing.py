"""
preprocessing.py  –  StressLens E4 preprocessing pipeline
===========================================================
Accepts four Streamlit UploadedFile objects (ACC, BVP, EDA, TEMP),
replicates the exact feature-extraction logic used during model training,
runs the joblib model bundle, and returns a list of stress episodes.

Public API
----------
    episodes, feature_df = preprocessing_pipeline(acc_file, bvp_file, eda_file, temp_file)

    episodes   : list[dict]  –  each dict has keys:
                     start_iso  (str,  ISO-8601 UTC)
                     end_iso    (str,  ISO-8601 UTC)
                     start_unix (float)
                     end_unix   (float)
                     duration_sec (int)

    feature_df : pd.DataFrame  –  one row per window, includes
                     window_start_unix / window_end_unix / window_start_iso /
                     predicted_stress / stress_prob
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import butter, find_peaks, sosfiltfilt

# ── Model path ────────────────────────────────────────────────────────────────
_MODEL_PATH = Path(__file__).parent / "model" / "stress_model.joblib"

# ── Sliding-window parameters (must match training) ───────────────────────────
WINDOW_SEC        = 60.0   # feature window length in seconds
STEP_SEC          = 30.0   # hop between consecutive windows
ACC_SUBWINDOW_SEC = 5.0    # ACC is sub-windowed before averaging

# ── Episode-grouping thresholds ───────────────────────────────────────────────
GAP_TOLERANCE_SEC = 300    # 5 min of consecutive non-stress closes an episode
MIN_EPISODE_SEC   = 120    # discard episodes shorter than 2 min (noise)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 – Model loader
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading stress-detection model…")
def load_model() -> object:
    """
    Loads the joblib model bundle once per Streamlit session and caches it.

    The bundle is expected to be a dict with at least:
        bundle["model"]        – fitted sklearn-compatible estimator
        bundle["feature_cols"] – ordered list of feature column names

    If the bundle IS the estimator directly (no wrapper dict), we handle
    that too and fall back to a hard-coded column order.
    """
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at '{_MODEL_PATH}'. "
            "Place stress_model.joblib inside the model/ folder."
        )
    bundle = joblib.load(_MODEL_PATH)

    # Support both raw estimator and dict-wrapped bundle
    if isinstance(bundle, dict):
        if "model" not in bundle:
            raise KeyError("Model bundle dict must contain a 'model' key.")
        return bundle          # {"model": estimator, "feature_cols": [...]}
    else:
        return {"model": bundle, "feature_cols": None}  # raw estimator


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 – Parse Empatica E4 uploads
# ═════════════════════════════════════════════════════════════════════════════

def _parse_e4_upload(
    uploaded_file,
    expected_cols: list[str],
) -> tuple[float, float, np.ndarray]:
    """
    Parses a single Empatica E4 file.

    E4 CSV format:
        Row 0  → Unix start timestamp  (single value)
        Row 1  → Sampling frequency Hz (single value)
        Row 2+ → Signal data

    Returns
    -------
    start_ts : float  –  Unix timestamp of first sample
    fs       : float  –  Sampling frequency in Hz
    data     : np.ndarray shape (n_samples, n_cols)
    """
    uploaded_file.seek(0)
    raw = pd.read_csv(uploaded_file, header=None)

    if len(raw) < 3:
        raise ValueError(
            f"{uploaded_file.name} has fewer than 3 rows — not a valid E4 CSV."
        )

    start_ts = float(raw.iloc[0, 0])
    fs       = float(raw.iloc[1, 0])

    if fs <= 0:
        raise ValueError(
            f"{uploaded_file.name}: invalid sampling frequency {fs}."
        )

    data_df = (
        raw.iloc[2:]
        .dropna(axis=1, how="all")
        .reset_index(drop=True)
        .apply(pd.to_numeric, errors="coerce")
        .ffill()
        .bfill()
    )

    if data_df.empty:
        raise ValueError(f"{uploaded_file.name}: no data rows found after header.")

    data = data_df.to_numpy(dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Validate column count
    if data.shape[1] < len(expected_cols):
        raise ValueError(
            f"{uploaded_file.name}: expected {len(expected_cols)} data column(s) "
            f"({expected_cols}), got {data.shape[1]}."
        )

    return start_ts, fs, data


def parse_e4_uploads(
    acc_file, bvp_file, eda_file, temp_file
) -> dict[str, tuple[float, float, np.ndarray]]:
    """
    Parses all four E4 uploads and returns a dict keyed by signal name.

    Each value is a (start_ts, fs, data_array) tuple.
    ACC data has shape (n, 3); all others (n, 1).
    """
    signals = {
        "ACC":  (_parse_e4_upload(acc_file,  ["x", "y", "z"]), 3),
        "BVP":  (_parse_e4_upload(bvp_file,  ["bvp"]),         1),
        "EDA":  (_parse_e4_upload(eda_file,  ["eda"]),         1),
        "TEMP": (_parse_e4_upload(temp_file, ["temp"]),        1),
    }

    parsed: dict[str, tuple[float, float, np.ndarray]] = {}
    for key, ((start_ts, fs, data), ncols) in signals.items():
        parsed[key] = (start_ts, fs, data[:, :ncols])

    return parsed


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 – Signal preprocessing (filtering + decomposition)
# ═════════════════════════════════════════════════════════════════════════════

def _butterworth(
    signal: np.ndarray,
    cutoff,
    fs: float,
    btype: str,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth filter (forward + backward pass via sosfiltfilt).

    cutoff : float for low/high-pass, [low, high] for band-pass
    btype  : "low" | "high" | "band"
    """
    nyq = fs / 2.0
    if isinstance(cutoff, (list, tuple)):
        norm = [min(max(c / nyq, 1e-6), 0.9999) for c in cutoff]
    else:
        norm = min(max(cutoff / nyq, 1e-6), 0.9999)
    sos = butter(order, norm, btype=btype, output="sos")
    return sosfiltfilt(sos, signal)


def _decompose_eda(
    eda: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters EDA then decomposes it into tonic (SCL) and phasic (SCR) components.

    1. Low-pass filter at 5 Hz removes motion/electrical noise.
    2. Try neurokit2 for research-grade decomposition.
    3. Fall back to a very-low-pass filter (0.05 Hz) for the tonic baseline;
       phasic = filtered_eda − tonic.

    Returns
    -------
    eda_filt : np.ndarray  –  noise-filtered EDA
    scl      : np.ndarray  –  tonic component (slow arousal baseline)
    scr      : np.ndarray  –  phasic component (fast sympathetic bursts)
    """
    cutoff   = min(5.0, fs / 2.0 * 0.99)
    eda_filt = _butterworth(eda, cutoff, fs, "low")

    try:
        import neurokit2 as nk
        decomp = nk.eda_phasic(eda_filt, sampling_rate=int(fs))
        scl = decomp["EDA_Tonic"].to_numpy(dtype=float)
        scr = decomp["EDA_Phasic"].to_numpy(dtype=float)
    except Exception:
        # Fallback: very-low-pass for tonic, residual for phasic
        scl_cutoff = max(min(0.05, fs / 2.0 * 0.99), 1e-3)
        scl        = _butterworth(eda_filt, scl_cutoff, fs, "low")
        scr        = eda_filt - scl

    return eda_filt, scl, scr


def _filter_bvp(
    bvp: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Band-pass filters BVP to the cardiac range (0.5–8 Hz) to remove baseline
    wander and high-frequency noise, then detects all heartbeat peaks globally.

    Peaks are found once on the full signal so that IBI sequences are not
    broken at window boundaries.

    Returns
    -------
    bvp_filt       : np.ndarray  –  filtered BVP signal
    bvp_peaks_all  : np.ndarray  –  sample indices of detected peaks
    """
    bvp_filt = _butterworth(bvp, [0.5, 8.0], fs, "band")
    std      = float(np.std(bvp_filt))
    min_dist = max(1, int(0.35 * fs))        # minimum ~350 ms between beats
    prominence = max(0.02, 0.1 * std)
    peaks, _  = find_peaks(bvp_filt, distance=min_dist, prominence=prominence)
    return bvp_filt, peaks


def preprocess_signals(
    parsed: dict[str, tuple[float, float, np.ndarray]]
) -> dict:
    """
    Runs all signal-level preprocessing and returns a dict of processed arrays
    ready for windowed feature extraction.

    Keys in returned dict
    ---------------------
    acc_start, acc_fs, acc          – raw ACC (n,3), no filtering needed
    bvp_start, bvp_fs, bvp_filt, bvp_peaks_all
    eda_start, eda_fs, eda_filt, scl, scr
    temp_start, temp_fs, temp
    session_start, session_end      – overlapping Unix time range
    """
    acc_start,  acc_fs,  acc  = parsed["ACC"]
    bvp_start,  bvp_fs,  bvp  = parsed["BVP"]
    eda_start,  eda_fs,  eda  = parsed["EDA"]
    temp_start, temp_fs, temp = parsed["TEMP"]

    # Flatten 1-D signals (shape (n,1) → (n,))
    bvp  = bvp[:, 0].astype(float)
    eda  = eda[:, 0].astype(float)
    temp = temp[:, 0].astype(float)
    acc  = acc[:, :3].astype(float)

    # Filter / decompose
    eda_filt, scl, scr        = _decompose_eda(eda, eda_fs)
    bvp_filt, bvp_peaks_all   = _filter_bvp(bvp, bvp_fs)

    # Compute the overlapping time window shared by all four signals
    session_start = max(acc_start,  bvp_start,  eda_start,  temp_start)
    session_end   = min(
        acc_start  + len(acc)  / acc_fs,
        bvp_start  + len(bvp)  / bvp_fs,
        eda_start  + len(eda)  / eda_fs,
        temp_start + len(temp) / temp_fs,
    )

    if session_end - session_start < WINDOW_SEC:
        raise ValueError(
            f"The overlapping recording duration "
            f"({session_end - session_start:.1f} s) is shorter than one "
            f"feature window ({WINDOW_SEC} s). Check that all four files are "
            f"from the same recording session."
        )

    return dict(
        acc_start=acc_start,   acc_fs=acc_fs,   acc=acc,
        bvp_start=bvp_start,   bvp_fs=bvp_fs,
        bvp_filt=bvp_filt,     bvp_peaks_all=bvp_peaks_all,
        eda_start=eda_start,   eda_fs=eda_fs,
        eda_filt=eda_filt,     scl=scl,         scr=scr,
        temp_start=temp_start, temp_fs=temp_fs, temp=temp,
        session_start=session_start,
        session_end=session_end,
    )


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 – Per-window feature extraction helpers
# ═════════════════════════════════════════════════════════════════════════════

def _slice_window(
    data: np.ndarray,
    signal_start: float,
    fs: float,
    win_start: float,
    win_end: float,
) -> np.ndarray:
    """
    Slices a numpy array by Unix time range rather than array index.
    Handles the fact that each E4 signal has its own start timestamp and fs.
    """
    i0 = max(0,          int(np.floor((win_start - signal_start) * fs)))
    i1 = min(len(data),  int(np.floor((win_end   - signal_start) * fs)))
    return data[i0:i1]


def _extract_eda_features(
    eda_win: np.ndarray,
    scl_win: np.ndarray,
    scr_win: np.ndarray,
    fs: float,
) -> dict[str, float]:
    """
    13 features covering the full EDA signal, its tonic baseline (SCL),
    and its phasic sympathetic bursts (SCR).
    """
    feats: dict[str, float] = {}

    # Raw EDA statistics
    feats["EDA_mean"]  = float(np.mean(eda_win))
    feats["EDA_std"]   = float(np.std(eda_win))
    feats["EDA_min"]   = float(np.min(eda_win))
    feats["EDA_max"]   = float(np.max(eda_win))
    feats["EDA_range"] = float(feats["EDA_max"] - feats["EDA_min"])
    feats["EDA_slope"] = float(np.polyfit(np.arange(len(eda_win)), eda_win, 1)[0])

    # Tonic (SCL) – slow baseline arousal
    feats["SCL_mean"]     = float(np.mean(scl_win))
    feats["SCL_std"]      = float(np.std(scl_win))
    corr = np.corrcoef(scl_win, np.arange(len(scl_win)))
    feats["SCL_timecorr"] = float(corr[0, 1]) if np.isfinite(corr[0, 1]) else 0.0

    # Phasic (SCR) – fast sympathetic bursts, most discriminative for stress
    feats["SCR_std"] = float(np.std(scr_win))
    min_h = max(0.01, 0.01 * float(np.std(scr_win)))
    min_d = max(1, int(fs))
    peaks, props = find_peaks(scr_win, height=min_h, distance=min_d)
    feats["SCR_nPeaks"] = float(len(peaks))
    if len(peaks) > 0:
        feats["SCR_sumAmplitude"] = float(np.sum(props["peak_heights"]))
        feats["SCR_area"]         = float(np.trapezoid(np.abs(scr_win)))
    else:
        feats["SCR_sumAmplitude"] = 0.0
        feats["SCR_area"]         = 0.0

    return feats


def _extract_temp_features(temp_win: np.ndarray) -> dict[str, float]:
    """
    6 temperature features.
    Stress → vasoconstriction → peripheral temperature drop.
    The slope is particularly useful (negative slope during window = stress signal).
    """
    return {
        "TEMP_mean":  float(np.mean(temp_win)),
        "TEMP_std":   float(np.std(temp_win)),
        "TEMP_min":   float(np.min(temp_win)),
        "TEMP_max":   float(np.max(temp_win)),
        "TEMP_range": float(np.ptp(temp_win)),
        "TEMP_slope": float(np.polyfit(np.arange(len(temp_win)), temp_win, 1)[0]),
    }


def _extract_bvp_features(
    bvp_win: np.ndarray,
    local_peaks: np.ndarray,
    fs: float,
) -> dict[str, float]:
    """
    17 BVP + PRV (Pulse Rate Variability) features.

    PRV is the beat-to-beat variability derived from BVP peaks.
    Low HRV (↓ RMSSD, ↓ pNN50) = sympathetic dominance = stress.
    These are clinically critical for cardiac rehab patients.
    """
    feats: dict[str, float] = {}

    # Signal statistics
    feats["BVP_mean"]  = float(np.mean(bvp_win))
    feats["BVP_std"]   = float(np.std(bvp_win))
    feats["BVP_min"]   = float(np.min(bvp_win))
    feats["BVP_max"]   = float(np.max(bvp_win))
    feats["BVP_range"] = float(feats["BVP_max"] - feats["BVP_min"])
    feats["BVP_slope"] = float(np.polyfit(np.arange(len(bvp_win)), bvp_win, 1)[0])

    # Peak-derived features
    n_peaks      = len(local_peaks)
    duration_sec = len(bvp_win) / fs
    feats["BVP_nPeaks"]       = float(n_peaks)
    feats["BVP_peakRate"]     = float(60.0 * n_peaks / duration_sec) if duration_sec > 0 else 0.0

    if n_peaks > 0:
        peak_amp = bvp_win[local_peaks]
        feats["BVP_peakAmp_mean"] = float(np.mean(peak_amp))
        feats["BVP_peakAmp_std"]  = float(np.std(peak_amp))
    else:
        feats["BVP_peakAmp_mean"] = 0.0
        feats["BVP_peakAmp_std"]  = 0.0

    # PRV / HRV features – require at least 3 peaks for meaningful IBI series
    if n_peaks >= 3:
        ibi_ms   = np.diff(local_peaks) / fs * 1000.0   # inter-beat intervals in ms
        d_ibi_ms = np.diff(ibi_ms)                       # successive differences

        feats["BVP_PRV_IBI_mean"] = float(np.mean(ibi_ms))
        feats["BVP_PRV_IBI_std"]  = float(np.std(ibi_ms))
        feats["BVP_PRV_SDNN"]     = float(np.std(ibi_ms))
        feats["BVP_PRV_RMSSD"]    = float(np.sqrt(np.mean(d_ibi_ms**2))) if len(d_ibi_ms) else np.nan
        feats["BVP_PRV_pNN50"]    = float(np.mean(np.abs(d_ibi_ms) > 50.0) * 100.0) if len(d_ibi_ms) else np.nan

        bpm = 60000.0 / ibi_ms
        feats["BVP_HR_mean"] = float(np.mean(bpm))
        feats["BVP_HR_std"]  = float(np.std(bpm))
    else:
        for key in ["BVP_PRV_IBI_mean", "BVP_PRV_IBI_std", "BVP_PRV_SDNN",
                    "BVP_PRV_RMSSD", "BVP_PRV_pNN50", "BVP_HR_mean", "BVP_HR_std"]:
            feats[key] = np.nan

    return feats


def _extract_acc_features(
    acc_seg: np.ndarray,
    fs: float,
) -> dict[str, float]:
    """
    16 ACC features: 4 per axis (X, Y, Z) + 4 for the 3D magnitude vector.

    Features per axis/magnitude:
        mean       – average acceleration (gravity + motion)
        std        – variability
        absint     – absolute integral (total energy), normalised by length
        peakfreq   – dominant movement frequency via FFT
    """
    feats: dict[str, float] = {}

    for i, axis in enumerate(["x", "y", "z"]):
        col = acc_seg[:, i].astype(float)
        feats[f"ACC_mean_{axis}"]    = float(np.mean(col))
        feats[f"ACC_std_{axis}"]     = float(np.std(col))
        feats[f"ACC_absint_{axis}"]  = float(np.trapezoid(np.abs(col)) / max(len(col), 1))
        col_detrend = col - feats[f"ACC_mean_{axis}"]
        fft_vals    = np.abs(np.fft.rfft(col_detrend))
        freqs       = np.fft.rfftfreq(len(col_detrend), 1.0 / fs)
        feats[f"ACC_peakfreq_{axis}"] = float(freqs[np.argmax(fft_vals)]) if len(fft_vals) > 1 else 0.0

    mag = np.linalg.norm(acc_seg.astype(float), axis=1)
    feats["ACC_mean_3D"]    = float(np.mean(mag))
    feats["ACC_std_3D"]     = float(np.std(mag))
    feats["ACC_absint_3D"]  = float(np.trapezoid(np.abs(mag)) / max(len(mag), 1))
    mag_detrend = mag - feats["ACC_mean_3D"]
    fft_mag     = np.abs(np.fft.rfft(mag_detrend))
    freqs_3d    = np.fft.rfftfreq(len(mag_detrend), 1.0 / fs)
    feats["ACC_peakfreq_3D"] = float(freqs_3d[np.argmax(fft_mag)]) if len(fft_mag) > 1 else 0.0

    return feats


def _aggregate_acc_window(
    acc_win: np.ndarray,
    fs_acc: float,
) -> dict[str, float]:
    """
    ACC is at 32 Hz — much higher resolution than EDA (4 Hz).
    Rather than extracting features over the whole 60 s block, we chop it into
    ACC_SUBWINDOW_SEC (5 s) sub-windows, extract features from each, then
    average them. This gives more stable, less noise-sensitive estimates.
    """
    sub_len = max(1, int(round(ACC_SUBWINDOW_SEC * fs_acc)))
    sub_feats: list[dict[str, float]] = []

    for start in range(0, len(acc_win) - sub_len + 1, sub_len):
        sub_feats.append(_extract_acc_features(acc_win[start: start + sub_len], fs_acc))

    if not sub_feats:
        # Window shorter than one sub-window: extract features on whatever we have
        return _extract_acc_features(acc_win, fs_acc) if len(acc_win) > 1 else {
            k: np.nan for k in [
                "ACC_mean_x","ACC_std_x","ACC_absint_x","ACC_peakfreq_x",
                "ACC_mean_y","ACC_std_y","ACC_absint_y","ACC_peakfreq_y",
                "ACC_mean_z","ACC_std_z","ACC_absint_z","ACC_peakfreq_z",
                "ACC_mean_3D","ACC_std_3D","ACC_absint_3D","ACC_peakfreq_3D",
            ]
        }

    acc_df = pd.DataFrame(sub_feats)
    return {col: float(acc_df[col].mean()) for col in acc_df.columns}


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 – Sliding window loop → feature DataFrame
# ═════════════════════════════════════════════════════════════════════════════

def extract_all_windows(preprocessed: dict) -> pd.DataFrame:
    """
    Slides a WINDOW_SEC window forward by STEP_SEC over the shared recording
    duration, computes all features for each window, and returns a DataFrame
    with one row per window.

    Minimum window sizes enforce that we never pass a truncated segment to
    the feature extractors.
    """
    acc_start = preprocessed["acc_start"];  acc_fs = preprocessed["acc_fs"]
    bvp_start = preprocessed["bvp_start"];  bvp_fs = preprocessed["bvp_fs"]
    eda_start = preprocessed["eda_start"];  eda_fs = preprocessed["eda_fs"]
    temp_start= preprocessed["temp_start"]; temp_fs= preprocessed["temp_fs"]

    acc          = preprocessed["acc"]
    bvp_filt     = preprocessed["bvp_filt"]
    bvp_peaks_all= preprocessed["bvp_peaks_all"]
    eda_filt     = preprocessed["eda_filt"]
    scl          = preprocessed["scl"]
    scr          = preprocessed["scr"]
    temp         = preprocessed["temp"]

    session_start = preprocessed["session_start"]
    session_end   = preprocessed["session_end"]

    # Minimum samples required per window per signal
    # (allow up to 10 % fewer samples to handle rounding at boundaries)
    min_acc_len  = max(1, int(round(WINDOW_SEC * acc_fs))  - int(round(acc_fs  * 0.1)))
    min_bvp_len  = max(1, int(round(WINDOW_SEC * bvp_fs))  - int(round(bvp_fs  * 0.1)))
    min_eda_len  = max(1, int(round(WINDOW_SEC * eda_fs))  - int(round(eda_fs  * 0.1)))
    min_temp_len = max(1, int(round(WINDOW_SEC * temp_fs)) - int(round(temp_fs * 0.1)))

    rows: list[dict] = []
    window_starts = np.arange(session_start, session_end - WINDOW_SEC + 1e-9, STEP_SEC)

    for win_start in window_starts:
        win_end = win_start + WINDOW_SEC

        # Slice each signal to the current window
        acc_win  = _slice_window(acc,      acc_start,  acc_fs,  win_start, win_end)
        bvp_win  = _slice_window(bvp_filt, bvp_start,  bvp_fs,  win_start, win_end).flatten()
        eda_win  = _slice_window(eda_filt, eda_start,  eda_fs,  win_start, win_end).flatten()
        scl_win  = _slice_window(scl,      eda_start,  eda_fs,  win_start, win_end).flatten()
        scr_win  = _slice_window(scr,      eda_start,  eda_fs,  win_start, win_end).flatten()
        temp_win = _slice_window(temp,     temp_start, temp_fs, win_start, win_end).flatten()

        # Skip windows with insufficient data (can happen at the very end)
        if (
            len(acc_win)  < min_acc_len
            or len(bvp_win)  < min_bvp_len
            or len(eda_win)  < min_eda_len
            or len(scl_win)  < min_eda_len
            or len(scr_win)  < min_eda_len
            or len(temp_win) < min_temp_len
        ):
            continue

        # Remap global BVP peak indices to be local to this window
        peak_i0     = max(0, int(np.floor((win_start - bvp_start) * bvp_fs)))
        peak_i1     = min(len(bvp_filt), int(np.floor((win_end - bvp_start) * bvp_fs)))
        local_peaks = (
            bvp_peaks_all[(bvp_peaks_all >= peak_i0) & (bvp_peaks_all < peak_i1)] - peak_i0
        )

        # Build feature row
        feats: dict = {}
        feats.update(_extract_bvp_features(bvp_win,  local_peaks, bvp_fs))
        feats.update(_extract_eda_features(eda_win, scl_win, scr_win, eda_fs))
        feats.update(_extract_temp_features(temp_win))
        feats.update(_aggregate_acc_window(acc_win, acc_fs))

        # Attach window metadata (not fed to model, used for episode grouping)
        feats["window_start_unix"] = float(win_start)
        feats["window_end_unix"]   = float(win_end)
        feats["window_start_iso"]  = datetime.fromtimestamp(win_start, tz=timezone.utc).isoformat()

        rows.append(feats)

    if not rows:
        raise ValueError(
            "No complete windows could be extracted. "
            "Ensure all four files are from the same recording session and "
            f"the recording is at least {WINDOW_SEC} seconds long."
        )

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 – Run the ML model
# ═════════════════════════════════════════════════════════════════════════════

def run_model(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads the model bundle, aligns feature columns to the training schema,
    runs prediction, and appends predicted_stress + stress_prob columns.

    The model bundle may contain a "feature_cols" list that defines the exact
    column order used during training. If present, we reindex to that order
    (filling any missing columns with NaN — a signal to investigate the
    feature extractor if this happens).

    Returns the feature_df with two new columns appended.
    """
    bundle = load_model()
    model  = bundle["model"]
    feature_cols: list[str] | None = bundle.get("feature_cols")

    # Metadata columns that must not be passed to the model
    meta_cols = {"window_start_unix", "window_end_unix", "window_start_iso"}

    if feature_cols:
        # Enforce exact training-time column order
        missing = [c for c in feature_cols if c not in feature_df.columns]
        if missing:
            raise ValueError(
                f"Feature mismatch: the following columns expected by the model "
                f"are absent from the extracted features: {missing}.\n"
                f"Check that the preprocessing parameters match training."
            )
        X = feature_df[feature_cols].to_numpy(dtype=float)
    else:
        # No schema in bundle — use all non-metadata columns in existing order
        model_cols = [c for c in feature_df.columns if c not in meta_cols]
        X = feature_df[model_cols].to_numpy(dtype=float)

    # Replace any remaining NaN with column medians (same strategy as training)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask    = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    predictions = model.predict(X)

    try:
        probabilities = model.predict_proba(X)[:, 1]
    except AttributeError:
        # Model does not support predict_proba (e.g. SVM without probability=True)
        probabilities = predictions.astype(float)

    result_df = feature_df.copy()
    result_df["predicted_stress"] = predictions
    result_df["stress_prob"]      = probabilities

    return result_df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 – Group windows into stress episodes
# ═════════════════════════════════════════════════════════════════════════════

def group_stress_episodes(result_df: pd.DataFrame) -> list[dict]:
    """
    Converts per-window binary predictions into discrete stress episodes.

    Logic
    -----
    - A new episode begins when a stressed window is encountered.
    - A non-stressed window starts a gap counter (in seconds).
    - Once the gap reaches GAP_TOLERANCE_SEC (5 min), the episode is closed.
    - Episodes shorter than MIN_EPISODE_SEC (2 min) are discarded as noise.
    - An episode still open at the end of the recording is closed automatically.

    Each episode dict contains
    --------------------------
    start_iso    : str    ISO-8601 UTC timestamp
    end_iso      : str    ISO-8601 UTC timestamp
    start_unix   : float  Unix timestamp
    end_unix     : float  Unix timestamp
    duration_sec : int    episode length in seconds
    """
    episodes: list[dict] = []

    in_episode    = False
    gap_sec       = 0.0
    episode_start_unix: float | None = None
    episode_start_iso:  str   | None = None
    last_stressed_unix: float | None = None
    last_stressed_iso:  str   | None = None

    for _, row in result_df.iterrows():
        win_start_unix = float(row["window_start_unix"])
        win_start_iso  = str(row["window_start_iso"])
        win_end_unix   = float(row["window_end_unix"])
        win_end_iso    = datetime.fromtimestamp(win_end_unix, tz=timezone.utc).isoformat()
        is_stressed    = int(row["predicted_stress"]) == 1

        if is_stressed:
            if not in_episode:
                # Start a new episode
                episode_start_unix = win_start_unix
                episode_start_iso  = win_start_iso
                in_episode         = True

            # Reset gap counter and record furthest stressed window seen
            gap_sec            = 0.0
            last_stressed_unix = win_end_unix
            last_stressed_iso  = win_end_iso

        else:
            if in_episode:
                gap_sec += STEP_SEC   # each non-stressed window = one step forward

                if gap_sec >= GAP_TOLERANCE_SEC:
                    # Gap exceeded threshold → close the episode
                    duration_sec = int(last_stressed_unix - episode_start_unix)
                    if duration_sec >= MIN_EPISODE_SEC:
                        episodes.append({
                            "start_iso":    episode_start_iso,
                            "end_iso":      last_stressed_iso,
                            "start_unix":   episode_start_unix,
                            "end_unix":     last_stressed_unix,
                            "duration_sec": duration_sec,
                        })
                    in_episode = False
                    gap_sec    = 0.0

    # Close any episode still open at the end of the recording
    if in_episode and episode_start_unix is not None and last_stressed_unix is not None:
        duration_sec = int(last_stressed_unix - episode_start_unix)
        if duration_sec >= MIN_EPISODE_SEC:
            episodes.append({
                "start_iso":    episode_start_iso,
                "end_iso":      last_stressed_iso,
                "start_unix":   episode_start_unix,
                "end_unix":     last_stressed_unix,
                "duration_sec": duration_sec,
            })

    return episodes


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def preprocessing_pipeline(
    acc_file,
    bvp_file,
    eda_file,
    temp_file,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Full end-to-end pipeline.

    Parameters
    ----------
    acc_file, bvp_file, eda_file, temp_file
        Streamlit UploadedFile objects (or any file-like object readable by
        pd.read_csv).

    Returns
    -------
    episodes   : list[dict]     – stress episodes with start/end/duration
    result_df  : pd.DataFrame   – per-window features + predictions + probs
    """
    # 1. Parse raw E4 uploads into (start_ts, fs, ndarray) tuples
    parsed = parse_e4_uploads(acc_file, bvp_file, eda_file, temp_file)

    # 2. Filter signals, decompose EDA, detect BVP peaks
    preprocessed = preprocess_signals(parsed)

    # 3. Slide window and extract ~47 features per window
    feature_df = extract_all_windows(preprocessed)

    # 4. Run ML model → appends predicted_stress + stress_prob columns
    result_df = run_model(feature_df)

    # 5. Group stressed windows into discrete episodes
    episodes = group_stress_episodes(result_df)

    return episodes, result_df