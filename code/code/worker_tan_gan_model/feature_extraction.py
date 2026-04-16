"""
feature_extraction.py
=====================
Sliding-window feature extraction from ECG, GSR, and EEG signals.

Produces a feature matrix (one row per window) with 22 physiological
features that are safe under the 15.5 Hz Nyquist ceiling (7.75 Hz).
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, periodogram, find_peaks
from scipy.stats import mode as scipy_mode

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Constants ────────────────────────────────────────────────────────────────
FS             = 15.5                     # Hz
NYQUIST        = FS / 2                   # 7.75 Hz — hard ceiling
WINDOW_SECONDS = 200
WINDOW_SIZE    = int(WINDOW_SECONDS * FS)  # 3100 samples
STEP_SIZE      = int(10 * FS)             # 155 samples
HRV_CLIP       = 1000                     # ms — physiological ceiling

FEATURE_COLS = [
    "ecg_mean", "ecg_var", "ecg_std", "ecg_rms",
    "hrv_rmssd", "hrv_sdnn", "ecg_lf_hf", "ecg_slope",
    "gsr_mean", "gsr_var", "gsr_std", "gsr_slope",
    "gsr_tonic_mean", "gsr_phasic_std", "gsr_scr_amp", "gsr_peak_count",
    "eeg_mean", "eeg_var", "eeg_rms", "eeg_energy",
    "eeg_delta", "eeg_theta",
]


# ─────────────────────────────────────────────────────────────────────────────
# Signal utilities
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray,
                    lowcut: float,
                    highcut: float,
                    fs: float = FS,
                    order: int = 4) -> np.ndarray:
    """Butterworth bandpass — clamps highcut below Nyquist."""
    nyq     = 0.5 * fs
    highcut = min(highcut, nyq * 0.95)
    if lowcut >= highcut:
        return signal
    try:
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
        return filtfilt(b, a, signal)
    except Exception:
        return signal


def lowpass_filter(signal: np.ndarray,
                   cutoff: float,
                   fs: float = FS,
                   order: int = 4) -> np.ndarray:
    """Butterworth lowpass — used for GSR tonic extraction."""
    nyq    = 0.5 * fs
    cutoff = min(cutoff, nyq * 0.95)
    try:
        b, a = butter(order, cutoff / nyq, btype="low")
        return filtfilt(b, a, signal)
    except Exception:
        return signal


# ─────────────────────────────────────────────────────────────────────────────
# ECG / HRV features  (8 features)
# ─────────────────────────────────────────────────────────────────────────────

def _get_rr_intervals(ecg_window: np.ndarray, fs: float = FS) -> np.ndarray:
    """Detect R-peaks and return R-R intervals in milliseconds."""
    min_distance = int(0.4 * fs)   # ≥ 400 ms between beats (~150 BPM max)
    peaks, _     = find_peaks(
        ecg_window,
        distance=min_distance,
        height=np.mean(ecg_window) + 0.5 * np.std(ecg_window),
    )
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / fs * 1000   # → ms


def get_ecg_features(w_ecg: np.ndarray, fs: float = FS) -> dict:
    """
    8 ECG features:
      ecg_mean, ecg_var, ecg_std, ecg_rms,
      hrv_rmssd, hrv_sdnn, ecg_lf_hf, ecg_slope
    """
    rr    = _get_rr_intervals(w_ecg, fs)
    sdnn  = float(np.std(rr))                               if len(rr) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))      if len(rr) > 2 else 0.0

    # LF / HF ratio (both bands are below Nyquist)
    freqs, psd = periodogram(w_ecg, fs)
    lf    = np.sum(psd[(freqs >= 0.04) & (freqs <= 0.15)])
    hf    = np.sum(psd[(freqs >= 0.15) & (freqs <= 0.40)])
    lf_hf = float(lf / hf) if hf > 0 else 0.0

    slope = float(np.polyfit(np.arange(len(w_ecg)), w_ecg, 1)[0])

    return {
        "ecg_mean":  float(np.mean(w_ecg)),
        "ecg_var":   float(np.var(w_ecg)),
        "ecg_std":   float(np.std(w_ecg)),
        "ecg_rms":   float(np.sqrt(np.mean(w_ecg ** 2))),
        "hrv_rmssd": rmssd,
        "hrv_sdnn":  sdnn,
        "ecg_lf_hf": lf_hf,
        "ecg_slope": slope,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GSR features  (8 features)
# ─────────────────────────────────────────────────────────────────────────────

def get_gsr_features(w_gsr: np.ndarray, fs: float = FS) -> dict:
    """
    8 GSR features:
      gsr_mean, gsr_var, gsr_std, gsr_slope,
      gsr_tonic_mean, gsr_phasic_std, gsr_scr_amp, gsr_peak_count
    """
    tonic  = lowpass_filter(w_gsr, cutoff=0.05, fs=fs)
    phasic = w_gsr - tonic

    peaks, props = find_peaks(phasic, prominence=0.05)
    scr_amp   = float(np.mean(props["prominences"])) if len(peaks) > 0 else 0.0
    slope     = float(np.polyfit(np.arange(len(w_gsr)), w_gsr, 1)[0])

    return {
        "gsr_mean":       float(np.mean(w_gsr)),
        "gsr_var":        float(np.var(w_gsr)),
        "gsr_std":        float(np.std(w_gsr)),
        "gsr_slope":      slope,
        "gsr_tonic_mean": float(np.mean(tonic)),
        "gsr_phasic_std": float(np.std(phasic)),
        "gsr_scr_amp":    scr_amp,
        "gsr_peak_count": int(len(peaks)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EEG features  (6 features)
# ─────────────────────────────────────────────────────────────────────────────

def get_eeg_features(w_eeg: np.ndarray, fs: float = FS) -> dict:
    """
    6 EEG features:
      eeg_mean, eeg_var, eeg_rms, eeg_energy,
      eeg_delta (1–4 Hz), eeg_theta (4–7.5 Hz)

    Only delta and theta bands are extracted — both lie below the 7.75 Hz
    Nyquist limit at 15.5 Hz sampling rate.
    """
    freqs, psd = periodogram(w_eeg, fs)
    delta = float(np.sum(psd[(freqs >= 1.0) & (freqs <= 4.0)]))
    theta = float(np.sum(psd[(freqs >= 4.0) & (freqs <= 7.5)]))

    return {
        "eeg_mean":   float(np.mean(w_eeg)),
        "eeg_var":    float(np.var(w_eeg)),
        "eeg_rms":    float(np.sqrt(np.mean(w_eeg ** 2))),
        "eeg_energy": float(np.sum(w_eeg ** 2)),
        "eeg_delta":  delta,
        "eeg_theta":  theta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame,
                     window_size: int  = WINDOW_SIZE,
                     step_size:   int  = STEP_SIZE,
                     fs:          float = FS) -> pd.DataFrame:
    """
    Sliding-window feature extraction over all workers.

    For each window:
      • label = majority-vote over window's per-sample labels
      • ECG / GSR are bandpass-filtered before windowing
      • EEG is used raw (band extraction per window via periodogram)

    Parameters
    ----------
    df          : labelled DataFrame (output of data_loader)
    window_size : number of samples per window  (default 3100, ~200 s)
    step_size   : hop size in samples           (default 155,  ~10 s)

    Returns
    -------
    feature_matrix : DataFrame, one row per window
    """
    for col in ["ECG", "GSR", "EEG", "label", "worker"]:
        assert col in df.columns, f"Missing column: {col}"

    features_list = []

    for worker_id, grp in df.groupby("worker"):
        grp = grp.reset_index(drop=True)

        # Pre-filter full signal to avoid window-edge artefacts
        sig_ecg = bandpass_filter(grp["ECG"].values, 0.5, 5.0, fs)
        sig_gsr = bandpass_filter(grp["GSR"].values, 0.01, 2.0, fs)
        sig_eeg = grp["EEG"].values   # no pre-filter; extracted per-window
        labels  = grp["label"].values

        n_windows = 0
        for start in range(0, len(grp) - window_size, step_size):
            end     = start + window_size
            w_label = labels[start:end]

            # Majority-vote label for the window
            label_mode = int(pd.Series(w_label).mode()[0])

            row = {"worker": worker_id, "label": label_mode}
            row.update(get_ecg_features(sig_ecg[start:end], fs))
            row.update(get_gsr_features(sig_gsr[start:end], fs))
            row.update(get_eeg_features(sig_eeg[start:end], fs))
            features_list.append(row)
            n_windows += 1

        print(f"  Worker {worker_id:2d}: {n_windows} windows extracted")

    feature_matrix = pd.DataFrame(features_list)

    # Clip physiologically impossible HRV values
    feature_matrix["hrv_rmssd"] = feature_matrix["hrv_rmssd"].clip(upper=HRV_CLIP)
    feature_matrix["hrv_sdnn"]  = feature_matrix["hrv_sdnn"].clip(upper=HRV_CLIP)

    print(f"\n[feature_extraction] Feature matrix shape : {feature_matrix.shape}")
    print(f"[feature_extraction] Features per window  : "
          f"{feature_matrix.shape[1] - 2}  (excl. worker, label)")
    print(f"\n[feature_extraction] Windowed class balance:")
    print(
        feature_matrix["label"]
        .value_counts()
        .sort_index()
        .rename({0: "0-Relaxed", 1: "1-Medium", 2: "2-Fatigue"})
    )
    return feature_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Quality checks
# ─────────────────────────────────────────────────────────────────────────────

def quality_check(feature_matrix: pd.DataFrame) -> None:
    """Print NaN counts and columns with > 10 % zeros."""
    print("\n[feature_extraction] NaN check:")
    print(feature_matrix.isna().sum())

    zero_pct = (feature_matrix == 0).mean()
    suspects = zero_pct[zero_pct > 0.10]
    if not suspects.empty:
        print("\n[feature_extraction] Columns with > 10 % zeros (possible dead signal):")
        print(suspects)

    print("\n[feature_extraction] HRV summary (post-clip):")
    print(feature_matrix[["hrv_rmssd", "hrv_sdnn"]].describe().round(2))


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Worker Fatigue — Feature Extraction"
    )
    parser.add_argument("--input",  required=True,
                        help="Path to loaded_data.csv (output of data_loader.py)")
    parser.add_argument("--output", default="outputs/feature_matrix.csv",
                        help="Where to save the feature matrix")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    data = pd.read_csv(args.input)
    print(f"[feature_extraction] Loaded {len(data):,} rows from {args.input}")

    print(f"\n[feature_extraction] Window size : {WINDOW_SIZE} samples ({WINDOW_SECONDS} s)")
    print(f"[feature_extraction] Step size   : {STEP_SIZE}  samples ({STEP_SIZE/FS:.1f} s)")

    fm = extract_features(data)
    quality_check(fm)
    fm.to_csv(args.output, index=False)
    print(f"\n[feature_extraction] Saved → {args.output}")
