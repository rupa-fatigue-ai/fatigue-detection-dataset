"""
data_loader.py
==============
Loads raw per-worker CSV files and the compiled workers_dataset.csv,
applies marker-based fatigue labels (NSP / MSP / HSP), imputes missing
values, and returns a clean, labelled DataFrame ready for feature extraction.
"""

import os
import argparse
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Default paths (override via CLI or config) ───────────────────────────────
DEFAULT_DATASET_CSV = "dataset/workers_dataset.csv"
DEFAULT_MARKER_CSV  = "dataset/marker_info.csv"

# ── Sampling rate ────────────────────────────────────────────────────────────
FS = 15.5   # Hz

# ── Period → fatigue label mapping ───────────────────────────────────────────
PERIOD_LABELS = {
    "NSP1": 0,   # Non-fatigued
    "HSP1": 2,   # High-fatigued
    "MSP1": 1,   # Medium-fatigued
    "HSP2": 2,
    "MSP2": 1,
    "HSP3": 2,
    "NSP2": 0,
}
PERIOD_COLS = list(PERIOD_LABELS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & normalise the compiled CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_compiled_csv(csv_path: str) -> pd.DataFrame:
    """
    Read workers_dataset.csv, map worker IDs to integers, rename / derive
    ECG / EEG / GSR columns to match the notebook schema.

    Parameters
    ----------
    csv_path : str
        Path to workers_dataset.csv

    Returns
    -------
    pd.DataFrame  with columns: worker, ECG, EEG, GSR, marker
    """
    raw = pd.read_csv(csv_path)
    print(f"[data_loader] Loaded {len(raw):,} rows, "
          f"{raw['worker_id'].nunique()} unique worker IDs")

    # Map alphanumeric worker IDs (e.g. '17a') to integers
    worker_map = {}
    next_id = 100
    for wid in raw["worker_id"].unique():
        try:
            worker_map[wid] = int(wid)
        except ValueError:
            worker_map[wid] = next_id
            next_id += 1

    data = pd.DataFrame()
    data["worker"] = raw["worker_id"].map(worker_map)
    data["ECG"]    = raw["ECG"]
    data["EEG"]    = raw["EMG"].fillna(0.0)          # EMG used as EEG surrogate
    data["GSR"]    = raw[["foot_GSR", "hand_GSR"]].mean(axis=1)
    data["marker"] = raw["marker"]

    print(f"[data_loader] Mapped workers: {sorted(data['worker'].unique())}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 2. Attach marker-based fatigue labels
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_starting_times(marker_df: pd.DataFrame) -> pd.DataFrame:
    """Convert period durations (minutes) → cumulative start times."""
    result = marker_df.copy()
    result["NSP1"] = 0
    dur_cols = marker_df.iloc[:, 1:8].columns
    for i, col in enumerate(dur_cols):
        if i < len(dur_cols) - 1:
            result[dur_cols[i + 1]] = result[col] + marker_df[col]
    return result


def attach_labels(data: pd.DataFrame, marker_csv: str) -> pd.DataFrame:
    """
    Segment each worker's time-series using marker_info.csv timing and
    assign integer labels: 0=Relaxed, 1=Medium, 2=High-Fatigue.

    Workers without marker info are silently dropped.

    Parameters
    ----------
    data       : DataFrame with columns [worker, ECG, EEG, GSR, marker]
    marker_csv : path to marker_info.csv

    Returns
    -------
    Labelled DataFrame (rows outside all defined periods are removed).
    """
    marker_data        = pd.read_csv(marker_csv)
    processed_markers  = _calculate_starting_times(marker_data)

    labelled_dfs = []
    for _, row in processed_markers.iterrows():
        work_name = row["Worker"]                             # e.g. 'Worker05'
        worker_id = int(work_name.replace("Worker", ""))

        if worker_id not in data["worker"].unique():
            print(f"  [data_loader] SKIP {work_name}: not in dataset")
            continue

        worker_data = (
            data[data["worker"] == worker_id]
            .copy()
            .reset_index(drop=True)
        )
        n_samples      = len(worker_data)
        start_indices  = (row[PERIOD_COLS] * FS * 60).astype(int).values

        worker_data["label"] = np.nan
        segments = list(zip(PERIOD_COLS, PERIOD_LABELS.values()))

        for seg_idx, (period, lbl) in enumerate(segments):
            seg_start = start_indices[seg_idx]
            seg_end   = (
                start_indices[seg_idx + 1]
                if seg_idx < len(segments) - 1
                else n_samples
            )
            seg_end = min(seg_end, n_samples)
            worker_data.loc[seg_start:seg_end - 1, "label"] = lbl

        before = len(worker_data)
        worker_data = (
            worker_data
            .dropna(subset=["label"])
            .reset_index(drop=True)
        )
        worker_data["label"] = worker_data["label"].astype(int)
        after = len(worker_data)

        print(
            f"  {work_name}: {after:,} samples kept ({before - after:,} dropped) | "
            f"Relaxed={(worker_data['label']==0).sum():,}  "
            f"Medium={(worker_data['label']==1).sum():,}  "
            f"Fatigue={(worker_data['label']==2).sum():,}"
        )
        labelled_dfs.append(worker_data)

    labelled = pd.concat(labelled_dfs, ignore_index=True)
    print(f"\n[data_loader] Total labelled samples : {len(labelled):,}")
    print(f"[data_loader] Class balance:\n"
          f"{labelled['label'].value_counts().sort_index().rename({0:'Relaxed',1:'Medium',2:'Fatigue'})}")
    return labelled


# ─────────────────────────────────────────────────────────────────────────────
# 3. Impute NaN values per worker
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing(data: pd.DataFrame,
                   signal_cols: list = None) -> pd.DataFrame:
    """
    Per-worker linear interpolation (with forward/back fill fallback).
    Columns that are 100 % NaN are zero-filled.

    Parameters
    ----------
    data        : labelled DataFrame
    signal_cols : list of columns to impute (default: ['ECG', 'EEG', 'GSR'])

    Returns
    -------
    DataFrame with NaNs removed.
    """
    if signal_cols is None:
        signal_cols = ["ECG", "EEG", "GSR"]

    data = data.copy()
    for worker in sorted(data["worker"].unique()):
        mask = data["worker"] == worker
        for col in signal_cols:
            n_nan = data.loc[mask, col].isna().sum()
            if n_nan == 0:
                continue
            total = mask.sum()
            pct   = n_nan / total * 100
            if pct == 100.0:
                data.loc[mask, col] = 0.0
                print(f"  Worker {worker:2d} | {col}: 100% missing → zero-filled")
            else:
                data.loc[mask, col] = (
                    data.loc[mask, col]
                    .interpolate(method="linear")
                    .ffill()
                    .bfill()
                )
                print(f"  Worker {worker:2d} | {col}: {n_nan} NaNs ({pct:.1f}%) → interpolated")

    remaining = data[signal_cols].isna().sum().sum()
    print(f"\n[data_loader] Remaining NaNs after imputation: {remaining}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 4. One-stop pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_data(dataset_csv: str = DEFAULT_DATASET_CSV,
              marker_csv:  str = DEFAULT_MARKER_CSV) -> pd.DataFrame:
    """
    Full loading pipeline:
      1. Read compiled CSV
      2. Attach marker-based labels
      3. Impute missing values

    Returns
    -------
    Clean, labelled DataFrame ready for feature extraction.
    """
    print("=" * 60)
    print("[data_loader] Step 1 — Loading compiled CSV")
    print("=" * 60)
    data = load_compiled_csv(dataset_csv)

    print("\n" + "=" * 60)
    print("[data_loader] Step 2 — Attaching fatigue labels")
    print("=" * 60)
    data = attach_labels(data, marker_csv)

    print("\n" + "=" * 60)
    print("[data_loader] Step 3 — Imputing missing values")
    print("=" * 60)
    data = impute_missing(data)

    print("\n[data_loader] ✓ Data ready.")
    print(f"  Shape   : {data.shape}")
    print(f"  Workers : {sorted(data['worker'].unique())}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker Fatigue — Data Loader")
    parser.add_argument("--dataset_csv", default=DEFAULT_DATASET_CSV,
                        help="Path to workers_dataset.csv")
    parser.add_argument("--marker_csv",  default=DEFAULT_MARKER_CSV,
                        help="Path to marker_info.csv")
    parser.add_argument("--output",      default="outputs/loaded_data.csv",
                        help="Where to save the processed DataFrame")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = load_data(args.dataset_csv, args.marker_csv)
    df.to_csv(args.output, index=False)
    print(f"\n[data_loader] Saved → {args.output}")
