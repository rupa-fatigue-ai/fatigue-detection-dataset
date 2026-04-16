"""
preprocessing.py
================
Scales features, builds worker-level train / test splits (no leakage),
and constructs fixed-length sequences for deep-learning models.
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ── Feature columns produced by feature_extraction.py ───────────────────────
FEATURE_COLS = [
    "ecg_mean", "ecg_var", "ecg_std", "ecg_rms",
    "hrv_rmssd", "hrv_sdnn", "ecg_lf_hf", "ecg_slope",
    "gsr_mean", "gsr_var", "gsr_std", "gsr_slope",
    "gsr_tonic_mean", "gsr_phasic_std", "gsr_scr_amp", "gsr_peak_count",
    "eeg_mean", "eeg_var", "eeg_rms", "eeg_energy",
    "eeg_delta", "eeg_theta",
]

# Hold-out workers for test set (no overlap with training)
TEST_WORKERS = [15, 16]
VAL_WORKER   = 12        # held out from train for early-stopping validation
SEQ_LEN      = 5         # number of consecutive windows per sequence


# ─────────────────────────────────────────────────────────────────────────────
# 1. Worker-level train / test split
# ─────────────────────────────────────────────────────────────────────────────

def worker_split(feature_matrix: pd.DataFrame,
                 test_workers: list = TEST_WORKERS):
    """
    Split feature_matrix by worker IDs so no worker appears in both
    train and test (prevents temporal / subject leakage).

    Returns
    -------
    train_df, test_df : DataFrames with columns FEATURE_COLS + [worker, label]
    """
    all_workers   = feature_matrix["worker"].unique().tolist()
    train_workers = [w for w in all_workers if w not in test_workers]

    train_df = feature_matrix[
        feature_matrix["worker"].isin(train_workers)
    ].reset_index(drop=True)

    test_df = feature_matrix[
        feature_matrix["worker"].isin(test_workers)
    ].reset_index(drop=True)

    overlap = set(train_df["worker"]) & set(test_df["worker"])
    assert len(overlap) == 0, f"Worker leakage detected: {overlap}"

    print(f"[preprocessing] Train : {len(train_df):,} windows | workers {train_workers}")
    print(f"[preprocessing] Test  : {len(test_df):,} windows  | workers {test_workers}")
    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Binary label
# ─────────────────────────────────────────────────────────────────────────────

def binarise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse 3-class labels (0=Relaxed, 1=Medium, 2=Fatigue) into binary:
      0 = Non-Fatigued (label == 0)
      1 = Elevated     (label  > 0)
    """
    df = df.copy()
    df["label_bin"] = (df["label"] > 0).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. StandardScaler fit on train, transform both splits
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(train_df: pd.DataFrame,
                   test_df:  pd.DataFrame,
                   feature_cols: list = FEATURE_COLS):
    """
    Fit StandardScaler on training features only, then transform both sets.

    Returns
    -------
    X_train_sc, X_test_sc : np.ndarray
    y_train, y_test       : np.ndarray (binary labels)
    scaler                : fitted StandardScaler
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(train_df[feature_cols].values)
    X_test_sc  = scaler.transform(test_df[feature_cols].values)

    y_train = train_df["label_bin"].values
    y_test  = test_df["label_bin"].values

    print(f"\n[preprocessing] X_train : {X_train_sc.shape}  |  y_train : {y_train.shape}")
    print(f"[preprocessing] X_test  : {X_test_sc.shape}   |  y_test  : {y_test.shape}")
    print(f"\n[preprocessing] Binary class balance (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        label = "Non-Fatigue" if u == 0 else "Elevated"
        print(f"  {label}: {c:,}")

    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-worker normalisation (for deep-learning sequences)
# ─────────────────────────────────────────────────────────────────────────────

def normalise_per_worker(df: pd.DataFrame,
                         cols: list = FEATURE_COLS) -> pd.DataFrame:
    """Z-score each worker's feature columns independently."""
    out      = df.copy()
    out[cols] = out[cols].astype("float64")
    for worker_id, grp in df.groupby("worker"):
        mask = out["worker"] == worker_id
        mu   = grp[cols].astype("float64").mean()
        sig  = grp[cols].astype("float64").std().replace(0, 1e-8)
        out.loc[mask, cols] = (
            grp[cols].astype("float64").values - mu.values
        ) / sig.values
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sequence builder for LSTM / TAN models
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame,
                    scaler:       StandardScaler,
                    feature_cols: list = FEATURE_COLS,
                    seq_len:      int  = SEQ_LEN):
    """
    Build overlapping sequences of length `seq_len` per worker.

    Each sequence (X[i]) has shape (seq_len, n_features).
    Label y[i] is the binary label of the last window in the sequence.

    Returns
    -------
    X_seqs : np.ndarray  shape (N, seq_len, n_features)
    y_seqs : np.ndarray  shape (N,)
    """
    X_seqs, y_seqs = [], []
    for worker_id, grp in df.groupby("worker"):
        grp    = grp.reset_index(drop=True)
        scaled = scaler.transform(grp[feature_cols].values)
        labels = grp["label_bin"].values
        for i in range(seq_len, len(grp)):
            X_seqs.append(scaled[i - seq_len: i])
            y_seqs.append(labels[i])

    X = np.array(X_seqs, dtype=np.float32)
    y = np.array(y_seqs, dtype=np.int64)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 6. Class weights helper
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(y_train: np.ndarray) -> dict:
    """
    Compute balanced class weights for handling class imbalance.
    Uses sklearn's compute_class_weight with 'balanced' strategy.
    Returns a dict {0: w0, 1: w1} — the notebook overrides these with
    {0: 1.8, 1: 1.0} after inspection; both options are printed.
    """
    cw = compute_class_weight("balanced",
                               classes=np.array([0, 1]),
                               y=y_train)
    computed = {0: float(cw[0]), 1: float(cw[1])}
    fixed    = {0: 1.8, 1: 1.0}
    print(f"[preprocessing] Computed class weights : {computed}")
    print(f"[preprocessing] Fixed class weights    : {fixed}  (used for training)")
    return fixed


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(feature_matrix: pd.DataFrame,
               feature_cols:  list = FEATURE_COLS,
               test_workers:  list = TEST_WORKERS,
               seq_len:       int  = SEQ_LEN):
    """
    Orchestrates the full preprocessing pipeline.

    Returns a dict with:
      X_train_sc, X_test_sc, y_train, y_test   → for ML models
      X_seq_train, X_seq_test,
      y_seq_train, y_seq_test                  → for DL models
      X_tr, y_tr, X_val, y_val                 → DL train/val sub-split
      scaler, class_weight
    """
    print("=" * 60)
    print("[preprocessing] Binarising labels")
    print("=" * 60)
    feature_matrix = binarise_labels(feature_matrix)

    print("\n" + "=" * 60)
    print("[preprocessing] Worker-level train/test split")
    print("=" * 60)
    train_df, test_df = worker_split(feature_matrix, test_workers)

    print("\n" + "=" * 60)
    print("[preprocessing] Scaling features (StandardScaler)")
    print("=" * 60)
    X_train_sc, X_test_sc, y_train, y_test, scaler = scale_features(
        train_df, test_df, feature_cols
    )

    print("\n" + "=" * 60)
    print("[preprocessing] Building sequences (per-worker normalisation)")
    print("=" * 60)
    train_norm = normalise_per_worker(train_df, feature_cols)
    test_norm  = normalise_per_worker(test_df,  feature_cols)

    # Global scaler fit on per-worker-normalised train
    seq_scaler = StandardScaler()
    seq_scaler.fit(train_norm[feature_cols].values)

    X_seq_train, y_seq_train = build_sequences(train_norm, seq_scaler,
                                               feature_cols, seq_len)
    X_seq_test,  y_seq_test  = build_sequences(test_norm,  seq_scaler,
                                               feature_cols, seq_len)

    # Validation sub-split (worker VAL_WORKER held out)
    train_workers_all = train_norm["worker"].unique().tolist()
    val_w   = VAL_WORKER if VAL_WORKER in train_workers_all else train_workers_all[-1]
    tr_sub  = train_norm[train_norm["worker"] != val_w].reset_index(drop=True)
    val_sub = train_norm[train_norm["worker"] == val_w].reset_index(drop=True)

    X_tr,  y_tr  = build_sequences(tr_sub,  seq_scaler, feature_cols, seq_len)
    X_val, y_val = build_sequences(val_sub, seq_scaler, feature_cols, seq_len)

    class_weight = get_class_weights(y_tr)

    print(f"\n[preprocessing] Sequence shapes:")
    print(f"  X_seq_train : {X_seq_train.shape}  X_seq_test : {X_seq_test.shape}")
    print(f"  X_tr        : {X_tr.shape}          X_val      : {X_val.shape}")

    return {
        # ML arrays
        "X_train_sc": X_train_sc,
        "X_test_sc":  X_test_sc,
        "y_train":    y_train,
        "y_test":     y_test,
        # DL sequences
        "X_seq_train": X_seq_train,
        "X_seq_test":  X_seq_test,
        "y_seq_train": y_seq_train,
        "y_seq_test":  y_seq_test,
        # DL early-stopping validation split
        "X_tr":  X_tr,
        "y_tr":  y_tr,
        "X_val": X_val,
        "y_val": y_val,
        # Artefacts
        "scaler":       scaler,
        "seq_scaler":   seq_scaler,
        "train_df":     train_df,
        "test_df":      test_df,
        "class_weight": class_weight,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker Fatigue — Preprocessing")
    parser.add_argument("--feature_matrix", required=True,
                        help="Path to feature_matrix.csv (output of feature_extraction.py)")
    parser.add_argument("--output_dir", default="outputs",
                        help="Directory to save preprocessed arrays")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fm = pd.read_csv(args.feature_matrix)
    result = preprocess(fm)

    # Save ML arrays as CSV for inspection
    for key in ["X_train_sc", "X_test_sc", "y_train", "y_test"]:
        np.save(os.path.join(args.output_dir, f"{key}.npy"), result[key])
        print(f"  Saved {key}.npy")

    print("\n[preprocessing] ✓ Done.")
