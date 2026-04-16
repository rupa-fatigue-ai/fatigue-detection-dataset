"""
training.py
===========
Trains all ML and DL models, logs results, and saves trained weights.

Pipeline:
  ML models  → Random Forest, SVM, Logistic Regression
  DL models  → Baseline LSTM, TAN v1, TAN v2 (full TAN)
  GAN + DL   → cGAN data augmentation → LSTM on augmented data
"""

import os
import warnings
import argparse
import pickle
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from model import (
    build_random_forest, build_svm, build_logistic_regression,
    build_baseline_lstm, build_tan_v1, build_tan_v2,
    build_lstm_for_cgan,
    build_cgan_generator, build_cgan_discriminator, build_cgan,
    f1_metric, LATENT_DIM, N_CLASSES,
)
from utils import (
    evaluate, evaluate_dl,
    plot_dashboard, plot_dashboard_dl,
    model_results_append,
)
from preprocessing import FEATURE_COLS

# ── TensorFlow guards ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    _TF = True
except ImportError:
    _TF = False
    print("[training] WARNING: TensorFlow not found — DL models will be skipped.")

# ── Default training hyper-parameters ───────────────────────────────────────
ML_THRESHOLD_RF  = 0.40
ML_THRESHOLD_SVM = 0.22
ML_THRESHOLD_LR  = 0.20
DL_EPOCHS        = 50
DL_BATCH         = 64
DL_LR            = 5e-4
CGAN_EPOCHS      = 100
CGAN_BATCH       = 32

# ─────────────────────────────────────────────────────────────────────────────
# Helper — reproducibility seed
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if _TF:
        tf.random.set_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ML  models
# ─────────────────────────────────────────────────────────────────────────────

def train_ml_models(prep:          dict,
                    model_results: list,
                    output_dir:    str) -> dict:
    """
    Train Random Forest, SVM, and Logistic Regression.

    Returns dict of fitted sklearn models.
    """
    X_tr  = prep["X_train_sc"]
    X_te  = prep["X_test_sc"]
    y_tr  = prep["y_train"]
    y_te  = prep["y_test"]

    ml_configs = [
        ("Random Forest",       build_random_forest(), ML_THRESHOLD_RF),
        ("SVM",                 build_svm(),            ML_THRESHOLD_SVM),
        ("Logistic Regression", build_logistic_regression(), ML_THRESHOLD_LR),
    ]

    trained_ml = {}
    for name, model, threshold in ml_configs:
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")

        model.fit(X_tr, y_tr)

        # Train metrics
        tr_prob = model.predict_proba(X_tr)[:, 1]
        tr_pred = (tr_prob > threshold).astype(int)
        train_metrics = evaluate(y_tr, tr_prob, tr_pred, f"{name} Train")

        # Test metrics
        te_prob = model.predict_proba(X_te)[:, 1]
        te_pred = (te_prob > threshold).astype(int)
        test_metrics = evaluate(y_te, te_prob, te_pred, f"{name} Test")

        # Log + plot
        model_results_append(model_results, train_metrics, test_metrics,
                              model_name=name, mode="ml")

        safe_name = name.replace(" ", "_")
        plot_dashboard(
            test_metrics, te_prob, model,
            title=name,
            feature_cols=FEATURE_COLS,
            save_path=os.path.join(output_dir, "plots", f"{safe_name}_dashboard.png"),
        )

        # Persist
        model_path = os.path.join(output_dir, "models", f"{safe_name}.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)
        print(f"  Saved → {model_path}")

        trained_ml[name] = model

    return trained_ml


# ─────────────────────────────────────────────────────────────────────────────
# 2. cGAN data augmentation
# ─────────────────────────────────────────────────────────────────────────────

def train_cgan(train_df:    pd.DataFrame,
               n_features:  int,
               output_dir:  str) -> dict:
    """
    Train a conditional GAN on the flat (non-sequential) training features
    and generate synthetic Non-Fatigue samples to balance the dataset.

    Returns dict with:
        generator, mms (MinMaxScaler),
        aug_df (augmented flat DataFrame)
    """
    if not _TF:
        print("[training] Skipping cGAN — TensorFlow not available.")
        return {}

    from sklearn.preprocessing import MinMaxScaler

    print(f"\n{'='*60}")
    print("  Training cGAN")
    print(f"{'='*60}")

    X_flat = train_df[FEATURE_COLS].values.astype("float32")
    y_flat = (train_df["label"].values > 0).astype("int32")

    mms = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = mms.fit_transform(X_flat).astype("float32")

    generator     = build_cgan_generator(n_features)
    discriminator = build_cgan_discriminator(n_features)

    d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    discriminator.compile(optimizer=d_optimizer,
                          loss="binary_crossentropy",
                          metrics=["accuracy"])

    cgan_model = build_cgan(generator, discriminator)

    real_lbl = np.ones((CGAN_BATCH, 1), dtype="float32")
    fake_lbl = np.zeros((CGAN_BATCH, 1), dtype="float32")
    d_losses, g_losses = [], []

    for epoch in range(CGAN_EPOCHS):
        idx    = np.random.randint(0, X_scaled.shape[0], CGAN_BATCH)
        real_x = X_scaled[idx]
        real_y = y_flat[idx].reshape(-1, 1).astype("int32")

        noise  = np.random.normal(0, 1, (CGAN_BATCH, LATENT_DIM)).astype("float32")
        fake_y = np.random.randint(0, N_CLASSES, (CGAN_BATCH, 1)).astype("int32")
        fake_x = generator.predict([noise, fake_y], verbose=0)

        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch([real_x, real_y], real_lbl)
        d_loss_fake = discriminator.train_on_batch([fake_x, fake_y], fake_lbl)
        d_loss      = 0.5 * (d_loss_real[0] + d_loss_fake[0])

        discriminator.trainable = False
        noise  = np.random.normal(0, 1, (CGAN_BATCH, LATENT_DIM)).astype("float32")
        gen_y  = np.random.randint(0, N_CLASSES, (CGAN_BATCH, 1)).astype("int32")
        g_loss = cgan_model.train_on_batch([noise, gen_y], real_lbl)

        d_losses.append(d_loss)
        g_losses.append(g_loss)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:4d}/{CGAN_EPOCHS} | D loss: {d_loss:.4f} | G loss: {g_loss:.4f}")

    print("[training] cGAN training complete.")

    # Generate synthetic Non-Fatigue samples to balance dataset
    n_non = (y_flat == 0).sum()
    n_fat = (y_flat == 1).sum()
    n_synth = max(0, n_fat - n_non)

    synth_feats = np.empty((0, n_features), dtype="float32")
    if n_synth > 0:
        noise_s  = np.random.normal(0, 1, (n_synth, LATENT_DIM)).astype("float32")
        labels_s = np.zeros((n_synth, 1), dtype="int32")
        synth_sc = generator.predict([noise_s, labels_s], verbose=0)
        synth_feats = mms.inverse_transform(synth_sc).astype("float32")

    synth_df = pd.DataFrame(synth_feats, columns=FEATURE_COLS)
    synth_df["label"]     = 0
    synth_df["label_bin"] = 0
    synth_df["worker"]    = -1   # synthetic marker

    real_df            = train_df[FEATURE_COLS + ["label", "worker"]].copy()
    real_df["label_bin"] = (real_df["label"] > 0).astype(int)
    aug_df             = pd.concat([real_df, synth_df], ignore_index=True)

    print(f"  Augmented dataset: {len(aug_df):,} rows | "
          f"Non-Fatigue={( aug_df['label_bin']==0).sum():,}  "
          f"Fatigue={(aug_df['label_bin']==1).sum():,}")

    # Save generator weights
    gen_path = os.path.join(output_dir, "models", "cgan_generator.h5")
    os.makedirs(os.path.dirname(gen_path), exist_ok=True)
    generator.save(gen_path)
    print(f"  Generator saved → {gen_path}")

    return {"generator": generator, "mms": mms, "aug_df": aug_df}


# ─────────────────────────────────────────────────────────────────────────────
# 3. DL models
# ─────────────────────────────────────────────────────────────────────────────

def _dl_callbacks(patience: int = 10):
    return [
        EarlyStopping(monitor="val_loss",
                      patience=patience,
                      restore_best_weights=True,
                      verbose=1),
    ]


def train_dl_models(prep:          dict,
                    model_results: list,
                    output_dir:    str,
                    cgan_data:     dict = None) -> dict:
    """
    Train Baseline LSTM, TAN v1, TAN v2, and (optionally) cGAN+LSTM.

    Parameters
    ----------
    prep        : output of preprocessing.preprocess()
    cgan_data   : output of train_cgan()  — enables cGAN+LSTM if provided

    Returns
    -------
    dict of trained Keras models.
    """
    if not _TF:
        print("[training] Skipping DL — TensorFlow not available.")
        return {}

    set_seeds()
    SEQ_LEN    = prep["X_seq_train"].shape[1]
    N_FEATURES = prep["X_seq_train"].shape[2]

    X_tr   = prep["X_tr"]
    y_tr   = prep["y_tr"]
    X_val  = prep["X_val"]
    y_val  = prep["y_val"]
    X_test = prep["X_seq_test"]
    y_test = prep["y_seq_test"]
    cw     = prep["class_weight"]

    # ── Shuffle training subset ───────────────────────────────────────────────
    idx    = np.random.permutation(len(X_tr))
    X_shuf = X_tr[idx]
    y_shuf = y_tr[idx]

    trained_dl = {}

    # ─── 3a. Baseline LSTM ────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  Training: Baseline LSTM\n{'='*60}")
    lstm_model = build_baseline_lstm(SEQ_LEN, N_FEATURES, lr=DL_LR)
    history_lstm = lstm_model.fit(
        X_shuf, y_shuf,
        validation_split=0.15,
        epochs=DL_EPOCHS,
        batch_size=DL_BATCH,
        class_weight=cw,
        callbacks=_dl_callbacks(10),
        verbose=1,
    )
    y_prob = lstm_model.predict(X_test, verbose=0).flatten()
    metrics = evaluate_dl(y_test, y_prob, "Baseline LSTM")
    model_results_append(model_results, model_name="baseline_LSTM",
                         mode="dl", history=history_lstm, metrics=metrics)
    plot_dashboard_dl(metrics, y_test, y_prob, history_lstm,
                      title="Baseline LSTM",
                      save_path=os.path.join(output_dir, "plots", "LSTM_dashboard.png"))
    lstm_model.save(os.path.join(output_dir, "models", "baseline_lstm.h5"))
    trained_dl["baseline_LSTM"] = lstm_model

    # ─── 3b. TAN v1 (LSTM + Self-Attention) ──────────────────────────────────
    print(f"\n{'='*60}\n  Training: TAN v1 (LSTM + Self-Attention)\n{'='*60}")
    idx    = np.random.permutation(len(prep["X_seq_train"]))
    X_all  = prep["X_seq_train"][idx]
    y_all  = (prep["y_seq_train"][idx] > 0).astype(int)

    tan_v1 = build_tan_v1(SEQ_LEN, N_FEATURES, lr=DL_LR)
    history_v1 = tan_v1.fit(
        X_all, y_all,
        validation_split=0.15,
        epochs=100,
        batch_size=DL_BATCH,
        class_weight=cw,
        callbacks=_dl_callbacks(10),
        verbose=1,
    )
    y_prob = tan_v1.predict(X_test, verbose=0).flatten()
    metrics = evaluate_dl(y_test, y_prob, "TAN_v1")
    model_results_append(model_results, model_name="TAN_v1",
                         mode="dl", history=history_v1, metrics=metrics)
    plot_dashboard_dl(metrics, y_test, y_prob, history_v1,
                      title="TAN_v1 (LSTM + Self-Attention)",
                      save_path=os.path.join(output_dir, "plots", "TAN_v1_dashboard.png"))
    tan_v1.save(os.path.join(output_dir, "models", "tan_v1.h5"))
    trained_dl["TAN_v1"] = tan_v1

    # ─── 3c. TAN v2 (LSTM + Self-Attention + General Attention) ──────────────
    print(f"\n{'='*60}\n  Training: TAN v2 (Full Dual-Attention)\n{'='*60}")
    tan_v2 = build_tan_v2(SEQ_LEN, N_FEATURES, lr=DL_LR)
    idx    = np.random.permutation(len(X_all))
    history_v2 = tan_v2.fit(
        X_all[idx], y_all[idx],
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        class_weight=cw,
        callbacks=_dl_callbacks(10),
        verbose=1,
    )
    y_prob = tan_v2.predict(X_test, verbose=0).flatten()
    metrics = evaluate_dl(y_test, y_prob, "TAN_v2")
    model_results_append(model_results, model_name="TAN_v2",
                         mode="dl", history=history_v2, metrics=metrics)
    plot_dashboard_dl(metrics, y_test, y_prob, history_v2,
                      title="TAN_v2 (Full Dual-Attention)",
                      save_path=os.path.join(output_dir, "plots", "TAN_v2_dashboard.png"))
    tan_v2.save(os.path.join(output_dir, "models", "tan_v2.h5"))
    trained_dl["TAN_v2"] = tan_v2

    # ─── 3d. cGAN + LSTM (if augmented data available) ────────────────────────
    if cgan_data and "aug_df" in cgan_data:
        print(f"\n{'='*60}\n  Training: cGAN + LSTM\n{'='*60}")

        from preprocessing import build_sequences, normalise_per_worker
        from sklearn.preprocessing import StandardScaler

        aug_df = cgan_data["aug_df"]

        aug_norm   = normalise_per_worker(aug_df, FEATURE_COLS)
        scaler_aug = StandardScaler()
        scaler_aug.fit(aug_norm[FEATURE_COLS].values)

        def _build_seqs(df, scaler):
            from preprocessing import SEQ_LEN as SL
            X_s, y_s = [], []
            for wid, grp in df.groupby("worker"):
                grp    = grp.reset_index(drop=True)
                scaled = scaler.transform(grp[FEATURE_COLS].values)
                labels = grp["label_bin"].values
                for i in range(SL, len(grp)):
                    X_s.append(scaled[i - SL: i])
                    y_s.append(labels[i])
            return (np.array(X_s, dtype=np.float32),
                    np.array(y_s, dtype=np.int64))

        X_aug_train, y_aug_train = _build_seqs(aug_norm, scaler_aug)

        # Validation set from val_sub
        test_norm = normalise_per_worker(prep["test_df"], FEATURE_COLS)
        X_te_aug, y_te_aug = _build_seqs(
            test_norm.assign(label_bin=(test_norm["label"] > 0).astype(int)),
            scaler_aug,
        )

        lstm_aug = build_lstm_for_cgan(SEQ_LEN, N_FEATURES, lr=1e-3)
        history_aug = lstm_aug.fit(
            X_aug_train, y_aug_train,
            validation_split=0.15,
            epochs=100,
            batch_size=DL_BATCH,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10,
                              restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                  patience=5, min_lr=1e-6, verbose=0),
            ],
            verbose=1,
        )

        y_pred_aug = (lstm_aug.predict(X_te_aug, verbose=0) > 0.5).astype(int).flatten()
        metrics    = evaluate_dl(y_te_aug, y_pred_aug.astype(float), "cGAN+LSTM")
        model_results_append(model_results, model_name="cGAN+LSTM",
                             mode="dl", history=history_aug, metrics=metrics)
        plot_dashboard_dl(metrics, y_te_aug, y_pred_aug.astype(float),
                          history_aug, title="cGAN + LSTM",
                          save_path=os.path.join(output_dir, "plots", "cGAN_LSTM_dashboard.png"))
        lstm_aug.save(os.path.join(output_dir, "models", "cgan_lstm.h5"))
        trained_dl["cGAN+LSTM"] = lstm_aug

    return trained_dl


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_all(prep: dict,
              output_dir:  str = "outputs",
              skip_dl:     bool = False,
              skip_cgan:   bool = False) -> dict:
    """
    Run the complete training pipeline.

    Parameters
    ----------
    prep       : dict from preprocessing.preprocess()
    output_dir : base directory for saved models / plots / results
    skip_dl    : if True, train only ML models (faster)
    skip_cgan  : if True, skip cGAN augmentation step

    Returns
    -------
    dict with keys: model_results, trained_ml, trained_dl
    """
    set_seeds()
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"),  exist_ok=True)

    model_results = []

    # ── ML ────────────────────────────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  PHASE 1 — Machine Learning Models")
    print("█" * 60)
    trained_ml = train_ml_models(prep, model_results, output_dir)

    trained_dl  = {}
    cgan_data   = {}

    if not skip_dl:
        # ── cGAN augmentation ──────────────────────────────────────────────────
        if not skip_cgan:
            print("\n" + "█" * 60)
            print("  PHASE 2 — cGAN Data Augmentation")
            print("█" * 60)
            cgan_data = train_cgan(
                prep["train_df"],
                n_features=len(FEATURE_COLS),
                output_dir=output_dir,
            )

        # ── DL ─────────────────────────────────────────────────────────────────
        print("\n" + "█" * 60)
        print("  PHASE 3 — Deep Learning Models")
        print("█" * 60)
        trained_dl = train_dl_models(prep, model_results, output_dir, cgan_data)

    # ── Save model_results ────────────────────────────────────────────────────
    results_path = os.path.join(output_dir, "model_results.csv")
    pd.DataFrame(model_results).to_csv(results_path, index=False)
    print(f"\n[training] Results table saved → {results_path}")

    return {
        "model_results": model_results,
        "trained_ml":    trained_ml,
        "trained_dl":    trained_dl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker Fatigue — Training")
    parser.add_argument("--feature_matrix", required=True,
                        help="Path to feature_matrix.csv")
    parser.add_argument("--output_dir", default="outputs",
                        help="Directory for results / models / plots")
    parser.add_argument("--skip_dl",   action="store_true",
                        help="Train only ML models")
    parser.add_argument("--skip_cgan", action="store_true",
                        help="Skip cGAN augmentation step")
    args = parser.parse_args()

    from preprocessing import preprocess

    fm   = pd.read_csv(args.feature_matrix)
    prep = preprocess(fm)

    train_all(prep,
              output_dir=args.output_dir,
              skip_dl=args.skip_dl,
              skip_cgan=args.skip_cgan)
