"""
model.py
========
Defines all ML and DL model architectures used in the Worker Fatigue project:

  ML models  (scikit-learn)
    • Random Forest
    • SVM (RBF kernel)
    • Logistic Regression

  DL models  (TensorFlow / Keras)
    • Baseline LSTM
    • TAN v1  — LSTM + Self-Attention
    • TAN v2  — LSTM + Self-Attention + General Attention (full TAN model)
    • LSTM for cGAN-augmented data

  GAN
    • cGAN Generator
    • cGAN Discriminator
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np

# ── TensorFlow import (optional — only required for DL models) ───────────────
try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Input, LSTM, Dense, Dropout, Bidirectional,
        Layer, Concatenate,
        BatchNormalization, LeakyReLU, Flatten, Embedding
    )
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    print("[model] WARNING: TensorFlow not available — DL models disabled.")

# ─────────────────────────────────────────────────────────────────────────────
# Shared Keras metric: F1 Score
# ─────────────────────────────────────────────────────────────────────────────

def f1_metric(y_true, y_pred):
    """Differentiable F1 score used as a Keras training metric."""
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for f1_metric.")
    y_pred = tf.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, "float"))
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"))
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"))
    precision = tp / (tp + fp + K.epsilon())
    recall    = tp / (tp + fn + K.epsilon())
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# ─────────────────────────────────────────────────────────────────────────────
# ML Models
# ─────────────────────────────────────────────────────────────────────────────

def build_random_forest(n_estimators: int = 3,
                        max_depth:    int = 5,
                        random_state: int = 42):
    """
    Random Forest classifier with balanced class weights.
    Uses n_estimators=3 / max_depth=5 (fast; tune for production).
    """
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_svm(C:            float = 2.0,
              gamma:        str   = "scale",
              random_state: int   = 42):
    """SVM with RBF kernel and balanced class weights."""
    from sklearn.svm import SVC
    return SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        class_weight="balanced",
        probability=True,
        random_state=random_state,
    )


def build_logistic_regression(max_iter: int = 1000):
    """Logistic Regression with balanced class weights."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(class_weight="balanced", max_iter=max_iter)


# ─────────────────────────────────────────────────────────────────────────────
# DL custom layers
# ─────────────────────────────────────────────────────────────────────────────

if _TF_AVAILABLE:

    class SelfAttention(Layer):
        """
        Additive (Bahdanau-style) self-attention.
        Learns a scalar weight for each time-step and returns the
        weighted sum of the hidden states.
        """
        def build(self, input_shape):
            self.W = self.add_weight(
                shape=(input_shape[-1], 1),
                initializer="random_normal",
                trainable=True,
                name="self_attn_W",
            )
            self.b = self.add_weight(
                shape=(input_shape[1], 1),
                initializer="zeros",
                trainable=True,
                name="self_attn_b",
            )

        def call(self, x):
            e = K.tanh(K.dot(x, self.W) + self.b)   # (batch, time, 1)
            a = K.softmax(e, axis=1)                 # attention weights
            return K.sum(x * a, axis=1)              # (batch, features)

        def get_config(self):
            return super().get_config()


    class GeneralAttention(Layer):
        """
        General (Luong-style) attention.
        Uses a trainable weight matrix W to produce attention scores
        between hidden states.
        """
        def build(self, input_shape):
            dim = input_shape[-1]
            self.W = self.add_weight(
                shape=(dim, dim),
                initializer="random_normal",
                trainable=True,
                name="gen_attn_W",
            )

        def call(self, x):
            score   = K.batch_dot(x, K.dot(x, self.W), axes=[2, 2])
            weights = K.softmax(score, axis=1)
            context = K.batch_dot(weights, x)
            return K.sum(context, axis=1)

        def get_config(self):
            return super().get_config()


# ─────────────────────────────────────────────────────────────────────────────
# DL Models
# ─────────────────────────────────────────────────────────────────────────────

def build_baseline_lstm(seq_len:    int,
                        n_features: int,
                        lr:         float = 5e-4) -> "Sequential":
    """
    Baseline stacked LSTM.
    Architecture: LSTM(32) → Dropout → LSTM(16) → Dropout → Dense(1, sigmoid)
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow required.")
    model = Sequential([
        LSTM(32,
             input_shape=(seq_len, n_features),
             return_sequences=True,
             activation="tanh",
             recurrent_activation="sigmoid"),
        Dropout(0.4),
        LSTM(16, activation="tanh", recurrent_activation="sigmoid"),
        Dropout(0.4),
        Dense(1, activation="sigmoid"),
    ], name="Baseline_LSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", "Precision", "Recall", f1_metric],
    )
    return model


def build_tan_v1(seq_len:    int,
                 n_features: int,
                 lstm1:      int   = 64,
                 lstm2:      int   = 32,
                 dropout:    float = 0.3,
                 lr:         float = 5e-4) -> "Model":
    """
    TAN v1 — LSTM + Self-Attention.
    Architecture: LSTM(lstm1) → LSTM(lstm2) → SelfAttention → Dense(32) → Dense(1)
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow required.")

    inputs  = Input(shape=(seq_len, n_features))
    x       = LSTM(lstm1, return_sequences=True)(inputs)
    x       = Dropout(dropout)(x)
    x       = LSTM(lstm2, return_sequences=True)(x)
    x       = Dropout(dropout)(x)
    context = SelfAttention()(x)
    x       = Dense(32, activation="relu")(context)
    x       = Dropout(dropout)(x)
    output  = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, output, name="TAN_v1_SelfAttention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", "Precision", "Recall", f1_metric],
    )
    return model


def build_tan_v2(seq_len:    int,
                 n_features: int,
                 lstm1:      int   = 128,
                 lstm2:      int   = 32,
                 dropout:    float = 0.3,
                 lr:         float = 5e-4) -> "Model":
    """
    TAN v2 — Full TAN: LSTM + Self-Attention + General Attention (DAN).
    Architecture:
      LSTM(lstm1) → LSTM(lstm2, return_sequences=True)
        ├─ SelfAttention ─┐
        └─ GeneralAttn  ──┴─ Concatenate → Dense(64) → Dense(32) → Dense(1)
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow required.")

    inputs   = Input(shape=(seq_len, n_features))
    x        = LSTM(lstm1, return_sequences=True)(inputs)
    x        = Dropout(dropout)(x)
    x        = LSTM(lstm2, return_sequences=True)(x)
    x        = Dropout(dropout)(x)

    self_att = SelfAttention()(x)
    gen_att  = GeneralAttention()(x)
    combined = Concatenate()([self_att, gen_att])

    x      = Dense(64, activation="relu")(combined)
    x      = Dropout(dropout)(x)
    x      = Dense(32, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, output, name="TAN_v2_DualAttention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", "Precision", "Recall", f1_metric],
    )
    return model


def build_lstm_for_cgan(seq_len:    int,
                        n_features: int,
                        lstm1:      int   = 64,
                        lstm2:      int   = 32,
                        dropout:    float = 0.3,
                        lr:         float = 1e-3,
                        bidirectional: bool = False) -> "Sequential":
    """
    LSTM trained on cGAN-augmented data.
    Optionally wraps LSTM layers with Bidirectional.
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow required.")

    model = Sequential(name="LSTM_cGAN_augmented")

    if bidirectional:
        model.add(Bidirectional(
            LSTM(lstm1, return_sequences=True),
            input_shape=(seq_len, n_features),
        ))
    else:
        model.add(LSTM(lstm1, return_sequences=True,
                       input_shape=(seq_len, n_features)))
    model.add(Dropout(dropout))

    if bidirectional:
        model.add(Bidirectional(LSTM(lstm2)))
    else:
        model.add(LSTM(lstm2))

    model.add(Dropout(dropout))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", "Precision", "Recall"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# cGAN
# ─────────────────────────────────────────────────────────────────────────────

LATENT_DIM = 32
N_CLASSES  = 2


def build_cgan_generator(n_features: int,
                         latent_dim: int = LATENT_DIM,
                         n_classes:  int = N_CLASSES) -> "Model":
    """
    Conditional GAN Generator.
    Input : noise vector (LATENT_DIM,) + class label (int)
    Output: synthetic feature window of shape (n_features,)
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow required.")

    noise_in     = Input(shape=(latent_dim,), name="noise")
    label_in     = Input(shape=(1,), dtype="int32", name="label")
    label_onehot = tf.one_hot(tf.squeeze(label_in, axis=1), depth=n_classes)

    x = Concatenate()([noise_in, label_onehot])
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)
    out = Dense(n_features, activation="tanh")(x)

    return Model([noise_in, label_in], out, name="cGAN_Generator")


def build_cgan_discriminator(n_features: int,
                             n_classes:  int = N_CLASSES) -> "Model":
    """
    Conditional GAN Discriminator.
    Input : feature window (n_features,) + class label (int)
    Output: real/fake probability (sigmoid)
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow required.")

    feat_in      = Input(shape=(n_features,), name="features")
    label_in     = Input(shape=(1,), dtype="int32", name="label")
    label_onehot = tf.one_hot(tf.squeeze(label_in, axis=1), depth=n_classes)

    x = Concatenate()([feat_in, label_onehot])
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(64)(x)
    x = LeakyReLU(0.2)(x)
    out = Dense(1, activation="sigmoid")(x)

    return Model([feat_in, label_in], out, name="cGAN_Discriminator")


def build_cgan(generator: "Model",
               discriminator: "Model") -> "Model":
    """
    Full cGAN: freeze discriminator weights and wrap inside a combined model
    for generator training.
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow required.")

    discriminator.trainable = False
    noise_in  = Input(shape=(LATENT_DIM,))
    label_in  = Input(shape=(1,), dtype="int32")
    fake_feat = generator([noise_in, label_in])
    validity  = discriminator([fake_feat, label_in])
    cgan      = Model([noise_in, label_in], validity, name="cGAN")
    cgan.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        loss="binary_crossentropy",
    )
    return cgan


# ─────────────────────────────────────────────────────────────────────────────
# Model summary helper
# ─────────────────────────────────────────────────────────────────────────────

def print_model_summary(model, name: str = "") -> None:
    label = name or getattr(model, "name", "Model")
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if _TF_AVAILABLE and isinstance(model, (Sequential, Model)):
        model.summary()
    else:
        print(f"  {model}")


if __name__ == "__main__":
    # Quick smoke test
    print("ML models:")
    for fn in [build_random_forest, build_svm, build_logistic_regression]:
        m = fn()
        print(f"  {m.__class__.__name__} ✓")

    if _TF_AVAILABLE:
        print("\nDL models (seq_len=5, n_features=22):")
        for builder, kwargs in [
            (build_baseline_lstm, {}),
            (build_tan_v1,        {}),
            (build_tan_v2,        {}),
            (build_lstm_for_cgan, {}),
        ]:
            m = builder(seq_len=5, n_features=22, **kwargs)
            print(f"  {m.name} ✓  params={m.count_params():,}")

        print("\ncGAN architectures:")
        g = build_cgan_generator(n_features=22)
        d = build_cgan_discriminator(n_features=22)
        print(f"  Generator     ✓  params={g.count_params():,}")
        print(f"  Discriminator ✓  params={d.count_params():,}")
