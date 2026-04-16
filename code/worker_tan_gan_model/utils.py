"""
utils.py
========
Shared evaluation, plotting, model-result logging, and SHAP/LIME
explainability helpers used across training.py and results.py.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import seaborn as sns
    _SNS = True
except ImportError:
    _SNS = False

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
)
from sklearn.inspection import permutation_importance

CLASS_NAMES = ["Non Fatigue", "Fatigue"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true:  np.ndarray,
             y_prob:  np.ndarray,
             y_pred:  np.ndarray,
             label:   str = "Model") -> dict:
    """
    Full evaluation for ML models (fixed-threshold predictions already supplied).

    Parameters
    ----------
    y_true : ground-truth binary labels
    y_prob : predicted probabilities for positive class
    y_pred : thresholded binary predictions

    Returns
    -------
    dict with acc, precision, recall, f1, roc_auc, cm, fpr/tpr, …
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n=== {label} Evaluation ===")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  {label} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob)

    thresholds = np.linspace(0, 1, 50)
    f1_scores  = [f1_score(y_true, (y_prob > t).astype(int), zero_division=0)
                  for t in thresholds]
    best_t     = float(thresholds[np.argmax(f1_scores)])
    best_f1    = float(np.max(f1_scores))

    cm = confusion_matrix(y_true, y_pred)

    return {
        "acc": acc, "precision": prec, "recall": rec, "f1": f1,
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True),
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "pr_precision": pr_prec, "pr_recall": pr_rec,
        "thresholds": thresholds, "f1_scores": f1_scores,
        "best_t": best_t, "best_f1": best_f1, "cm": cm,
    }


def evaluate_dl(y_true: np.ndarray,
                y_prob: np.ndarray,
                label:  str = "Model") -> dict:
    """
    Evaluation for DL models with automatic threshold tuning (maximises F1).

    Parameters
    ----------
    y_true : ground-truth binary labels
    y_prob : predicted probabilities (sigmoid output)

    Returns
    -------
    Same structure as evaluate() with an additional y_pred key.
    """
    thresholds = np.linspace(0, 1, 100)
    f1_scores  = [f1_score(y_true, (y_prob > t).astype(int), zero_division=0)
                  for t in thresholds]
    best_t     = float(thresholds[np.argmax(f1_scores)])
    best_f1    = float(np.max(f1_scores))
    y_pred     = (y_prob > best_t).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n=== {label} Evaluation ===")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Best Threshold : {best_t:.3f}")
    print(f"\n  {label} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)
    pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "acc": acc, "precision": prec, "recall": rec, "f1": f1,
        "best_t": best_t, "best_f1": best_f1,
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True),
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "pr_precision": pr_prec, "pr_recall": pr_rec,
        "thresholds": thresholds, "f1_scores": f1_scores,
        "cm": cm, "y_pred": y_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dashboard plots
# ─────────────────────────────────────────────────────────────────────────────

def _safe_heatmap(ax, cm):
    """Draw confusion matrix heatmap (with or without seaborn)."""
    if _SNS:
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES,
                    cmap="Blues", ax=ax)
    else:
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1]);  ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticks([0, 1]);  ax.set_yticklabels(CLASS_NAMES)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


def plot_dashboard(metrics:    dict,
                   y_prob:     np.ndarray,
                   model,
                   title:      str  = "Model",
                   feature_cols: list = None,
                   save_path:  str  = None) -> None:
    """2×3 performance dashboard for ML models."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    cm = metrics["cm"]

    # 1. Confusion Matrix
    _safe_heatmap(axes[0, 0], cm)

    # 2. ROC
    axes[0, 1].plot(metrics["fpr"], metrics["tpr"],
                    label=f"AUC={metrics['roc_auc']:.3f}")
    axes[0, 1].plot([0, 1], [0, 1], "--")
    axes[0, 1].legend(); axes[0, 1].set_title("ROC Curve")

    # 3. Precision-Recall
    axes[0, 2].plot(metrics["pr_recall"], metrics["pr_precision"])
    axes[0, 2].set_title("Precision-Recall Curve")

    # 4. F1 vs Threshold
    axes[1, 0].plot(metrics["thresholds"], metrics["f1_scores"])
    axes[1, 0].axvline(metrics["best_t"], linestyle="--")
    axes[1, 0].set_title(f"F1 vs Threshold (best={metrics['best_t']:.2f})")

    # 5. Probability Distribution
    ax = axes[1, 1]
    # y_test needs to be passed alongside metrics — stored externally
    ax.set_title("Probability Distribution")

    # 6. Feature Importance
    ax = axes[1, 2]
    if hasattr(model, "feature_importances_") and feature_cols is not None:
        fi   = model.feature_importances_
        top  = np.argsort(fi)[-10:]
        ax.barh(range(len(top)), fi[top])
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(np.array(feature_cols)[top])
        ax.set_title("Top-10 Feature Importances")
    else:
        ax.axis("off")

    plt.suptitle(f"{title} Performance Dashboard", fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_dashboard_dl(metrics:   dict,
                      y_true:    np.ndarray,
                      y_prob:    np.ndarray,
                      history,
                      title:     str = "DL Model",
                      save_path: str = None) -> None:
    """2×4 performance dashboard for deep-learning models."""
    print(f"\n=== {title} Summary ===")
    print(f"  AUC Score      : {metrics.get('roc_auc', 'N/A')}")
    print(f"  Best Threshold : {metrics['best_t']:.3f}")
    print(f"  Best F1 Score  : {metrics['best_f1']:.3f}")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    hist = history.history

    # 1. Loss
    axes[0, 0].plot(hist["loss"],     label="train")
    axes[0, 0].plot(hist["val_loss"], label="val")
    axes[0, 0].set_title("Loss"); axes[0, 0].legend()

    # 2. Accuracy
    axes[0, 1].plot(hist["accuracy"],     label="train")
    axes[0, 1].plot(hist["val_accuracy"], label="val")
    axes[0, 1].set_title("Accuracy"); axes[0, 1].legend()

    # 3. Confusion Matrix
    _safe_heatmap(axes[0, 2], metrics["cm"])

    # 4. ROC
    if metrics.get("fpr") is not None:
        axes[0, 3].plot(metrics["fpr"], metrics["tpr"],
                        label=f"AUC={metrics['roc_auc']:.3f}")
        axes[0, 3].plot([0, 1], [0, 1], "--")
    axes[0, 3].set_title("ROC Curve"); axes[0, 3].legend()

    # 5. Precision-Recall
    if metrics.get("pr_recall") is not None:
        axes[1, 0].plot(metrics["pr_recall"], metrics["pr_precision"])
    axes[1, 0].set_title("Precision-Recall")

    # 6. F1 vs Threshold
    axes[1, 1].plot(metrics["thresholds"], metrics["f1_scores"])
    axes[1, 1].axvline(metrics["best_t"], color="red", linestyle="--")
    axes[1, 1].set_title(f"F1 vs Threshold (best={metrics['best_t']:.2f})")

    # 7. Probability Distribution
    axes[1, 2].hist(y_prob[y_true == 0], bins=30, alpha=0.5, label="Non Fatigue")
    axes[1, 2].hist(y_prob[y_true == 1], bins=30, alpha=0.5, label="Fatigue")
    axes[1, 2].set_title("Probability Distribution"); axes[1, 2].legend()

    # 8. Prediction Timeline (first 200 samples)
    y_pred = metrics.get("y_pred", (y_prob > metrics["best_t"]).astype(int))
    axes[1, 3].plot(y_true[:200],         label="True")
    axes[1, 3].plot(y_pred[:200], alpha=0.7, label="Pred")
    axes[1, 3].set_title("Prediction Timeline"); axes[1, 3].legend()

    plt.suptitle(f"{title} Performance Dashboard", fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [utils] Figure saved → {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model results registry
# ─────────────────────────────────────────────────────────────────────────────

def _to_float(x):
    """Safely convert Tensor / np scalar / float to Python float."""
    try:
        if hasattr(x, "numpy"):  return float(x.numpy())
        if hasattr(x, "item"):   return float(x.item())
        return float(x)
    except Exception:
        return None


def model_results_append(model_results:  list,
                         train_metrics:  dict = None,
                         test_metrics:   dict = None,
                         model_name:     str  = "Model",
                         mode:           str  = "ml",
                         history=None,
                         metrics:        dict = None) -> None:
    """
    Append one model's results to the global model_results list.

    Parameters
    ----------
    mode : 'ml' (classic ML) or 'dl' (deep learning with history object)
    """
    if mode == "ml":
        model_results.append({
            "model":          model_name,
            "train_acc":      _to_float(train_metrics["acc"]),
            "test_acc":       _to_float(test_metrics["acc"]),
            "train_precision":_to_float(train_metrics["precision"]),
            "test_precision": _to_float(test_metrics["precision"]),
            "train_recall":   _to_float(train_metrics["recall"]),
            "test_recall":    _to_float(test_metrics["recall"]),
            "train_f1":       _to_float(train_metrics["f1"]),
            "test_f1":        _to_float(test_metrics["f1"]),
            "test_auc":       _to_float(test_metrics.get("roc_auc")),
        })

    elif mode == "dl":
        hist = history.history
        model_results.append({
            "model":          model_name,
            "train_acc":      _to_float(hist.get("accuracy",  [None])[-1]),
            "train_precision":_to_float(hist.get("precision", [None])[-1]),
            "train_recall":   _to_float(hist.get("recall",    [None])[-1]),
            "train_f1":       _to_float(hist.get("f1_metric", [None])[-1]),
            "train_loss":     _to_float(hist.get("loss",      [None])[-1]),
            "val_acc":        _to_float(hist.get("val_accuracy",  [None])[-1]),
            "val_precision":  _to_float(hist.get("val_precision", [None])[-1]),
            "val_recall":     _to_float(hist.get("val_recall",    [None])[-1]),
            "val_f1":         _to_float(hist.get("val_f1_metric", [None])[-1]),
            "val_loss":       _to_float(hist.get("val_loss",  [None])[-1]),
            "test_acc":       _to_float(metrics["acc"]),
            "test_precision": _to_float(metrics["precision"]),
            "test_recall":    _to_float(metrics["recall"]),
            "test_f1":        _to_float(metrics["f1"]),
            "test_auc":       _to_float(metrics.get("roc_auc")),
            "best_threshold": _to_float(metrics.get("best_t")),
            "best_f1":        _to_float(metrics.get("best_f1")),
        })
    else:
        raise ValueError("mode must be 'ml' or 'dl'")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAP / LIME explainability
# ─────────────────────────────────────────────────────────────────────────────

def run_shap(model,
             X_seq_train: np.ndarray,
             X_seq_test:  np.ndarray,
             seq_len:     int,
             n_features:  int,
             feature_cols: list,
             n_background: int = 100,
             n_explain:    int = 100,
             save_dir:     str = "outputs/plots") -> np.ndarray:
    """
    Compute SHAP values for a DL model and plot aggregated feature importance.

    Returns
    -------
    shap_values : np.ndarray  shape (n_explain, seq_len * n_features)
    """
    try:
        import shap
    except ImportError:
        print("[utils] SHAP not installed — skipping. pip install shap")
        return None

    background  = X_seq_train[
        np.random.choice(X_seq_train.shape[0], n_background, replace=False)
    ]

    def model_predict(x):
        x_r = x.reshape((-1, seq_len, n_features))
        return model.predict(x_r, verbose=0)

    explainer   = shap.KernelExplainer(
        model_predict,
        background.reshape((n_background, -1)),
    )
    X_sample     = X_seq_test[:n_explain]
    X_sample_flat = X_sample.reshape((n_explain, -1))
    shap_values  = explainer.shap_values(X_sample_flat, nsamples=100)

    # Aggregate over time steps
    sv_arr = np.array(shap_values).reshape(-1, seq_len, n_features)
    fi     = np.mean(np.abs(sv_arr), axis=(0, 1))

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(n_features), fi)
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
    ax.set_title("SHAP Feature Importance (Aggregated)")
    ax.set_ylabel("Mean |SHAP value|")
    plt.tight_layout()
    sp = os.path.join(save_dir, "shap_importance.png")
    fig.savefig(sp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[utils] SHAP plot saved → {sp}")

    return shap_values


def run_lime(model,
             X_seq_train: np.ndarray,
             X_seq_test:  np.ndarray,
             seq_len:     int,
             n_features:  int,
             instance_idx: int = 0,
             save_dir:    str = "outputs/plots") -> None:
    """
    Generate a LIME explanation for a single test instance.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        print("[utils] LIME not installed — skipping. pip install lime")
        return

    X_train_flat = X_seq_train.reshape((X_seq_train.shape[0], -1))
    feat_names   = [f"f{i}" for i in range(X_train_flat.shape[1])]

    explainer = LimeTabularExplainer(
        X_train_flat,
        mode="classification",
        feature_names=feat_names,
        class_names=CLASS_NAMES,
        discretize_continuous=True,
    )

    def predict_fn(x):
        x_r   = x.reshape((-1, seq_len, n_features))
        probs = model.predict(x_r, verbose=0)
        return np.hstack([1 - probs, probs])

    X_test_flat = X_seq_test.reshape((X_seq_test.shape[0], -1))
    exp         = explainer.explain_instance(
        X_test_flat[instance_idx], predict_fn, num_features=10
    )
    lime_weights = dict(exp.as_list())

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(list(lime_weights.keys()), list(lime_weights.values()))
    ax.set_title(f"LIME Explanation (instance {instance_idx})")
    ax.set_xlabel("Contribution")
    plt.tight_layout()
    sp = os.path.join(save_dir, "lime_explanation.png")
    fig.savefig(sp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[utils] LIME plot saved → {sp}")
