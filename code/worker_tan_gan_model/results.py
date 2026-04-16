"""
results.py
==========
Loads model_results.csv and generates:
  • Grouped bar chart comparing all models on test metrics
  • Radar chart  +  line chart side-by-side
  • Pretty comparison table (CSV + console print)
  • (Optional) SHAP and LIME explanations for the best DL model
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import seaborn as sns
    _SNS = True
except ImportError:
    _SNS = False

# ── Model rename map (matches notebook conventions) ─────────────────────────
RENAME_MAP = {
    "LSTM_SelfAttention": "TAN(LSTM+Self_attn)",
    "TAN_LSTM_DAN":       "TAN(LSTM+Self_attn+Gen_attn)",
    "baseline_LSTM":      "TAN(LSTM_alone)",
    "cGAN_LSTM":          "LSTM(cGAN_augmented)",
}

# Models to exclude from the comparison table (intermediate runs)
EXCLUDE_MODELS = {"TAN_v1", "TAN_v2", "cGAN+LSTM", "LSTM"}

METRIC_COLORS  = ["#6366F1", "#EC4899", "#F97316", "#22C55E"]
TEST_COLS      = ["test_acc", "test_precision", "test_recall", "test_f1"]
TEST_LABELS    = ["Accuracy", "Precision", "Recall", "F1-Score"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build comparison table
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(model_results: list) -> pd.DataFrame:
    """
    Build a clean, ranked comparison DataFrame from the raw model_results list.

    Applies:
      • column selection
      • rename map
      • numeric conversion
      • sort by test_f1
      • rank column
    """
    df = pd.DataFrame(model_results)

    keep_cols = [
        "model",
        "train_acc", "train_precision", "train_recall", "train_f1",
        "val_acc",
        "test_acc",  "test_precision",  "test_recall",  "test_f1",
        "test_auc",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    df["model"] = df["model"].replace(RENAME_MAP)

    for col in df.columns:
        if col != "model":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.round(4)
    if "test_f1" in df.columns:
        df = df.sort_values("test_f1", ascending=False).reset_index(drop=True)
        df["Rank"] = df["test_f1"].rank(ascending=False, method="min").astype("Int64")

    return df


def filter_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    """Remove intermediate / duplicate model entries for the paper table."""
    excluded = {RENAME_MAP.get(m, m) for m in EXCLUDE_MODELS}
    return df[~df["model"].isin(excluded)].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar_comparison(df:       pd.DataFrame,
                        save_dir: str = "outputs/plots") -> None:
    """
    Grouped bar chart — Test Accuracy / Precision / Recall / F1-Score
    for every model in df.
    """
    models   = df["model"].tolist()
    n_models = len(models)
    x        = np.arange(n_models)
    width    = 0.18

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, (col, label, color) in enumerate(
            zip(TEST_COLS, TEST_LABELS, METRIC_COLORS)):
        if col not in df.columns:
            continue
        vals = df[col].values
        ax.bar(x + i * width, vals, width, color=color,
               label=label, edgecolor="white", linewidth=0.8)

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 1.19)
    ax.set_title("Test Performance — All Models All Metrics",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "comparison_bar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[results] Bar chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Radar + line chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar_line(df:       pd.DataFrame,
                   save_dir: str = "outputs/plots") -> None:
    """
    Left  : Radar chart showing each model on the 4 test metrics.
    Right : Line chart showing metric trajectories across models.
    """
    metrics_present = [c for c in TEST_COLS if c in df.columns]
    labels          = [TEST_LABELS[TEST_COLS.index(c)] for c in metrics_present]
    n_metrics       = len(metrics_present)
    angles          = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles         += angles[:1]

    models   = df["model"].tolist()
    n_models = len(models)
    x        = np.arange(n_models)

    fig = plt.figure(figsize=(18, 8))

    # ── Radar ──────────────────────────────────────────────────────────────
    ax_radar = fig.add_subplot(1, 2, 1, polar=True)
    for _, row in df.iterrows():
        vals  = row[metrics_present].values.tolist()
        vals += vals[:1]
        ax_radar.plot(angles, vals, marker="o", linewidth=1.5, label=row["model"])
        ax_radar.fill(angles, vals, alpha=0.12)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels)
    ax_radar.set_ylim(0, 1.05)
    ax_radar.set_title("Model Test Metrics — Radar Chart", pad=30)
    ax_radar.grid(True, alpha=0.5)
    ax_radar.legend(loc="upper right",
                    bbox_to_anchor=(1.4, 1.05), fontsize=9)

    # ── Line ───────────────────────────────────────────────────────────────
    ax_line = fig.add_subplot(1, 2, 2)
    for col, color, label in zip(metrics_present, METRIC_COLORS, labels):
        ax_line.plot(x, df[col].values,
                     marker="o", linestyle="-", color=color, label=label)
    ax_line.set_xticks(x)
    ax_line.set_xticklabels(models, rotation=25, ha="right")
    ax_line.set_ylim(0, 1.05)
    ax_line.set_title("Model Test Metrics — Line Chart")
    ax_line.set_ylabel("Score")
    ax_line.grid(axis="y", alpha=0.3)
    ax_line.legend(loc="lower left")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "comparison_radar_line.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[results] Radar/line chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Heatmap of all metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_heatmap(df:       pd.DataFrame,
                         save_dir: str = "outputs/plots") -> None:
    """Heat-map of numeric metrics per model."""
    numeric_cols = [c for c in df.columns
                    if c not in ("model", "Rank") and df[c].notna().any()]
    heat_df = df.set_index("model")[numeric_cols]

    fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols) * 0.9),
                                    max(4, len(df) * 0.6)))
    if _SNS:
        sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlGnBu",
                    linewidths=0.5, ax=ax)
    else:
        im = ax.imshow(heat_df.values, cmap="YlGnBu", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["model"].tolist())

    ax.set_title("All-Model Metrics Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "metrics_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[results] Heatmap saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Explainability (SHAP + LIME)
# ─────────────────────────────────────────────────────────────────────────────

def run_explainability(best_model,
                       prep:      dict,
                       save_dir:  str = "outputs/plots") -> None:
    """Run SHAP and LIME on the best DL model (if libraries are available)."""
    from utils import run_shap, run_lime
    from preprocessing import FEATURE_COLS

    SEQ_LEN    = prep["X_seq_train"].shape[1]
    N_FEATURES = prep["X_seq_train"].shape[2]

    print("\n[results] Running SHAP explainer...")
    run_shap(best_model,
             prep["X_seq_train"],
             prep["X_seq_test"],
             SEQ_LEN, N_FEATURES,
             FEATURE_COLS,
             save_dir=save_dir)

    print("[results] Running LIME explainer...")
    run_lime(best_model,
             prep["X_seq_train"],
             prep["X_seq_test"],
             SEQ_LEN, N_FEATURES,
             save_dir=save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Master generate_results function
# ─────────────────────────────────────────────────────────────────────────────

def generate_results(model_results:  list,
                     output_dir:     str  = "outputs",
                     prep:           dict = None,
                     best_dl_model         = None,
                     run_explain:    bool  = False) -> pd.DataFrame:
    """
    Produces all result artefacts from model_results list.

    Parameters
    ----------
    model_results  : accumulated list of per-model metric dicts
    output_dir     : base output directory
    prep           : preprocessing dict (needed for explainability only)
    best_dl_model  : best trained Keras model (for SHAP / LIME)
    run_explain    : if True, run SHAP and LIME (slow)

    Returns
    -------
    comparison_df  : pd.DataFrame — ranked comparison table
    """
    save_dir = os.path.join(output_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── Build tables ─────────────────────────────────────────────────────────
    full_df   = build_comparison_table(model_results)
    paper_df  = filter_for_paper(full_df)

    print("\n" + "=" * 60)
    print("  FULL MODEL COMPARISON TABLE")
    print("=" * 60)
    with pd.option_context("display.max_columns", None,
                           "display.width",       200,
                           "display.float_format", "{:.4f}".format):
        print(full_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("  PAPER TABLE (filtered)")
    print("=" * 60)
    with pd.option_context("display.max_columns", None,
                           "display.width",       200,
                           "display.float_format", "{:.4f}".format):
        print(paper_df.to_string(index=False))

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    full_path  = os.path.join(output_dir, "model_comparison_full.csv")
    paper_path = os.path.join(output_dir, "model_comparison_paper.csv")
    full_df.to_csv(full_path,  index=False)
    paper_df.to_csv(paper_path, index=False)
    print(f"\n[results] Full table  → {full_path}")
    print(f"[results] Paper table → {paper_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    use_df = paper_df if len(paper_df) >= 2 else full_df

    plot_bar_comparison(use_df, save_dir=save_dir)
    plot_radar_line(use_df,     save_dir=save_dir)
    plot_metrics_heatmap(full_df, save_dir=save_dir)

    # ── Explainability ────────────────────────────────────────────────────────
    if run_explain and best_dl_model is not None and prep is not None:
        run_explainability(best_dl_model, prep,
                           save_dir=os.path.join(save_dir, "xai"))

    print("\n[results] ✓ All result artefacts generated.")
    return full_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker Fatigue — Results")
    parser.add_argument("--results_csv", required=True,
                        help="Path to model_results.csv (output of training.py)")
    parser.add_argument("--output_dir",  default="outputs",
                        help="Directory to save plots / tables")
    parser.add_argument("--explain",     action="store_true",
                        help="Run SHAP + LIME (requires feature_matrix.csv "
                             "and a saved model)")
    args = parser.parse_args()

    results_df = pd.read_csv(args.results_csv)
    model_results = results_df.to_dict(orient="records")

    generate_results(model_results,
                     output_dir=args.output_dir,
                     run_explain=args.explain)
