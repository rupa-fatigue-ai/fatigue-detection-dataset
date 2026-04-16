"""
main.py
=======
One-command entry point for the complete Worker Fatigue Detection pipeline.

Usage
-----
  # Full run (ML + DL + cGAN augmentation):
  python main.py --dataset_csv dataset/workers_dataset.csv \\
                 --marker_csv  dataset/marker_info.csv \\
                 --output_dir  outputs

  # ML only (much faster):
  python main.py --dataset_csv dataset/workers_dataset.csv \\
                 --marker_csv  dataset/marker_info.csv \\
                 --output_dir  outputs --skip_dl

  # Skip cGAN (DL models only, no augmentation):
  python main.py ... --skip_cgan

  # Run SHAP + LIME explainability after training:
  python main.py ... --explain
"""

import os
import sys
import time
import argparse
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def banner(text: str, char: str = "═") -> None:
    width = 64
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{text}{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}")


def main():
    parser = argparse.ArgumentParser(
        description="Worker Fatigue Detection — Full Pipeline"
    )
    parser.add_argument("--dataset_csv",
                        default="dataset/workers_dataset.csv",
                        help="Path to workers_dataset.csv")
    parser.add_argument("--marker_csv",
                        default="dataset/marker_info.csv",
                        help="Path to marker_info.csv")
    parser.add_argument("--output_dir",
                        default="outputs",
                        help="Base output directory for models, plots, results")
    parser.add_argument("--skip_dl",
                        action="store_true",
                        help="Train only classical ML models (fast mode)")
    parser.add_argument("--skip_cgan",
                        action="store_true",
                        help="Skip cGAN augmentation step")
    parser.add_argument("--explain",
                        action="store_true",
                        help="Run SHAP + LIME explainability on the best DL model")
    parser.add_argument("--feature_matrix_cache",
                        default=None,
                        help="Load pre-computed feature_matrix.csv instead of "
                             "re-running extraction (saves time on re-runs)")
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — Data loading
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 1 — DATA LOADING")

    from data_loader import load_data
    data = load_data(
        dataset_csv=args.dataset_csv,
        marker_csv=args.marker_csv,
    )

    # Cache loaded data
    loaded_path = os.path.join(args.output_dir, "loaded_data.csv")
    data.to_csv(loaded_path, index=False)
    print(f"\n  Loaded data cached → {loaded_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — Feature extraction
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 2 — FEATURE EXTRACTION")

    fm_path = os.path.join(args.output_dir, "feature_matrix.csv")

    if args.feature_matrix_cache and os.path.exists(args.feature_matrix_cache):
        print(f"  Loading cached feature matrix from {args.feature_matrix_cache}")
        feature_matrix = pd.read_csv(args.feature_matrix_cache)
    elif os.path.exists(fm_path):
        print(f"  Loading cached feature matrix from {fm_path}")
        feature_matrix = pd.read_csv(fm_path)
    else:
        from feature_extraction import extract_features, quality_check
        feature_matrix = extract_features(data)
        quality_check(feature_matrix)
        feature_matrix.to_csv(fm_path, index=False)
        print(f"\n  Feature matrix saved → {fm_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Preprocessing
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 3 — PREPROCESSING")

    from preprocessing import preprocess
    prep = preprocess(feature_matrix)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — Training
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 4 — MODEL TRAINING")

    from training import train_all
    train_output = train_all(
        prep,
        output_dir=args.output_dir,
        skip_dl=args.skip_dl,
        skip_cgan=args.skip_cgan,
    )

    model_results = train_output["model_results"]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — Results & visualisation
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 5 — RESULTS & VISUALISATION")

    # Pick best DL model for explainability (highest test_f1 among DL)
    best_dl = None
    if not args.skip_dl and train_output["trained_dl"]:
        import numpy as np
        dl_names = list(train_output["trained_dl"].keys())
        dl_res   = [r for r in model_results if r.get("model") in dl_names]
        if dl_res:
            best_name = max(dl_res, key=lambda r: r.get("test_f1", 0))["model"]
            best_dl   = train_output["trained_dl"].get(best_name)
            print(f"\n  Best DL model for explainability: {best_name}")

    from results import generate_results
    comparison_df = generate_results(
        model_results,
        output_dir=args.output_dir,
        prep=prep,
        best_dl_model=best_dl,
        run_explain=(args.explain and best_dl is not None),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    banner("PIPELINE COMPLETE")
    print(f"\n  Total time : {elapsed / 60:.1f} min  ({elapsed:.0f} s)")
    print(f"  Output dir : {os.path.abspath(args.output_dir)}\n")

    print("  Key files:")
    artefacts = [
        os.path.join(args.output_dir, "loaded_data.csv"),
        os.path.join(args.output_dir, "feature_matrix.csv"),
        os.path.join(args.output_dir, "model_results.csv"),
        os.path.join(args.output_dir, "model_comparison_full.csv"),
        os.path.join(args.output_dir, "model_comparison_paper.csv"),
        os.path.join(args.output_dir, "plots", "comparison_bar.png"),
        os.path.join(args.output_dir, "plots", "comparison_radar_line.png"),
        os.path.join(args.output_dir, "plots", "metrics_heatmap.png"),
    ]
    for f in artefacts:
        status = "✓" if os.path.exists(f) else "✗ (not generated)"
        print(f"    {status}  {f}")

    print()


if __name__ == "__main__":
    main()
