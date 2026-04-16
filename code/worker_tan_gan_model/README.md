# Worker Fatigue Detection using TAN + cGAN

## Overview

This module implements a multimodal fatigue detection system for industrial workers using physiological and environmental signals. The framework integrates Temporal Attention Networks (TAN) with Conditional Generative Adversarial Networks (cGAN) to improve classification performance under data imbalance.

The system processes multimodal signals including ECG, EEG, GSR, and environmental factors to predict fatigue states.

---

## Key Features

* Multimodal signal processing (ECG, EEG, GSR)
* Sliding-window feature extraction
* Temporal Attention Network (TAN)
* Conditional GAN (cGAN) for data augmentation
* Classical ML baselines (RF, SVM, Logistic Regression)
* Explainability support (SHAP, LIME)

---

## Dataset

The dataset used in this work is available at:

https://drive.google.com/drive/folders/133d4LgHQ6PHoftxWmM_0G7kPBhQPWX3R

It includes synchronized physiological and environmental signals with fatigue labels.

---

## Installation

Install dependencies using:

pip install -r requirements.txt

---

## Usage

### Full Pipeline

python main.py 
--dataset_csv path_to_dataset.csv 
--marker_csv path_to_marker.csv 
--output_dir outputs

### ML Only

python main.py 
--dataset_csv path_to_dataset.csv 
--marker_csv path_to_marker.csv 
--output_dir outputs 
--skip_dl

### Without cGAN

python main.py 
--dataset_csv path_to_dataset.csv 
--marker_csv path_to_marker.csv 
--output_dir outputs 
--skip_cgan

---

## Repository Structure

* main.py → Full pipeline execution
* data_loader.py → Data loading and labeling
* feature_extraction.py → Signal feature extraction
* preprocessing.py → Data preprocessing and splitting
* model.py → ML, TAN, and cGAN models
* training.py → Model training routines
* results.py → Evaluation and visualization
* utils.py → Utility functions
* demo_worker_tan_gan.ipynb → Notebook demonstration

---

## Output

The pipeline generates:

* Model performance metrics (Accuracy, Precision, Recall, F1)
* Comparison tables
* Visualization plots
* Saved trained models

---

## Reproducibility

* Fixed random seeds ensure reproducibility
* Worker-level train-test split avoids data leakage
* Modular pipeline enables independent execution

---

## Citation

If you use this work, please cite:

Rupa Verma,
"Multimodal Fatigue Detection using TAN and cGAN",
GitHub Repository, 2026.

---

## License

This project is intended for academic and research purposes only.

