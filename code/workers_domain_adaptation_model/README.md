# Industrial Workers Fatigue Detection using Domain Adaptation

## Description
This project implements a domain adaptation-based approach for fatigue detection in industrial workers using physiological and/or multimodal data. The model improves generalization across different data distributions.

## Dataset
The dataset used in this study is available in this repository:
datasets/worker_dataset/workers_dataset.zip

It includes physiological signals (ECG, EEG, GSR) with fatigue labels.

## Methodology
- Data preprocessing and normalization  
- Feature extraction  
- Domain Adaptation Network (DAN) for cross-domain learning  
- Classification of fatigue states  

## Requirements
Install dependencies using:

pip install -r requirements.txt

Common libraries include:
- numpy  
- pandas  
- scikit-learn  
- matplotlib  
- seaborn  
- tensorflow / pytorch (depending on implementation)

## How to Run
1. Open the notebook:
   DAN_notebook.ipynb  

   OR  

2. Run the pipeline:
   python run_pipeline.py  

## Figures
Model performance visualizations are available in the `figures/` folder.

## Notes
This model is designed for industrial fatigue detection and demonstrates improved performance using domain adaptation techniques for cross-domain scenarios.
