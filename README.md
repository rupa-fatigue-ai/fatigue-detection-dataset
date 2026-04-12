# Fatigue Detection Dataset

## Multimodal Physiological Datasets for Fatigue Detection

This repository provides curated multimodal physiological datasets and corresponding machine learning models for fatigue detection research. The datasets include signals such as ECG, EEG, and GSR along with labeled fatigue states, enabling the development and evaluation of predictive models.

---

## Repository Structure

### Datasets

- `datasets/worker_dataset/`  
  Multimodal dataset containing ECG, EEG, and GSR signals with fatigue labels.

  Download link:  
  https://github.com/rupa-fatigue-ai/fatigue-detection-dataset/blob/main/datasets/worker_dataset/workers_dataset.zip

- `datasets/housewife_dataset/`  
  Physiological dataset containing ECG and GSR signals with fatigue labels.

  Download link:  
  https://github.com/rupa-fatigue-ai/fatigue-detection-dataset/blob/main/datasets/housewife_dataset/housewives_dataset.zip

---

### Code

The repository includes machine learning models developed for different subject groups:

- `code/adolescent_model/`  
  Fatigue detection model for adolescents  

- `code/housewife_model/`  
  Fatigue detection model for household subjects  

- `code/workers_domain_adaptation_model/`  
  Domain adaptation-based fatigue detection model for industrial workers  

---

## Usage

Each dataset is organized independently and can be used for training and evaluating machine learning models for fatigue detection.

Detailed instructions for running the models are provided in the respective README files inside each code folder.

---

## Reproducibility

- All datasets are available in the `/datasets` directory  
- Model implementations are provided in the `/code` directory  
- Each model can be executed independently using provided scripts or notebooks  
- Dependencies are listed in the respective `requirements.txt` files  

---

## Author

Rupa Verma  

---

## Citation

If you use this dataset or code in your research, please cite:

Rupa Verma,  
"Multimodal Physiological Dataset for Fatigue Detection",  
GitHub Repository, 2026.  
Available at: https://github.com/rupa-fatigue-ai/fatigue-detection-dataset  

---

## License

This repository is intended for academic and research purposes only.  
Users are requested to provide appropriate credit when using this dataset or code in their work.
