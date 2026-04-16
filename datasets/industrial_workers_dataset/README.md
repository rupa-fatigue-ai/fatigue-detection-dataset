# Industrial Worker Fatigue Dataset

## Description

This dataset contains multimodal physiological and environmental signals collected from industrial workers for fatigue detection. It is designed to support machine learning and deep learning models for fatigue prediction and analysis.

## Modalities Included

* ECG (Electrocardiogram)
* EEG (Electroencephalogram)
* GSR (Galvanic Skin Response)
* Environmental signals

## Dataset Structure

* `sample_worker_data.csv` → Sample physiological data from an industrial worker
* `marker_info.csv` → Fatigue labels and annotations

> **Note:** The complete dataset contains data from multiple workers and is large in size. Therefore, only a representative sample is included in this repository.

## Access

Full dataset available at:
https://drive.google.com/drive/folders/133d4LgHQ6PHoftxWmM_0G7kPBhQPWX3R?usp=drive_link

## How to Use

1. Download the full dataset from the Google Drive link above.

2. Extract the dataset to your local system.

3. Ensure the following files are available:

   * `workers_dataset.csv` (or equivalent combined dataset file)
   * `marker_info.csv`

4. Run the model using:

```
python main.py --dataset_csv path/to/workers_dataset.csv --marker_csv path/to/marker_info.csv --output_dir results/
```

## Run Example

```
python main.py \
  --dataset_csv ../../datasets/industrial_workers_dataset/sample_worker_data.csv \
  --marker_csv ../../datasets/industrial_workers_dataset/marker_info.csv \
  --output_dir results/
```

## Dataset (Code Integration)

This model uses the Industrial Workers Fatigue Dataset.

* Sample dataset available in:
  `../../datasets/industrial_workers_dataset/`

* Full dataset:
  https://drive.google.com/drive/folders/133d4LgHQ6PHoftxWmM_0G7kPBhQPWX3R?usp=drive_link

## Notes

* Used in TAN + cGAN-based fatigue detection model
* Data has been preprocessed and synchronized
* Suitable for both classical machine learning and deep learning approaches

## License

This dataset is intended for academic and research purposes only. Please contact the authors for any commercial use.

## Acknowledgment

If you use this dataset in your research, please cite the corresponding paper or repository.
