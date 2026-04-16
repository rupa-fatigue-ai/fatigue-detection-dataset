## Description
This dataset contains multimodal physiological and environmental signals collected from industrial workers for fatigue detection.

Modalities include:
- ECG
- EEG
- GSR
- Environmental signals

## Dataset Structure
- workers_dataset.csv → Physiological features
- marker_info.csv → Fatigue labels

## Access
Dataset available at:
[PASTE YOUR GOOGLE DRIVE LINK HERE]

## Notes
- Used in TAN + cGAN model
- Preprocessed and synchronized

## Access

Dataset available at:  
[Download Industrial Worker Dataset from Google Drive](https://drive.google.com/drive/folders/133d4LgHQ6PHoftxWmM_0G7kPBhQPWX3R?usp=drive_link)

## Usage

After downloading, place the dataset in the following structure:

datasets/
└── industrial_workers_dataset/
    ├── workers_dataset.csv
    └── marker_info.csv

Run the model using:

python main.py --dataset_csv ../../datasets/industrial_workers_dataset/workers_dataset.csv --marker_csv ../../datasets/industrial_workers_dataset/marker_info.csv --output_dir outputs
