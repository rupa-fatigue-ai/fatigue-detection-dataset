# ============================================================
# Imports (same as notebook)
# ============================================================

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.preprocessing import preprocess_signals
from src.feature_extraction import create_feature_windows
from src.train_model import train_svm, train_decision_tree, StandaloneRNN, StandaloneLSTM, DAN


# ============================================================
# Dataset path
# ============================================================

FOLDER_PATH = "data/"

file_list = sorted(
    glob.glob(os.path.join(FOLDER_PATH, "drive*.csv"))
)

print(f"Files found: {len(file_list)}")

# ============================================================
# Load each driver file
# ============================================================

dfs = []

for file in file_list:

    driver_id = int(
        os.path.basename(file)
        .replace("drive", "")
        .replace(".csv", "")
    )

    df = pd.read_csv(file)

    # Fix column name inconsistency in driver 13
    if "ECG-mV" in df.columns:
        df = df.rename(columns={"ECG-mV": "ECG"})

    # Drivers 14 & 24 have EMG instead of EEG
    if "EEG" not in df.columns:
        df["EEG"] = np.nan

    df = df[["ECG", "EEG", "GSR"]]

    df["driver"] = driver_id

    dfs.append(df)


data = pd.concat(dfs, ignore_index=True)

print("Data loaded")
print("Drivers:", data["driver"].nunique())
print("Rows:", len(data))


# ============================================================
# Load questionnaire labels
# ============================================================

questionnaire = pd.read_csv(
    f"{FOLDER_PATH}Questionnaire_VS_Model.csv"
)

questionnaire["driver"] = (
    questionnaire["Sub ID"]
    .str.extract(r"(\d+)$")
    .astype(int) % 100
)

label_lookup = dict(
    zip(
        questionnaire["driver"],
        questionnaire["MFI20 Questionnaire"]
    )
)

data["label"] = data["driver"].map(label_lookup)


# ============================================================
# Drop driver 3 (same as notebook)
# ============================================================

data = data[data["driver"] != 3].reset_index(drop=True)


# ============================================================
# Impute missing values per driver
# ============================================================

for driver in sorted(data["driver"].unique()):

    mask = data["driver"] == driver

    for col in ["ECG", "EEG", "GSR"]:

        n_nan = data.loc[mask, col].isna().sum()

        if n_nan == 0:
            continue

        total = mask.sum()

        pct = n_nan / total * 100

        if pct == 100:

            data.loc[mask, col] = 0.0

        else:

            data.loc[mask, col] = (
                data.loc[mask, col]
                .interpolate(method="linear")
                .ffill()
                .bfill()
            )


print("Remaining NaNs:",
      data[["ECG","EEG","GSR"]].isna().sum().sum())


# ============================================================
# Preprocessing
# ============================================================

print("Running preprocessing...")

data = preprocess_signals(data)


# ============================================================
# Feature Extraction
# ============================================================

print("Extracting features...")

windows_df = create_feature_windows(
    data,
    window_seconds=10,
    overlap=0.5
)

feature_cols = [
    c for c in windows_df.columns
    if c not in ["driver", "label"]
]

print("Windows extracted:", len(windows_df))
print("Features per window:", len(feature_cols))


# ============================================================
# Prepare ML dataset
# ============================================================

X = windows_df[feature_cols]

y = windows_df["label"]


# ============================================================
# Train test split
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# Train baseline models
# ============================================================

print("Training SVM...")

svm_model, svm_acc, svm_prec, svm_rec, svm_f1, svm_cm = train_svm(
    X_train, y_train,
    X_test, y_test
)

print("Training Decision Tree...")

dt_model, dt_acc, dt_prec, dt_rec, dt_f1, dt_cm = train_decision_tree(
    X_train, y_train,
    X_test, y_test
)


print("\nML Training complete")



# ============================================================
# Standardization + PCA 
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=10)

X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# Convert to sequence format (required for RNN/LSTM)
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

X_train_seq = torch.tensor(
    X_train, dtype=torch.float32
).unsqueeze(1)

X_test_seq = torch.tensor(
    X_test, dtype=torch.float32
).unsqueeze(1)

y_train_t = torch.tensor(
    y_train.values,
    dtype=torch.long
)

y_test_t = torch.tensor(
    y_test.values,
    dtype=torch.long
)

train_dataset = TensorDataset(X_train_seq, y_train_t)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

criterion = nn.CrossEntropyLoss()


# ============================================================
# GRID SEARCH PARAMETERS (same structure as notebook)
# ============================================================

param_grid = {
    "hidden_dim": [64, 128],
    "dropout": [0.2, 0.3],
    "lr": [0.001]
}


# ============================================================
# Train Standalone BiRNN
# ============================================================

print("\nTraining Standalone BiRNN...")

birnn_model = StandaloneRNN(
    input_dim=X_train_seq.shape[2]
).to(device)

optimizer = torch.optim.Adam(
    birnn_model.parameters(),
    lr=0.001
)

for epoch in range(50):

    birnn_model.train()

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        preds = birnn_model(xb)

        loss = criterion(preds, yb)

        loss.backward()

        optimizer.step()

print("BiRNN training complete")


# ============================================================
# Train Standalone BiLSTM
# ============================================================

print("\nTraining Standalone BiLSTM...")

bilstm_model = StandaloneLSTM(
    input_dim=X_train_seq.shape[2]
).to(device)

optimizer = torch.optim.Adam(
    bilstm_model.parameters(),
    lr=0.001
)

for epoch in range(50):

    bilstm_model.train()

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        preds = bilstm_model(xb)

        loss = criterion(preds, yb)

        loss.backward()

        optimizer.step()

print("BiLSTM training complete")


# ============================================================
# Train DAN Model
# ============================================================

print("\nTraining DAN model...")

dan_model = DAN(
    input_dim=X_train_seq.shape[2]
).to(device)

optimizer = torch.optim.Adam(
    dan_model.parameters(),
    lr=0.001
)

for epoch in range(300):

    dan_model.train()

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        preds, embedding = dan_model(xb)

        loss = criterion(preds, yb)

        loss.backward()

        optimizer.step()

print("DAN training complete")


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, X_test_seq, y_test):

    model.eval()

    with torch.no_grad():

        preds = model(X_test_seq.to(device))

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = preds.argmax(dim=1).cpu().numpy()

    from sklearn.metrics import accuracy_score

    return accuracy_score(y_test, preds)


birnn_acc = evaluate_model(
    birnn_model,
    X_test_seq,
    y_test
)

bilstm_acc = evaluate_model(
    bilstm_model,
    X_test_seq,
    y_test
)

dan_acc = evaluate_model(
    dan_model,
    X_test_seq,
    y_test
)

print("\nDeep Model Results")
print("Saved in results folder")
