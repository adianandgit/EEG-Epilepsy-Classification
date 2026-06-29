import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import MinMaxScaler

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input and output paths
input_file = os.path.join(BASE_DIR, "data_files", "data.csv")
output_dir = os.path.join(BASE_DIR, "..", "..", "data", "epilepsy")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load data
data = pd.read_csv(input_file)

# Separate features and labels
y = data.iloc[:, -1].to_numpy()
x = data.iloc[:, 1:-1].to_numpy()

# Preprocess
y = y - 1
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = (y != 0).astype(int)  # Convert all non-zero to 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Save in PyTorch format
def save_split(X, y, filename):
    dat_dict = {
        "samples": torch.from_numpy(X).unsqueeze(1),
        "labels": torch.from_numpy(y)
    }
    torch.save(dat_dict, os.path.join(output_dir, filename))

save_split(X_train, y_train, "train.pt")
save_split(X_val, y_val, "val.pt")
save_split(X_test, y_test, "test.pt")
