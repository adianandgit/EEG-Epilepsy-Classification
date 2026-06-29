import torch
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_PATH = "./dataset_storage/MyNewEEG_Trusted"
TEST_FILE = "test.pt"
PLOT_PATH = "./sample_waveform_plots" # New folder for these plots
os.makedirs(PLOT_PATH, exist_ok=True)

# The samples you want to investigate
SAMPLE_INDICES = [12, 26, 32] 
# ---------------------

# --- Dataset Class (to load the data) ---
class EEGDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.samples = data['samples'].numpy() # Raw numpy data
        self.labels = data['labels'].numpy()
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        # Return the raw sample and label
        return self.samples[idx], self.labels[idx]

# --- Main Script ---
print(f"Loading test data from {os.path.join(DATA_PATH, TEST_FILE)}...")
try:
    test_ds = EEGDataset(os.path.join(DATA_PATH, TEST_FILE))
    print("Dataset loaded.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure 'MyNewEEG_Trusted/test.pt' exists.")
    sys.exit(1)

print(f"Generating plots for samples: {SAMPLE_INDICES}...")

for sample_idx in SAMPLE_INDICES:
    if sample_idx >= len(test_ds):
        print(f"Warning: Sample index {sample_idx} is out of bounds. Skipping.")
        continue
        
    # Get the specific sample and its label
    sample_data, sample_label = test_ds[sample_idx]
    
    # Get just the first channel to plot
    # sample_data shape is (4, 750)
    signal_to_plot = sample_data[0, :]
    time_axis = np.arange(len(signal_to_plot))
    
    label_str = "Abnormal" if sample_label == 1 else "Normal"
    
    # Create the plot
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, signal_to_plot, color='crimson' if sample_label == 1 else 'royalblue')
    plt.title(f"Waveform for Sample {sample_idx} | True Label: {label_str}", fontsize=14)
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.5)
    
    # Save the plot
    save_path = os.path.join(PLOT_PATH, f"sample_{sample_idx}_waveform.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Saved plot for sample {sample_idx} to {save_path}")

print("All plots generated.")