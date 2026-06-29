import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist # For finding neighbors
import warnings

# --- Suppress warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. Imports ---
from new_model import NewTransferModel

# --- 2. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./dataset_storage/MyNewEEG_Trusted"
PLOT_PATH = "./xai_validation_plots"
os.makedirs(PLOT_PATH, exist_ok=True)

# --- Model Path (We use the STABLE model for this analysis) ---
STABLE_MODEL_PATH = "best_trusted_model.pt"
NUM_NEIGHBORS_TO_PLOT = 5 # For the waveform-only grid
NUM_SALIENCY_NEIGHBORS = 3 # For the new 2x2 saliency grid

print(f"Using device: {DEVICE}")

# --- Global Plot Style ---
plt.rcParams.update({
    'axes.titlepad': 12, 'axes.labelpad': 8, 'figure.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1, 'savefig.facecolor': 'white',
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10, 'legend.fontsize': 10,
})

def save_tight_figure(fig, path, dpi=300):
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=dpi, facecolor='white')
    plt.close(fig)

# --- 3. Dataset Class ---
class EEGDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.samples = data['samples'].float()
        self.labels = data['labels'].long()
        self.raw_numpy_samples = data['samples'].numpy()
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.samples[idx], self.labels[idx]

# --- 4. Helper Functions ---
def load_model(path, model_class, num_classes=2):
    model = model_class(num_new_classes=num_classes).to(DEVICE)
    if not os.path.exists(path):
        print(f"FATAL ERROR: Model file not found at {path}"); sys.exit(1)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model: {path}")
    return model

def get_features_and_predictions(model, dataloader):
    print(f"Extracting features & predictions...")
    all_features, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(DEVICE)
            features = model.extract_features(data)
            outputs = model.feedforward_net(features)
            preds = torch.argmax(outputs, dim=1)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    print(f"Done. Feature shape: {all_features.shape}")
    return all_features, all_labels, all_preds

def compute_saliency_map(model, input_tensor):
    """ Computes a vanilla saliency map (gradient) for a 1D signal. """
    model.eval()
    input_tensor.requires_grad = True
    output = model(input_tensor)
    pred_class = torch.argmax(output, dim=1)
    score = output[:, pred_class].squeeze()
    model.zero_grad()
    score.backward(retain_graph=True)
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
    saliency_1d = np.sum(saliency, axis=0)
    saliency_1d = MinMaxScaler().fit_transform(saliency_1d.reshape(-1, 1)).flatten()
    input_tensor.grad.zero_()
    input_tensor.requires_grad = False
    return saliency_1d

def plot_single_saliency(ax, signal, saliency, title, pred_label, true_label, color='crimson'):
    """ Helper function to draw one saliency plot on a given axis. """
    time_axis = np.arange(signal.shape[0])
    ax.plot(time_axis, signal, color='royalblue', alpha=0.7, label='Raw EEG (Ch 0)')
    ax_twin = ax.twinx() # Create a second y-axis
    ax_twin.plot(time_axis, saliency, color=color, label='Model Focus', linewidth=2)
    ax_twin.fill_between(time_axis, 0, saliency, color=color, alpha=0.2)
    ax_twin.set_ylim(0, 1.05)
    ax.set_title(f"{title}\nPred: {pred_label} (True: {true_label})", fontsize=10)
    ax.set_ylabel('Amplitude', color='royalblue')
    ax_twin.set_ylabel('Saliency', color=color)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax_twin.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=8)

def plot_waveform_grid(main_sample_idx, neighbor_indices, val_ds, title, filename):
    """ Plots the main sample and its neighbors (WAVEFORMS ONLY). """
    print(f"Plotting waveform grid for '{title}'...")
    main_sample = val_ds.raw_numpy_samples[main_sample_idx]
    main_label = val_ds.labels[main_sample_idx].item()
    neighbor_samples = val_ds.raw_numpy_samples[neighbor_indices]
    neighbor_labels = val_ds.labels[neighbor_indices].numpy()

    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=24, y=1.02)
    time_axis = np.arange(main_sample.shape[1])
    
    # Plot 1: The Main "Confusing" Sample
    ax = axes[0]
    main_color = 'crimson' if main_label == 1 else 'royalblue'
    main_label_str = 'ABNORMAL' if main_label == 1 else 'NORMAL'
    ax.plot(time_axis, main_sample[0, :], color=main_color, linewidth=2)
    ax.set_title(f"Main Sample (Index: {main_sample_idx}) | TRUE LABEL: {main_label_str}", 
                 fontsize=14, color=main_color)
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.5)

    # Plot 2-6: The Neighbors
    for i in range(NUM_NEIGHBORS_TO_PLOT):
        ax = axes[i+1]
        sample = neighbor_samples[i]
        label = neighbor_labels[i]
        neighbor_color = 'crimson' if label == 1 else 'royalblue'
        neighbor_label_str = 'ABNORMAL' if label == 1 else 'NORMAL'
        ax.plot(time_axis, sample[0, :], color=neighbor_color, linewidth=1.5, alpha=0.8)
        ax.set_title(f"Neighbor {i+1} (Index: {neighbor_indices[i]}) | TRUE LABEL: {neighbor_label_str}", 
                     fontsize=14, color=neighbor_color)
        ax.grid(True, alpha=0.5)
        
    save_tight_figure(fig, os.path.join(PLOT_PATH, filename))
    print(f"Saved {filename}")

# --- NEW FUNCTION: 2x2 Saliency Grid ---
def plot_saliency_neighbor_grid(model, main_sample_idx, neighbor_indices, val_ds, title, filename):
    """
    Plots the main sample AND its 3 neighbors WITH SALIENCY maps
    in a 2x2 grid for visual comparison of model "focus".
    """
    print(f"Plotting saliency grid for '{title}'...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=20, y=1.02)
    sns.set_style("whitegrid")

    # --- Plot 1: The Main "Confusing" Sample ---
    main_label = val_ds.labels[main_sample_idx].item()
    main_pred = preds_val[main_sample_idx].item()
    main_label_str = "ABNORMAL" if main_label == 1 else "NORMAL"
    main_pred_str = "ABNORMAL" if main_pred == 1 else "NORMAL"
    
    sample_tensor, _ = val_ds[main_sample_idx]
    sample_tensor_batch = sample_tensor.unsqueeze(0).to(DEVICE)
    
    saliency = compute_saliency_map(model, sample_tensor_batch.clone())
    raw_signal = sample_tensor_batch.squeeze().cpu().numpy()[0, :] # Ch 0
    
    plot_single_saliency(axes[0], raw_signal, saliency, 
                         f'Main Sample (Idx: {main_sample_idx})', 
                         main_pred_str, main_label_str, 
                         color='crimson' if main_label == 1 else 'royalblue')

    # --- Plot 2-4: The Neighbors ---
    for i in range(NUM_SALIENCY_NEIGHBORS):
        ax = axes[i+1]
        sample_idx = neighbor_indices[i]
        
        label = val_ds.labels[sample_idx].item()
        pred = preds_val[sample_idx].item()
        label_str = "ABNORMAL" if label == 1 else "NORMAL"
        pred_str = "ABNORMAL" if pred == 1 else "NORMAL"
        
        sample_tensor, _ = val_ds[sample_idx]
        sample_tensor_batch = sample_tensor.unsqueeze(0).to(DEVICE)

        saliency = compute_saliency_map(model, sample_tensor_batch.clone())
        raw_signal = sample_tensor_batch.squeeze().cpu().numpy()[0, :]
        
        plot_single_saliency(ax, raw_signal, saliency, 
                             f'Neighbor {i+1} (Idx: {sample_idx})', 
                             pred_str, label_str, 
                             color='crimson' if label == 1 else 'royalblue')

    save_tight_figure(fig, os.path.join(PLOT_PATH, filename))
    print(f"Saved {filename}")

# --- 5. Main XAI Analysis ---
print("\nLoading VALIDATION dataset...")
val_ds = EEGDataset(os.path.join(DATA_PATH, "val.pt"))
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

model_stable = load_model(STABLE_MODEL_PATH, NewTransferModel)
features_val, labels_true, preds_val = get_features_and_predictions(model_stable, val_dl)

# --- PLOT 1: t-SNE "Map" of the Validation Set ---
print("\nRunning t-SNE on validation set...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
tsne_coords = tsne.fit_transform(features_val)
df_tsne = pd.DataFrame(tsne_coords, columns=['x', 'y'])
df_tsne['label'] = labels_true

# --- Find the "Confusing" Samples (from 3072-D space) ---
print("Finding confusing samples based on 3072-D Feature Space...")
dist_matrix = cdist(features_val, features_val, 'euclidean')
k_waveforms = NUM_NEIGHBORS_TO_PLOT
k_saliency = NUM_SALIENCY_NEIGHBORS
neighbor_indices_all_wave = np.argsort(dist_matrix, axis=1)[:, 1:k_waveforms+1]
neighbor_indices_all_sal = np.argsort(dist_matrix, axis=1)[:, 1:k_saliency+1]

case_1_sample_idx = -1
case_1_neighbor_indices_wave = None
case_1_neighbor_indices_sal = None
case_2_sample_idx = -1
case_2_neighbor_indices_wave = None
case_2_neighbor_indices_sal = None

for i in range(len(labels_true)):
    if labels_true[i] == 1: # Abnormal
        my_neighbor_labels = labels_true[neighbor_indices_all_wave[i]]
        if np.sum(my_neighbor_labels) == 0: # All 5 neighbors are Normal
            case_1_sample_idx = i
            case_1_neighbor_indices_wave = neighbor_indices_all_wave[i]
            case_1_neighbor_indices_sal = neighbor_indices_all_sal[i]
            print(f"Found Case 1 (Abnormal-in-Normal): Sample {i}")
            break

for i in range(len(labels_true)):
    if labels_true[i] == 0: # Normal
        my_neighbor_labels = labels_true[neighbor_indices_all_wave[i]]
        if np.sum(my_neighbor_labels) == k_waveforms: # All 5 neighbors are Abnormal
            case_2_sample_idx = i
            case_2_neighbor_indices_wave = neighbor_indices_all_wave[i]
            case_2_neighbor_indices_sal = neighbor_indices_all_sal[i]
            print(f"Found Case 2 (Normal-in-Abnormal): Sample {i}")
            break
        
# --- Create new plot categories ---
df_tsne['plot_category'] = np.where(df_tsne['label'] == 1, 'Abnormal (General)', 'Normal (General)')
category_list = df_tsne['plot_category'].values
if case_1_sample_idx != -1:
    category_list[case_1_sample_idx] = "Isolated Abnormal (Case 1)"
    category_list[case_1_neighbor_indices_wave] = "Neighbors (Normal)"
if case_2_sample_idx != -1:
    category_list[case_2_sample_idx] = "Isolated Normal (Case 2)"
    category_list[case_2_neighbor_indices_wave] = "Neighbors (Abnormal)"
df_tsne['plot_category'] = category_list
palette = {
    "Normal (General)": "#a9d6e5", "Abnormal (General)": "#f7a6a5",
    "Isolated Normal (Case 2)": "#014f86", "Neighbors (Abnormal)": "#9d0208",
    "Isolated Abnormal (Case 1)": "#d00000", "Neighbors (Normal)": "#012a4a"
}

# --- Plot the t-SNE with 6 colors ---
print("Saving t-SNE plot with 6-color categories...")
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
sns.kdeplot(data=df_tsne[df_tsne['label'] == 0], x='x', y='y', color="royalblue", ax=ax, alpha=0.05, levels=4, fill=True, legend=False)
sns.kdeplot(data=df_tsne[df_tsne['label'] == 1], x='x', y='y', color="crimson", ax=ax, alpha=0.05, levels=4, fill=True, legend=False)
sns.scatterplot(data=df_tsne, x='x', y='y', hue='plot_category', palette=palette, ax=ax, s=40, alpha=0.9, hue_order=palette.keys())
ax.set_title('Validation Set Feature Space (t-SNE)', fontsize=20)
ax.legend(loc='best', fontsize=12, frameon=True, shadow=True)
save_tight_figure(fig, os.path.join(PLOT_PATH, 'xai_validation_tsne_map_6color.png'))
print("t-SNE map plot saved.")

# --- PLOT 2: Waveform Grid for Case 1 ---
if case_1_sample_idx != -1:
    plot_waveform_grid(
        main_sample_idx=case_1_sample_idx,
        neighbor_indices=case_1_neighbor_indices_wave,
        val_ds=val_ds,
        title=f"Why is Sample {case_1_sample_idx} (Abnormal) in the Normal Cluster? (True 3072D Neighbors)",
        filename="xai_case1_abnormal_in_normal_WAVEFORMS.png"
    )
    # --- NEW SALIENCY PLOT ---
    plot_saliency_neighbor_grid(
        model=model_stable,
        main_sample_idx=case_1_sample_idx,
        neighbor_indices=case_1_neighbor_indices_sal,
        val_ds=val_ds,
        title=f"Saliency: Why is Sample {case_1_sample_idx} (Abnormal) Confused?",
        filename="xai_case1_abnormal_in_normal_SALIENCY.png"
    )

# --- PLOT 3: Waveform Grid for Case 2 ---
if case_2_sample_idx != -1:
    plot_waveform_grid(
        main_sample_idx=case_2_sample_idx,
        neighbor_indices=case_2_neighbor_indices_wave,
        val_ds=val_ds,
        title=f"Why is Sample {case_2_sample_idx} (Normal) in the Abnormal Cluster? (True 3072D Neighbors)",
        filename="xai_case2_normal_in_abnormal_WAVEFORMS.png"
    )
    # --- NEW SALIENCY PLOT ---
    plot_saliency_neighbor_grid(
        model=model_stable,
        main_sample_idx=case_2_sample_idx,
        neighbor_indices=case_2_neighbor_indices_sal,
        val_ds=val_ds,
        title=f"Saliency: Why is Sample {case_2_sample_idx} (Normal) Confused?",
        filename="xai_case2_normal_in_abnormal_SALIENCY.png"
    )

if case_1_sample_idx == -1 and case_2_sample_idx == -1:
    print("Could not find any 'confusing' samples to plot.")

print("\n--- XAI Validation Analysis Complete ---")