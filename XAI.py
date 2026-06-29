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
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning)

from new_model import NewTransferModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./dataset_storage/MyNewEEG_Trusted"
PLOT_PATH = "./xai_plots"
os.makedirs(PLOT_PATH, exist_ok=True)

MINIMAL_MODEL_PATH = "best_minimal_model.pt"
STABLE_MODEL_PATH = "best_trusted_model.pt"

print(f"Device: {DEVICE}")

plt.rcParams.update({
    'axes.titlepad': 12,
    'axes.labelpad': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.facecolor': 'white',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
})

def save_tight_figure(fig, path, dpi=300):
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=dpi, facecolor='white')
    plt.close(fig)

class EEGDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.samples = data['samples'].float()
        self.labels = data['labels'].long()
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.samples[idx], self.labels[idx]

def load_model(path, model_class, num_classes=2):
    model = model_class(num_new_classes=num_classes).to(DEVICE)
    if not os.path.exists(path):
        print(f"Model not found: {path}")
        sys.exit(1)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model: {path}")
    return model

def get_features_and_predictions(model, dataloader):
    print(f"Extracting features for {type(model).__name__}...")
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
    print(f"Feature shape: {all_features.shape}")
    return all_features, all_labels, all_preds

def compute_saliency_map(model, input_tensor):
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
    t = np.arange(signal.shape[0])
    ax.plot(t, signal, color='royalblue', alpha=0.7, label='Raw EEG (Ch 0)')
    ax_twin = ax.twinx()
    ax_twin.plot(t, saliency, color=color, linewidth=2, label='Model Focus')
    ax_twin.fill_between(t, 0, saliency, color=color, alpha=0.2)
    ax_twin.set_ylim(0, 1.05)
    ax.set_title(f"{title}\nPred: {pred_label} (True: {true_label})", fontsize=10)
    ax.set_ylabel('Amplitude', color='royalblue')
    ax_twin.set_ylabel('Saliency', color=color)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax_twin.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=8)

def save_individual_saliency_plots(model, sample_indices, title_prefix, subfolder, num_to_plot=4):
    if len(sample_indices) == 0:
        print(f"No samples found for {title_prefix}.")
        return
    output_folder = os.path.join(PLOT_PATH, subfolder)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving {title_prefix} plots in {output_folder}...")

    if "TP" in title_prefix: color = 'forestgreen'
    elif "TN" in title_prefix: color = 'darkorange'
    elif "FP" in title_prefix: color = 'crimson'
    else: color = 'purple'

    pred_label_str = "ABNORMAL" if "FP" in title_prefix or "TP" in title_prefix else "NORMAL"
    true_label_str = "ABNORMAL" if "FN" in title_prefix or "TP" in title_prefix else "NORMAL"

    for i, sample_idx in enumerate(sample_indices[:num_to_plot]):
        sample_tensor, _ = test_ds[sample_idx]
        sample_tensor_batch = sample_tensor.unsqueeze(0).to(DEVICE)
        saliency = compute_saliency_map(model, sample_tensor_batch.clone())
        raw_signal_ch0 = sample_tensor_batch.squeeze().cpu().numpy()[0, :]
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        plot_single_saliency(ax, raw_signal_ch0, saliency, f'{title_prefix} (Sample {sample_idx})', pred_label_str, true_label_str, color)
        save_tight_figure(fig, os.path.join(output_folder, f"sample_{sample_idx}.png"))
    print(f"Saved {min(len(sample_indices), num_to_plot)} {title_prefix} plots.")

print("\nLoading test dataset...")
test_ds = EEGDataset(os.path.join(DATA_PATH, "test.pt"))
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

model_minimal = load_model(MINIMAL_MODEL_PATH, NewTransferModel)
model_stable = load_model(STABLE_MODEL_PATH, NewTransferModel)

features_min, labels_true, preds_min = get_features_and_predictions(model_minimal, test_dl)
features_stable, _, preds_stable = get_features_and_predictions(model_stable, test_dl)

print("\nRunning t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
tsne_min = tsne.fit_transform(features_min)
tsne_stable = tsne.fit_transform(features_stable)

df_min = pd.DataFrame(tsne_min, columns=['x', 'y'])
df_min['label'] = ['Abnormal' if l == 1 else 'Normal' for l in labels_true]
df_stable = pd.DataFrame(tsne_stable, columns=['x', 'y'])
df_stable['label'] = ['Abnormal' if l == 1 else 'Normal' for l in labels_true]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
fig.suptitle('Feature Space Comparison (t-SNE)', fontsize=22, y=0.98)
palette = {'Normal': 'royalblue', 'Abnormal': 'crimson'}
sns.scatterplot(data=df_min, x='x', y='y', hue='label', palette=palette, ax=ax1, s=30, alpha=0.7, edgecolor='w', linewidth=0.5)
ax1.set_title('Minimal Model (F1=0.18)')
sns.scatterplot(data=df_stable, x='x', y='y', hue='label', palette=palette, ax=ax2, s=30, alpha=0.7, edgecolor='w', linewidth=0.5)
ax2.set_title('Stable Model (F1=0.70)')
save_tight_figure(fig, os.path.join(PLOT_PATH, 'xai_tsne_comparison.png'))
print("t-SNE plot saved.")

print("\nGenerating saliency comparison...")
comparison_sample_idx = -1
for i in range(len(labels_true)):
    if labels_true[i] == 1 and preds_min[i] == 0 and preds_stable[i] == 1:
        comparison_sample_idx = i
        break
if comparison_sample_idx == -1:
    for i in range(len(labels_true)):
        if labels_true[i] == 1 and preds_stable[i] == 1:
            comparison_sample_idx = i
            break

if comparison_sample_idx != -1:
    sample_tensor, _ = test_ds[comparison_sample_idx]
    sample_tensor = sample_tensor.unsqueeze(0).to(DEVICE)
    saliency_min = compute_saliency_map(model_minimal, sample_tensor.clone())
    saliency_stable = compute_saliency_map(model_stable, sample_tensor.clone())

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    fig.suptitle(f"Saliency Comparison (Sample {comparison_sample_idx}, True: Abnormal)", fontsize=18, y=0.99)
    t = np.arange(sample_tensor.shape[2])
    raw_signal_ch0 = sample_tensor.squeeze().cpu().numpy()[0, :]
    axes[0].plot(t, raw_signal_ch0, color='darkblue', linewidth=1.5)
    axes[0].set_title('Raw Input Signal')
    plot_single_saliency(axes[1], raw_signal_ch0, saliency_min, 'Minimal Model (F1=0.18)', 'Normal', 'Abnormal', color='gray')
    plot_single_saliency(axes[2], raw_signal_ch0, saliency_stable, 'Stable Model (F1=0.70)', 'Abnormal', 'Abnormal', color='crimson')
    save_tight_figure(fig, os.path.join(PLOT_PATH, 'xai_saliency_comparison.png'))
    print("Saliency comparison saved.")
else:
    print("No matching sample found for saliency comparison.")

print("\nSaving individual saliency plots...")
indices = np.arange(len(labels_true))
tp = indices[(labels_true == 1) & (preds_stable == 1)]
tn = indices[(labels_true == 0) & (preds_stable == 0)]
fn = indices[(labels_true == 1) & (preds_stable == 0)]
fp = indices[(labels_true == 0) & (preds_stable == 1)]

save_individual_saliency_plots(model_stable, tp, "True Positive", "saliency_true_positives")
save_individual_saliency_plots(model_stable, tn, "True Negative", "saliency_true_negatives")
save_individual_saliency_plots(model_stable, fn, "False Negative", "saliency_false_negatives")
save_individual_saliency_plots(model_stable, fp, "False Positive", "saliency_false_positives")

print("\nXAI analysis completed.")
print(f"All plots saved to {PLOT_PATH}")
