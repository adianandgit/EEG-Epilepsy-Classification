import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score

from new_model import NewTransferModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./dataset_storage/MyNewEEG_Trusted"
PLOT_PATH = "./final_metric_ensemble"
os.makedirs(PLOT_PATH, exist_ok=True)

STABLE_MODEL_PATH = "best_trusted_model.pt"
UNSTABLE_MODEL_PATH = "best_f1_model.pt"
RECALL_HEAD_PATH = os.path.join("final_metric", "best_head_smote.pt")

print(f"Device: {DEVICE}")

class EEGDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.samples = data['samples'].float()
        self.labels = data['labels'].long()
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.samples[idx], self.labels[idx]

print("Loading models...")
model_stable = NewTransferModel(num_new_classes=2).to(DEVICE)
model_stable.load_state_dict(torch.load(STABLE_MODEL_PATH, map_location=DEVICE))
model_stable.eval()
print("Loaded stable model.")

model_unstable = NewTransferModel(num_new_classes=2).to(DEVICE)
model_unstable.load_state_dict(torch.load(UNSTABLE_MODEL_PATH, map_location=DEVICE))
model_unstable.eval()
print("Loaded high-F1 model.")

model_recall_head = NewTransferModel(num_new_classes=2).feedforward_net
model_recall_head.load_state_dict(torch.load(RECALL_HEAD_PATH, map_location=DEVICE))
model_recall_head = model_recall_head.to(DEVICE)
model_recall_head.eval()
print("Loaded recall head.")

def get_all_probabilities(model_stable, model_unstable, model_recall_head, dataloader):
    print("Extracting probabilities...")
    all_labels, p_s, p_u, p_r = [], [], [], []
    feature_extractor = model_stable
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(DEVICE)
            p_s.append(F.softmax(model_stable(data), dim=1)[:, 1].cpu())
            p_u.append(F.softmax(model_unstable(data), dim=1)[:, 1].cpu())
            features = feature_extractor.extract_features(data)
            p_r.append(F.softmax(model_recall_head(features), dim=1)[:, 1].cpu())
            all_labels.append(labels)
    return (torch.cat(p_s).numpy(),
            torch.cat(p_u).numpy(),
            torch.cat(p_r).numpy(),
            torch.cat(all_labels).numpy())

print("\nStage 1: Optimizing ensemble weights on validation set...")
val_ds = EEGDataset(os.path.join(DATA_PATH, "val.pt"))
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
val_p_s, val_p_u, val_p_r, val_labels = get_all_probabilities(model_stable, model_unstable, model_recall_head, val_dl)

best_f1, best_weights = -1.0, (0, 0, 0)
w_stable = 1.0
for w_u in np.arange(0.0, 2.75, 0.25):
    for w_r in np.arange(0.0, 2.75, 0.25):
        if w_u == 0 and w_r == 0:
            continue
        total = w_stable + w_u + w_r
        final_p = (w_stable * val_p_s + w_u * val_p_u + w_r * val_p_r) / total
        f1 = f1_score(val_labels, (final_p > 0.5).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_weights = f1, (w_stable, w_u, w_r)
            print(f"New best weights: {best_weights} -> F1={f1:.4f}")

W_STABLE, W_UNSTABLE, W_RECALL = best_weights
print(f"\nOptimal weights: Stable={W_STABLE}, Unstable={W_UNSTABLE}, Recall={W_RECALL}")
print(f"Best validation F1={best_f1:.4f}")

print("\nStage 2: Evaluating on test set...")
test_ds = EEGDataset(os.path.join(DATA_PATH, "test.pt"))
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
p_s, p_u, p_r, test_labels = get_all_probabilities(model_stable, model_unstable, model_recall_head, test_dl)

total_weight = W_STABLE + W_UNSTABLE + W_RECALL
final_prob = (W_STABLE * p_s + W_UNSTABLE * p_u + W_RECALL * p_r) / total_weight
final_pred = (final_prob > 0.5).astype(int)

print("\nFinal Ensemble Report (Optimal Weights):")
report = classification_report(test_labels, final_pred, output_dict=True, zero_division=0, target_names=["Normal", "Abnormal"])
print(classification_report(test_labels, final_pred, zero_division=0, target_names=["Normal", "Abnormal"]))

f1 = report["Abnormal"]["f1-score"]
acc = report["accuracy"]
cm = confusion_matrix(test_labels, final_pred, labels=[0, 1])
try: auc = roc_auc_score(test_labels, final_prob)
except ValueError: auc = 0.5
print(f"Test Results: Accuracy={acc*100:.2f}%, F1={f1:.4f}, AUROC={auc:.4f}")

print(f"\nSaving plots to {PLOT_PATH}...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
plt.title('Optimal Weighted Ensemble Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, 'confusion_matrix_optimal_ensemble.png'))
plt.close()

fpr, tpr, _ = roc_curve(test_labels, final_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUROC={auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Optimal Ensemble)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, 'roc_curve_optimal_ensemble.png'))
plt.close()

num_points = 40
indices = np.arange(min(len(test_labels), num_points))
df_plot = pd.DataFrame({
    'Sample Index': indices,
    'True Label': ['Abnormal' if test_labels[i] == 1 else 'Normal' for i in indices],
    'Predicted Prob (Abnormal)': np.array(final_prob)[indices],
    'Prediction Correct': np.array(test_labels)[indices] == np.array(final_pred)[indices]
})
plt.figure(figsize=(14, 7))
sns.barplot(data=df_plot, x='Sample Index', y='Predicted Prob (Abnormal)',
            hue='True Label', dodge=False,
            palette={'Normal': 'royalblue', 'Abnormal': 'crimson'})
plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold')
for i, row in df_plot.iterrows():
    if not row['Prediction Correct']:
        plt.text(i, row['Predicted Prob (Abnormal)'] + 0.02, 'X',
                 ha='center', color='red', fontweight='bold', fontsize=14)
plt.ylim(0, 1.1)
plt.title(f'Confidence (First {num_points} Test Samples)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, 'confidence_plot_optimal_ensemble.png'))
plt.close()

print("Optimal ensemble evaluation completed.")
