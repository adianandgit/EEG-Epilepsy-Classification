import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from new_model import NewTransferModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./dataset_storage/MyNewEEG_Trusted"
PLOT_PATH = "./final_metric_ensemble"
os.makedirs(PLOT_PATH, exist_ok=True)

STABLE_MODEL_PATH = "best_trusted_model.pt"
UNSTABLE_MODEL_PATH = "best_f1_model.pt"
RECALL_HEAD_PATH = os.path.join("final_metric", "best_head_smote.pt")

# Ensemble weights
W_STABLE, W_UNSTABLE, W_RECALL = 2, 1, 2

print(f"Device: {DEVICE}")

class EEGDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.samples = data['samples'].float()
        self.labels = data['labels'].long()
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.samples[idx], self.labels[idx]

print("Loading ensemble models...")
model_files_exist = True

model_stable = NewTransferModel(num_new_classes=2).to(DEVICE)
if os.path.exists(STABLE_MODEL_PATH):
    model_stable.load_state_dict(torch.load(STABLE_MODEL_PATH, map_location=DEVICE))
    model_stable.eval()
    print(f"Loaded stable model: {STABLE_MODEL_PATH}")
else:
    print(f"Missing: {STABLE_MODEL_PATH}")
    model_files_exist = False

model_unstable = NewTransferModel(num_new_classes=2).to(DEVICE)
if os.path.exists(UNSTABLE_MODEL_PATH):
    model_unstable.load_state_dict(torch.load(UNSTABLE_MODEL_PATH, map_location=DEVICE))
    model_unstable.eval()
    print(f"Loaded high-F1 model: {UNSTABLE_MODEL_PATH}")
else:
    print(f"Missing: {UNSTABLE_MODEL_PATH}")
    model_files_exist = False

model_recall_head = NewTransferModel(num_new_classes=2).feedforward_net
if os.path.exists(RECALL_HEAD_PATH):
    model_recall_head.load_state_dict(torch.load(RECALL_HEAD_PATH, map_location=DEVICE))
    model_recall_head = model_recall_head.to(DEVICE)
    model_recall_head.eval()
    print(f"Loaded recall head: {RECALL_HEAD_PATH}")
else:
    print(f"Missing: {RECALL_HEAD_PATH}")
    model_files_exist = False

if not model_files_exist:
    print("Required model files missing. Exiting.")
    sys.exit(1)

feature_extractor = model_stable
test_ds = EEGDataset(os.path.join(DATA_PATH, "test.pt"))
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

print("Evaluating weighted ensemble...")
all_labels, all_preds, all_scores = [], [], []

with torch.no_grad():
    for data, labels in test_dl:
        data = data.to(DEVICE)
        out_stable = model_stable(data)
        out_unstable = model_unstable(data)
        features = feature_extractor.extract_features(data)
        out_recall = model_recall_head(features)

        prob_stable = F.softmax(out_stable, dim=1)[:, 1]
        prob_unstable = F.softmax(out_unstable, dim=1)[:, 1]
        prob_recall = F.softmax(out_recall, dim=1)[:, 1]

        total_weight = W_STABLE + W_UNSTABLE + W_RECALL
        final_prob = (W_STABLE * prob_stable + W_UNSTABLE * prob_unstable + W_RECALL * prob_recall) / total_weight
        final_pred = (final_prob > 0.5).long()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(final_pred.cpu().numpy())
        all_scores.extend(final_prob.cpu().numpy())

print("\nEnsemble Evaluation Report:")
report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0, target_names=["Normal", "Abnormal"])
print(classification_report(all_labels, all_preds, zero_division=0, target_names=["Normal", "Abnormal"]))

f1 = report.get("Abnormal", {}).get("f1-score", 0.0)
acc = report.get("accuracy", 0.0)
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
try:
    auc = roc_auc_score(all_labels, all_scores)
except ValueError:
    auc = 0.5

print(f"Results: Accuracy={acc*100:.2f}%, F1={f1:.4f}, AUROC={auc:.4f}")

print(f"Saving plots to {PLOT_PATH}...")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "Abnormal"],
            yticklabels=["Normal", "Abnormal"])
plt.title('Weighted Ensemble Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, 'confusion_matrix_weighted_ensemble.png'))
plt.close()

fpr, tpr, _ = roc_curve(all_labels, all_scores)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUROC={auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Weighted Ensemble)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, 'roc_curve_weighted_ensemble.png'))
plt.close()

num_points = 40
indices = np.arange(min(len(all_labels), num_points))
df_plot = pd.DataFrame({
    'Sample Index': indices,
    'True Label': ['Abnormal' if all_labels[i] == 1 else 'Normal' for i in indices],
    'Predicted Prob (Abnormal)': np.array(all_scores)[indices],
    'Prediction Correct': np.array(all_labels)[indices] == np.array(all_preds)[indices]
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
plt.savefig(os.path.join(PLOT_PATH, 'confidence_plot_weighted_ensemble.png'))
plt.close()

print("Ensemble evaluation completed.")
