import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

sys.path.append('./UGP')
from new_model import NewTransferModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_PATH = "./UGP/experiments_logs/Exp1/run1/supervised_seed_0/saved_models/ckp_last.pt"
DATA_PATH = "./dataset_storage/MyNewEEG_Trusted"
PLOT_PATH = "./plots_trusted"
os.makedirs(PLOT_PATH, exist_ok=True)

NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
GRADIENT_CLIP_NORM = 1.0

BEST_VAL_F1 = -1.0
BEST_MODEL_PATH = "best_trusted_model.pt"
FINAL_MODEL_PATH = "final_trusted_model.pt"

history = {'train_loss': [], 'val_acc': [], 'val_f1_class1': [], 'val_auc': []}
print(f"Device: {DEVICE}")

class PartialUndersamplingDataset(Dataset):
    def __init__(self, pt_file_path, augment=False, ratio=5):
        data = torch.load(pt_file_path)
        samples, labels = data['samples'].numpy(), data['labels'].numpy()
        normal_idx = np.where(labels == 0)[0]
        abnormal_idx = np.where(labels == 1)[0]
        num_abnormal = len(abnormal_idx)
        num_normal = min(len(normal_idx), num_abnormal * ratio)
        np.random.shuffle(normal_idx)
        selected_normals = normal_idx[:num_normal]
        combined_idx = np.concatenate([selected_normals, abnormal_idx])
        self.samples = samples[combined_idx]
        self.labels = labels[combined_idx]
        print(f"Partial undersampling: {num_normal} normal, {num_abnormal} abnormal samples.")
        self.augment = augment
        self.jitter_sigma = 0.03
        self.scale_factor = 0.15

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        x, y = self.samples[idx], self.labels[idx]
        if self.augment:
            if np.random.rand() < 0.5:
                x += np.random.normal(0., self.jitter_sigma, size=x.shape)
            if np.random.rand() < 0.5:
                x *= np.random.normal(1.0, self.scale_factor, size=(x.shape[0], 1))
        return torch.from_numpy(x.copy()).float(), torch.tensor(y).long()

class EEGDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.samples = data['samples'].numpy()
        self.labels = data['labels'].numpy()

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        x, y = self.samples[idx], self.labels[idx]
        return torch.from_numpy(x.copy()).float(), torch.tensor(y).long()

model = NewTransferModel(num_new_classes=NUM_CLASSES).to(DEVICE)
print(f"Loading pretrained weights from {PRETRAINED_PATH}")
if not os.path.exists(PRETRAINED_PATH):
    print("Pretrained model not found.")
    sys.exit(1)
model.backbone.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE)["model_state_dict"], strict=False)
model.freeze_backbone()
print("Pretrained weights loaded.")

print("Preparing datasets...")
train_ds = PartialUndersamplingDataset(os.path.join(DATA_PATH, "train.pt"), augment=True, ratio=5)
val_ds = EEGDataset(os.path.join(DATA_PATH, "val.pt"))
test_ds = EEGDataset(os.path.join(DATA_PATH, "test.pt"))

y_train = train_ds.labels
class_counts = np.bincount(y_train)
weights = 1. / class_counts
sample_weights = torch.tensor([weights[y] for y in y_train], dtype=torch.double)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
print(f"Training class counts: {class_counts}")

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

criterion = nn.CrossEntropyLoss()
params_to_train = list(model.stem.parameters()) + list(model.feedforward_net.parameters())
optimizer = optim.Adam(params_to_train, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=10)

def train_epoch(model, dl, optimizer):
    model.train()
    total_loss = 0
    for data, labels in dl:
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(data), labels)
        loss.backward()
        clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRADIENT_CLIP_NORM)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dl)

def evaluate(model, dl, print_report=False):
    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for data, labels in dl:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            outputs = model(data)
            probs = F.softmax(outputs, dim=1)
            scores = probs[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0, target_names=["Normal", "Abnormal"])
    if print_report:
        print(classification_report(all_labels, all_preds, zero_division=0, target_names=["Normal", "Abnormal"]))
    f1 = report.get("Abnormal", {}).get("f1-score", 0.0)
    acc = report.get("accuracy", 0.0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = 0.5
    return acc * 100, f1, cm, auc, all_labels, all_preds, all_scores

print("Training started.")
for epoch in range(EPOCHS):
    loss = train_epoch(model, train_dl, optimizer)
    acc, f1, _, auc, _, _, _ = evaluate(model, val_dl)
    scheduler.step(f1)
    history['train_loss'].append(loss)
    history['val_acc'].append(acc)
    history['val_f1_class1'].append(f1)
    history['val_auc'].append(auc)
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={loss:.4f}, Val Acc={acc:.2f}%, F1={f1:.4f}, AUROC={auc:.4f}, LR={optimizer.param_groups[0]['lr']:.2e}")
    if f1 > BEST_VAL_F1:
        BEST_VAL_F1 = f1
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"New best model saved (F1={f1:.4f})")

print("Training completed.")
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print("Final model saved.")

print(f"Evaluating best model (Val F1={BEST_VAL_F1:.4f})")
model.load_state_dict(torch.load(BEST_MODEL_PATH))
best_acc, best_f1, best_cm, best_auc, best_labels, best_preds, best_scores = evaluate(model, test_dl, print_report=True)
print(f"Test Results: Accuracy={best_acc:.2f}%, F1={best_f1:.4f}, AUROC={best_auc:.4f}")

print(f"Saving plots to {PLOT_PATH}...")
plt.figure(); plt.plot(history['train_loss']); plt.title('Training Loss'); plt.savefig(os.path.join(PLOT_PATH, 'train_loss.png')); plt.close()
plt.figure(); plt.plot(history['val_f1_class1']); plt.title('Validation F1'); plt.savefig(os.path.join(PLOT_PATH, 'val_f1.png')); plt.close()
plt.figure(); plt.plot(history['val_auc']); plt.title('Validation AUROC'); plt.savefig(os.path.join(PLOT_PATH, 'val_auc.png')); plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "Abnormal"],
            yticklabels=["Normal", "Abnormal"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, 'confusion_matrix.png'))
plt.close()

fpr, tpr, _ = roc_curve(best_labels, best_scores)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUROC={best_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(os.path.join(PLOT_PATH, 'roc_curve.png'))
plt.close()

print("Generating confidence plot...")
num_points = 40
indices = np.arange(min(len(best_labels), num_points))
df_plot = pd.DataFrame({
    'Sample Index': indices,
    'True Label': ['Abnormal' if best_labels[i] == 1 else 'Normal' for i in indices],
    'Predicted Prob (Normal)': 1.0 - np.array(best_scores)[indices],
    'Prediction Correct': np.array(best_labels)[indices] == np.array(best_preds)[indices]
})
plt.figure(figsize=(14, 7))
sns.barplot(data=df_plot, x='Sample Index', y='Predicted Prob (Normal)',
            hue='True Label', dodge=False,
            palette={'Normal': 'royalblue', 'Abnormal': 'crimson'})
plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold')
for i, row in df_plot.iterrows():
    if not row['Prediction Correct']:
        plt.text(i, row['Predicted Prob (Normal)'] + 0.02, 'X',
                 ha='center', color='red', fontweight='bold', fontsize=14)
plt.ylim(0, 1.1)
plt.title(f'Model Confidence (First {num_points} Samples)')
plt.ylabel('Probability of Normal')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, 'confidence_plot_normal.png'))
plt.close()

print("Process completed successfully.")
