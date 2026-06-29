import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.combine import SMOTEENN

sys.path.append('./UGP')
from new_model import NewTransferModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_PREVIOUS_MODEL = "best_trusted_model.pt"
DATA_PATH = "./dataset_storage/MyNewEEG_Trusted"
PLOT_PATH = "./final_metric"
os.makedirs(PLOT_PATH, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4

print(f"Device: {DEVICE}")

class EEGDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.samples = data['samples'].float()
        self.labels = data['labels'].long()
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.samples[idx], self.labels[idx]

print("Loading pre-trained feature extractor...")
model_full = NewTransferModel(num_new_classes=2).to(DEVICE)
if not os.path.exists(BEST_PREVIOUS_MODEL):
    print(f"Model file '{BEST_PREVIOUS_MODEL}' not found.")
    sys.exit(1)
model_full.load_state_dict(torch.load(BEST_PREVIOUS_MODEL, map_location=DEVICE))
model_full.eval()
for param in model_full.parameters():
    param.requires_grad = False
print("Feature extractor loaded.")

def extract_features(model, dataloader):
    features_list, labels_list = [], []
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(DEVICE)
            feats_flat = model.extract_features(data)
            features_list.append(feats_flat.cpu())
            labels_list.append(labels)
    return torch.cat(features_list).numpy(), torch.cat(labels_list).numpy()

print("Extracting features from data...")
train_ds_raw = EEGDataset(os.path.join(DATA_PATH, "train.pt"))
valid_ds_raw = EEGDataset(os.path.join(DATA_PATH, "val.pt"))
test_ds_raw = EEGDataset(os.path.join(DATA_PATH, "test.pt"))

X_train_feats, y_train = extract_features(model_full, DataLoader(train_ds_raw, batch_size=32))
X_val_feats, y_val = extract_features(model_full, DataLoader(valid_ds_raw, batch_size=32))
X_test_feats, y_test = extract_features(model_full, DataLoader(test_ds_raw, batch_size=32))

print("Applying SMOTE+ENN...")
sme = SMOTEENN(random_state=42)
X_train_res, y_train_res = sme.fit_resample(X_train_feats, y_train)
print(f"Resampled class counts: {np.bincount(y_train_res)}")

train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_train_res), torch.LongTensor(y_train_res)), batch_size=BATCH_SIZE, shuffle=True)
valid_dl = DataLoader(TensorDataset(torch.FloatTensor(X_val_feats), torch.LongTensor(y_val)), batch_size=BATCH_SIZE)
test_dl = DataLoader(TensorDataset(torch.FloatTensor(X_test_feats), torch.LongTensor(y_test)), batch_size=BATCH_SIZE)

classifier_head = model_full.feedforward_net
for param in classifier_head.parameters():
    param.requires_grad = True
for layer in classifier_head.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
classifier_head = classifier_head.to(DEVICE)

optimizer = optim.Adam(classifier_head.parameters(), lr=LR, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()

best_val_f1 = -1.0
history = {'train_loss': [], 'val_f1': []}
print("Training classifier head...")

for epoch in range(EPOCHS):
    classifier_head.train()
    total_loss = 0
    for feats, labels in train_dl:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(classifier_head(feats), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dl)
    history['train_loss'].append(avg_loss)

    classifier_head.eval()
    all_preds, all_labels_val = [], []
    with torch.no_grad():
        for feats, labels in valid_dl:
            feats = feats.to(DEVICE)
            all_preds.extend(torch.argmax(classifier_head(feats), dim=1).cpu().numpy())
            all_labels_val.extend(labels.numpy())

    val_f1 = classification_report(all_labels_val, all_preds, output_dict=True, zero_division=0)['1']['f1-score']
    history['val_f1'].append(val_f1)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(classifier_head.state_dict(), os.path.join(PLOT_PATH, "best_head_smote.pt"))

print(f"Final Evaluation | Best Val F1: {best_val_f1:.4f}")
classifier_head.load_state_dict(torch.load(os.path.join(PLOT_PATH, "best_head_smote.pt")))
classifier_head.eval()

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for feats, labels in test_dl:
        feats = feats.to(DEVICE)
        outputs = classifier_head(feats)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds, target_names=['Normal', 'Abnormal'], zero_division=0))
try:
    auc = roc_auc_score(all_labels, all_probs)
    print(f"Test AUROC: {auc:.4f}")
except:
    auc = 0.5
    print("AUROC computation failed.")

print(f"Generating plots in {PLOT_PATH}...")

plt.figure()
plt.plot(history['train_loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(PLOT_PATH, 'train_loss.png'))
plt.close()

plt.figure()
plt.plot(history['val_f1'])
plt.title('Validation F1 (Abnormal)')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.savefig(os.path.join(PLOT_PATH, 'val_f1.png'))
plt.close()

fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUROC={auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(os.path.join(PLOT_PATH, 'roc_curve.png'))
plt.close()

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal (Pred)', 'Abnormal (Pred)'],
            yticklabels=['Normal (True)', 'Abnormal (True)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, 'confusion_matrix_smote.png'))
plt.close()

num_points = 40
indices = np.arange(min(len(all_labels), num_points))
df_plot = pd.DataFrame({
    'Sample Index': indices,
    'True Label': ['Abnormal' if all_labels[i] == 1 else 'Normal' for i in indices],
    'Predicted Prob (Abnormal)': np.array(all_probs)[indices],
    'Prediction Correct': np.array(all_labels)[indices] == np.array(all_preds)[indices]
})

plt.figure(figsize=(14, 7))
sns.barplot(data=df_plot, x='Sample Index', y='Predicted Prob (Abnormal)',
            hue='True Label', dodge=False,
            palette={'Normal': 'royalblue', 'Abnormal': 'crimson'})
plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold')
for i, row in df_plot.iterrows():
    if not row['Prediction Correct']:
        plt.text(i, row['Predicted Prob (Abnormal)'] + 0.02, 'X', ha='center', color='red', fontweight='bold', fontsize=14)

plt.ylim(0, 1.1)
plt.title(f'Model Confidence (First {num_points} Test Samples)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, 'confidence_plot.png'))
plt.close()

print("Process completed successfully.")
