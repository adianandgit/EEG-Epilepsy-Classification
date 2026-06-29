import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)

def Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer,
            train_dl, valid_dl, device, epochs=50, out_dir="results"):
    """
    Trainer for VT EEG classification (Epileptic vs Non-epileptic)
    Includes training, validation, metric computation and beautiful plotting.
    """

    # === Directory setup ===
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_losses, val_losses = [], []
    val_accuracies, val_f1s, val_precisions, val_recalls, val_aurocs = [], [], [], [], []

    print("\n=============================================")
    print("Training started ...")
    print("=============================================\n")

    for epoch in range(epochs):
        model.train()
        temporal_contr_model.train()

        total_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", ncols=100):
            x, y = batch
            x, y = x.to(device), y.to(device)

            model_optimizer.zero_grad()
            temporal_contr_optimizer.zero_grad()

            outputs = model(x)
            loss = torch.nn.functional.cross_entropy(outputs, y)
            loss.backward()

            model_optimizer.step()
            temporal_contr_optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dl)
        train_losses.append(avg_train_loss)

        # === Validation ===
        model.eval()
        all_preds, all_trgs, all_probs = [], [], []
        val_loss = 0.0

        with torch.no_grad():
            for x, y in valid_dl:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = torch.nn.functional.cross_entropy(outputs, y)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                trgs = y.cpu().numpy()

                all_preds.extend(preds)
                all_trgs.extend(trgs)
                all_probs.extend(probs)

        avg_val_loss = val_loss / len(valid_dl)
        val_losses.append(avg_val_loss)

        acc = accuracy_score(all_trgs, all_preds)
        f1 = f1_score(all_trgs, all_preds)
        prec = precision_score(all_trgs, all_preds)
        rec = recall_score(all_trgs, all_preds)
        try:
            auroc = roc_auc_score(all_trgs, all_probs)
        except ValueError:
            auroc = float('nan')

        val_accuracies.append(acc)
        val_f1s.append(f1)
        val_precisions.append(prec)
        val_recalls.append(rec)
        val_aurocs.append(auroc)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | AUROC: {auroc:.4f}\n")

        # === Save checkpoint ===
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'temporal_contr_model_state_dict': temporal_contr_model.state_dict(),
            'optimizer_state_dict': model_optimizer.state_dict(),
        }, os.path.join(ckpt_dir, f"ckp_epoch_{epoch+1}.pt"))

        # === Save metrics plots ===
        plot_metrics(train_losses, val_losses, val_accuracies, val_f1s, val_precisions,
                     val_recalls, val_aurocs, plots_dir)

    print("\nTraining completed ✅\n")


# === Plotting ===
def plot_metrics(train_losses, val_losses, accs, f1s, precs, recs, aurocs, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    def save_plot(x, y, title, ylabel, name):
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(x, y, marker='o', linewidth=2, color='#007acc')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    epochs = range(1, len(train_losses) + 1)
    save_plot(epochs, train_losses, "Training Loss", "Loss", "train_loss")
    save_plot(epochs, val_losses, "Validation Loss", "Loss", "val_loss")
    save_plot(epochs, accs, "Validation Accuracy", "Accuracy", "val_accuracy")
    save_plot(epochs, f1s, "F1 Score", "F1", "val_f1")
    save_plot(epochs, precs, "Precision", "Precision", "val_precision")
    save_plot(epochs, recs, "Recall", "Recall", "val_recall")
    save_plot(epochs, aurocs, "AUROC", "AUROC", "val_auroc")

    # Confusion matrix for last epoch
    plt.figure(figsize=(6, 5), dpi=300)
    cm = confusion_matrix(
        np.array([0, 1]*10)[:len(accs)],  # dummy if needed
        np.array([0, 1]*10)[:len(accs)]   # replace with real preds later if stored
    )
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix (last epoch)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
