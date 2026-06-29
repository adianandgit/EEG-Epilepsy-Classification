# project_helpers_plotting.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score,
                             accuracy_score)
import itertools
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 12,
    "figure.figsize": (8, 6)
})

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_fig(path, name):
    p = os.path.join(path, name)
    plt.tight_layout()
    plt.savefig(p + ".png", dpi=300, bbox_inches="tight")
    try:
        plt.savefig(p + ".pdf", bbox_inches="tight")
    except Exception:
        pass
    plt.close()

def plot_loss_acc(history, out_dir):
    """
    history: dict with keys 'train_loss','val_loss','train_acc','val_acc' each list per epoch
    """
    out_dir = _ensure_dir(out_dir)
    epochs = np.arange(1, len(history['train_loss']) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per epoch")
    plt.legend()
    save_fig(out_dir, "loss_curve")

    # Accuracy
    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
    plt.plot(epochs, history['val_acc'], label='Val Acc', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per epoch")
    plt.legend()
    save_fig(out_dir, "accuracy_curve")

def plot_confusion_matrix(y_true, y_pred, classes, out_dir, normalize=False):
    """
    Annotated confusion matrix heatmap
    """
    out_dir = _ensure_dir(out_dir)
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title("Confusion matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
        plt.text(j, i, val, horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=11)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    save_fig(out_dir, "confusion_matrix" + ("_norm" if normalize else ""))

def compute_classification_metrics(y_true, y_pred, y_proba=None, classes=None, out_dir=None):
    """
    Compute metrics and save:
      - Bar chart for precision/recall/f1 per class and macro numbers
      - ROC curve (binary or multiclass OvR)
      - Precision-Recall curve
    Returns a dict of metric numbers.
    """
    if out_dir is not None:
        out_dir = _ensure_dir(out_dir)

    metrics = {}
    # basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # per-class
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    classes_list = classes if classes is not None else [str(i) for i in range(len(precisions))]

    # bar chart
    ind = np.arange(len(classes_list))
    width = 0.25
    plt.figure(figsize=(10, 5))
    plt.bar(ind - width, precisions, width=width, label='Precision')
    plt.bar(ind, recalls, width=width, label='Recall')
    plt.bar(ind + width, f1s, width=width, label='F1')
    plt.xticks(ind, classes_list, rotation=45, ha='right')
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title("Precision / Recall / F1 per class")
    save_fig(out_dir, "prf_per_class")

    # ROC and AUROC
    if y_proba is not None:
        y_proba_arr = np.array(y_proba)
        n_classes = y_proba_arr.shape[1]
        # binarize for multiclass
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        # ROC for each class
        plt.figure(figsize=(8, 6))
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba_arr[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle='--', lw=1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC curve")
            plt.legend(loc="lower right")
            save_fig(out_dir, "roc_curve")
            metrics['auroc'] = roc_auc
        else:
            # multiclass OvR
            all_fpr = dict()
            all_tpr = dict()
            aucs = []
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba_arr[:, i])
                all_fpr[i] = fpr; all_tpr[i] = tpr
                aucs.append(auc(fpr, tpr))
                plt.plot(fpr, tpr, lw=1.5, label=f"Class {i} (AUC={aucs[-1]:.3f})")
            plt.plot([0, 1], [0, 1], linestyle='--', lw=1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC curve (one-vs-rest)")
            plt.legend(loc="lower right", fontsize=9)
            save_fig(out_dir, "roc_curve_ovr")
            metrics['auroc_per_class'] = aucs
            metrics['auroc_mean'] = np.mean(aucs)

        # Precision-Recall curve for the positive class or each class
        plt.figure(figsize=(8, 6))
        if n_classes == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_proba_arr[:, 1])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f"PR AUC = {pr_auc:.3f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall curve")
            plt.legend()
            save_fig(out_dir, "pr_curve")
            metrics['pr_auc'] = pr_auc
        else:
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba_arr[:, i])
                plt.plot(recall, precision, lw=1.2, label=f"Class {i}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall curve (per class)")
            plt.legend(loc='lower left', fontsize=9)
            save_fig(out_dir, "pr_curve_ovr")

    # confusion matrix
    if out_dir is not None:
        plot_confusion_matrix(y_true, y_pred, classes_list, out_dir, normalize=False)
        plot_confusion_matrix(y_true, y_pred, classes_list, out_dir, normalize=True)

    # return numeric metrics
    metrics['per_class_precision'] = precisions.tolist()
    metrics['per_class_recall'] = recalls.tolist()
    metrics['per_class_f1'] = f1s.tolist()

    return metrics

def plot_tsne(embeddings, labels, out_dir, perplexity=30, n_iter=1000):
    """
    embeddings: (N, D) numpy array
    labels: (N,) array-like
    """
    out_dir = _ensure_dir(out_dir)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    for lab in np.unique(labels):
        mask = labels == lab
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=str(lab), alpha=0.6, s=15)
    plt.legend()
    plt.title("t-SNE of embeddings")
    save_fig(out_dir, "tsne_embeddings")
    return X_tsne
