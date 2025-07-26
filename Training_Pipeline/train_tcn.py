import os
import glob
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import (
    roc_curve, 
    precision_recall_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score, 
    average_precision_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Model_Definitions.TCN import TCN   # Your existing TCN model


# ----------------------------- RESULT CLASS -----------------------------
class SepsisTransformerResult:
    def __init__(self, best_thresh, f1, precision, recall, confusion_matrix, auroc, auprc):
        self.best_threshold = best_thresh
        self.f1_score = f1
        self.precision = precision
        self.recall = recall
        self.tn, self.fp, self.fn, self.tp = confusion_matrix
        self.auroc = auroc
        self.auprc = auprc

    def __str__(self):
        lines = [
            "╔" + "═" * 46 + "╗",
            "║              Sepsis TCN Results               ║",
            "╠" + "═" * 46 + "╣",
            f"  Best Threshold (max F1) : {self.best_threshold}",
            f"  F1 Score                : {self.f1_score}",
            f"  Precision (at max F1)   : {self.precision}",
            f"  Recall (at max F1)      : {self.recall}",
            f"  TN (at max F1)          : {self.tn}",
            f"  FP (at max F1)          : {self.fp}",
            f"  FN (at max F1)          : {self.fn}",
            f"  TP (at max F1)          : {self.tp}",
            f"  AUROC                   : {self.auroc}",
            f"  AUPRC                   : {self.auprc}",
            "╚" + "═" * 46 + "╝",
        ]
        label_lines = [line for line in lines[3:-1]]
        max_label_width = max(len(line.split(":")[0]) for line in label_lines)
        formatted_lines = lines[:3]
        for line in label_lines:
            label, value = line.split(":")
            formatted_lines.append(f"{label.ljust(max_label_width)} :   {value.strip()}")
        formatted_lines.append(lines[-1])
        return "\n".join(formatted_lines)


# ----------------------------- DATASET WRAPPER -----------------------------
class SepsisTransformerDataset(Dataset):
    def __init__(self, npz_dir, labels_csv):
        self.files = sorted(glob.glob(os.path.join(npz_dir, 'patient_*.npz')))
        labels_df = pd.read_csv(labels_csv)
        self.labels = dict(zip(labels_df['patient_id'], labels_df['SepsisLabel']))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        X = torch.from_numpy(data['X']).float()
        pid = int(os.path.basename(self.files[idx]).split('_')[1].split('.')[0])
        y = torch.tensor(self.labels[pid], dtype=torch.float32)
        return X, y

from sklearn.model_selection import train_test_split

def split_train_val(dataset, val_ratio=0.2, seed=42):
    """Split dataset into train and validation sets"""
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=seed, shuffle=True)

    # Subset datasets
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    return train_subset, val_subset

# ----------------------------- PLOTTING FUNCTIONS -----------------------------
def plot_loss_curve(epochs, train_losses, val_losses, split_name, out_dir):
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - {split_name}')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'))
    plt.close()


def plot_roc_curve(fpr, tpr, auc_value, split_name, out_dir):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {split_name}')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
    plt.close()

def plot_prc_curve(precision, recall, auc_value, split_name, out_dir):
    plt.figure()
    plt.plot(recall, precision, label=f'PRC (AUC = {auc_value:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {split_name}')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'prc_curve.png'))
    plt.close()

def plot_confusion_matrix(cm, split_name, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    cm_array = np.array([[cm[0], cm[1]], [cm[2], cm[3]]])  # tn, fp, fn, tp
    ConfusionMatrixDisplay(cm_array, display_labels=[0, 1]).plot(ax=ax)
    fig.suptitle(f'Confusion Matrix -- Split {split_name}')
    plt.savefig(os.path.join(out_dir, 'cmatrix.png'))
    plt.close(fig)



# ----------------------------- TRAIN/EVAL LOOP -----------------------------
def train_eval_tcn(
    model,
    criterion,
    train_set: SepsisTransformerDataset,
    test_set: SepsisTransformerDataset,
    batch_size=32,
    num_epochs=10,
    learning_rate=1e-3,
    patience=5
):
    # Internal train/val split
    train_subset, val_subset = split_train_val(train_set, val_ratio=0.2)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    train_losses, val_losses = [], []
    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        # ---- Training ----
        model.train()
        total_train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out.squeeze(dim=-1), y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X.size(0)
        epoch_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ---- Validation ----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                total_val_loss += criterion(out.squeeze(dim=-1), y).item() * X.size(0)
        epoch_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch:2d} | train loss {epoch_train_loss:.4f} | val loss {epoch_val_loss:.4f}")

        # ---- Early stopping ----
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
                break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Evaluate on test set ----
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds = torch.sigmoid(out).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.cpu().numpy())

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)

    # ROC & PRC
    fpr, tpr, _ = roc_curve(labels, preds)
    prec, rec, pr_thresholds = precision_recall_curve(labels, preds)
    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)

    # Best threshold via max F1
    f1s = []
    for thresh in pr_thresholds:
        y_pred = (preds >= thresh).astype(int)
        f1s.append(f1_score(labels, y_pred))
    best_idx = np.argmax(f1s)
    best_threshold = pr_thresholds[best_idx]
    best_f1 = f1s[best_idx]
    precision = prec[best_idx]
    recall = rec[best_idx]
    y_pred_best = (preds >= best_threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, y_pred_best).ravel()

    return (
        range(1, len(train_losses) + 1),
        train_losses,
        val_losses,
        (fpr, tpr),
        (prec, rec),
        (best_threshold, best_f1, precision, recall),
        (tn, fp, fn, tp),
        auroc,
        auprc
    )

# ----------------------------- MAIN -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Parent directory containing splits')
    parser.add_argument('output_dir', type=str, help='Where to store results')
    parser.add_argument('split_name', type=str, help='Which split to use (folder name)')
    parser.add_argument('raw_split_csv', type=str, help='Path to labels CSV')
    args = parser.parse_args()

    split_dir = os.path.join(args.data_dir, args.split_name)
    with open(os.path.join(split_dir, 'data.json'), 'r') as f:
        metadata = json.load(f)
    num_features = metadata['num_features']

    train_dir = os.path.join(split_dir, 'train')
    test_dir = os.path.join(split_dir, 'test')
    train_ds = SepsisTransformerDataset(train_dir, args.raw_split_csv)
    test_ds = SepsisTransformerDataset(test_dir, args.raw_split_csv)

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, args.split_name)
    os.makedirs(fig_dir, exist_ok=True)

    # Initialize TCN
    model = TCN(
        input_size=num_features,
        num_channels=[16,32,32]
    )
    criterion = nn.BCEWithLogitsLoss()

    # Train & evaluate
    epochs, train_loss, val_loss, roc_grid, prc_grid, best_thresh_scores, confusion_matrix_vals, auroc, auprc = train_eval_tcn(
        model,
        criterion,
        train_ds,
        test_ds,
        batch_size=16,
        num_epochs=30,
        learning_rate=5e-4,
        patience=5
    )



    threshold, f1, precision, recall = best_thresh_scores
    res = SepsisTransformerResult(
        best_thresh=threshold,
        f1=f1,
        precision=precision,
        recall=recall,
        confusion_matrix=confusion_matrix_vals,
        auroc=auroc,
        auprc=auprc
    )

    print(f'\n{res}\n')
    fpr, tpr = roc_grid
    prec, rec = prc_grid

    # Plots
    plot_loss_curve(epochs, train_loss, val_loss, split_name=args.split_name, out_dir=fig_dir)
    plot_roc_curve(fpr, tpr, auroc, split_name=args.split_name, out_dir=fig_dir)
    plot_prc_curve(prec, rec, auprc, split_name=args.split_name, out_dir=fig_dir)

    plot_confusion_matrix(cm=confusion_matrix_vals, split_name=args.split_name, out_dir=fig_dir)


if __name__ == '__main__':
    main()
