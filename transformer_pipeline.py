import os
import glob
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import(
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
from Model_Definitions import Sepsis_Predictor_Encoder, Sepsis_Predictor_Encoder_Hyperparameters
from Training_Pipeline import Train_Hyperparameters

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
            "║          Sepsis Transformer Results          ║",
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

        # Determine the max width of labels before the colon for alignment
        # Skip the note line as it doesn't have a colon for splitting
        label_lines = [line for line in lines[3:-1]]
        max_label_width = max(len(line.split(":")[0]) for line in label_lines)

        formatted_lines = lines[:3]  # include header and note

        for line in label_lines:
            label, value = line.split(":")
            formatted_lines.append(f"{label.ljust(max_label_width)} :   {value.strip()}")

        formatted_lines.append(lines[-1])  # bottom bar

        return "\n".join(formatted_lines)


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

def plot_loss_curve(epochs, train_losses, eval_losses, split_name, out_dir):
    plt.figure(); plt.plot(epochs, train_losses, label='Train'); plt.plot(epochs, eval_losses, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title(f'Loss ({split_name})')
    plt.savefig(os.path.join(out_dir, 'loss_curve.png')); plt.close()


def plot_roc_and_prc_curves(roc_curve, prc_curve, split_name, out_dir):
    fpr, tpr = roc_curve
    prec, rec = prc_curve
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    prc_display = PrecisionRecallDisplay(precision=prec, recall=rec)
    roc_display.plot(ax=ax1)
    prc_display.pllt(ax=ax2)
    
    fig.suptitle(f'ROC, Precision/Recall -- Split {split_name}')
    
    fig.savefig(os.path.join(out_dir, 'prc_roc_curve.png')); plt.close(fig)

def plot_confusion_matrix(cm, split_name, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    cm_display.plot(ax)
    fig.suptitle(f'Confusion Matrix -- Split {split_name}')
    plt.savefig(os.path.join(out_dir, 'cmatrix.png'))
    plt.close(fig)


def train_eval_transformer(
    model,
    criterion,
    train_set: SepsisTransformerDataset, 
    eval_set: SepsisTransformerDataset, 
    train_params: Train_Hyperparameters,
    ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), train_params.learning_rate)
    
    train_loader = DataLoader(train_set, batch_size=train_params.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=train_params.batch_size, shuffle=False)
    
    train_losses, eval_losses = [], []
    for epoch in range(1, train_params.num_epochs + 1):
        # train
        model.train()
        total_tr = 0.0
        for X, y in train_loader:
            X, y = [t.to(device) for t in (X, y)]
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out.squeeze(dim=-1), y)
            loss.backward()
            optimizer.step()
            total_tr += loss.item() * X.size(0)
        train_losses.append(total_tr / len(train_loader.dataset))
        
        # validate
        model.eval()
        total_eval, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for X, y in eval_loader:
                X, y = [t.to(device) for t in (X, y)]
                out = model(X)
                total_eval += criterion(out.squeeze(dim=-1), y).item() * X.size(0)
                preds = torch.sigmoid(out).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(y.cpu().numpy())
        eval_losses.append(total_eval / len(eval_loader.dataset))
        labels = np.concatenate(all_labels)
        preds = np.concatenate(all_preds)
        auroc = roc_auc_score(labels, preds)
        print(f"Epoch {epoch:2d} | train loss {train_losses[-1]:.4f}"
                f" | val loss {eval_losses[-1]:.4f} | val AUROC {auroc:.4f}")
        
    # save plots
    epochs_range = [e for e in range(1, train_params.num_epochs + 1)]
    fpr, tpr, _ = roc_curve(labels, preds)
    prec, rec, pr_thresholds = precision_recall_curve(labels, preds)
    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)
    
    # f1 score for each threshold
    f1s = []
    for thresh in pr_thresholds:
        y_pred = (preds >= thresh).astype(int)
        f1s.append(f1_score(labels, y_pred))
    f1s = np.array(f1s)
    
    # best threshold (max f1)
    best_idx = f1s.argmax()
    best_threshold = pr_thresholds[best_idx]
    best_f1 = f1s[best_idx]
    precision = prec[best_idx]
    recall = rec[best_idx]
    y_pred_best = (preds >= best_threshold).astype(int)
    
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, y_pred_best).ravel()
    
    
    
    return (
        (epochs_range, train_losses, eval_losses), 
        (fpr, tpr), 
        (prec, rec),
        (best_threshold, best_f1, precision, recall),
        (tn, fp, fn, tp),
        auroc,
        auprc
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Parent directory of transformer inputs, containing splits')
    parser.add_argument('output_dir', type=str, help='Where to store results')
    parser.add_argument('split_name', type=str, help='The split to train and evaluate on')
    parser.add_argument('raw_split_csv', type=str, help='Path to raw csv to compute labels')
    args = parser.parse_args()
    
    split_dir = os.path.join(args.data_dir, args.split_name)
    fname = 'data.json'
    
    metadata_file = os.path.join(split_dir, fname)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    num_features = metadata['num_features']
    
    train_dir = os.path.join(split_dir, 'train')
    test_dir = os.path.join(split_dir, 'test')
    train_ds = SepsisTransformerDataset(train_dir, args.raw_split_csv)
    test_ds = SepsisTransformerDataset(test_dir, args.raw_split_csv)
    
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    fig_dir = os.path.join(args.output_dir, args.split_name)
    os.makedirs(fig_dir, exist_ok=True)
    
    hyperparams = Sepsis_Predictor_Encoder_Hyperparameters(
        embedding_dim=64,
        feedforward_hidden_dim=128,
        n_heads=4,
        activation='relu',
        n_layers=6,
        dropout_p=0,
        pos_encoding_dropout_p=0
    )
    
    train_params = Train_Hyperparameters(
        batch_size=16,
        num_epochs=10,
        learning_rate=1e-4
    )
    
    model = Sepsis_Predictor_Encoder(
        input_size=num_features,
        output_size=1,
        hyperparameters=hyperparams
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    print(f'\n\n')
    print(f'------ Starting Training: Split {args.split_name} ------\n')
    
    print(f'{hyperparams}\n{train_params}\n')
    
    loss_grid, roc_grid, prc_grid, best_thresh_scores, confusion_matrix, auroc, auprc = train_eval_transformer(model, criterion, train_ds, test_ds, train_params)
    epochs, train_loss, eval_loss = loss_grid
    threshold, f1, precision, recall = best_thresh_scores
    
    res = SepsisTransformerResult(
        best_thresh=threshold,
        f1=f1,
        precision=precision,
        recall=recall,
        confusion_matrix=confusion_matrix,
        auroc=auroc,
        auprc=auprc
    )
    
    print(f'\n{res}\n')
    
    plot_loss_curve(epochs, train_loss, eval_loss, split_name=args.split_name, out_dir=fig_dir)
    plot_roc_and_prc_curves(roc_curve=roc_grid, prc_curve=prc_grid, split_name=args.split_name, out_dir=fig_dir)
    plot_confusion_matrix(cm=confusion_matrix, split_name=args.split_name, out_dir=fig_dir)

if __name__ == '__main__':
    main()