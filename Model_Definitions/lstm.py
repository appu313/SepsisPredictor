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
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix
)
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace

# -----------------------------
# LSTM Model Definition
# -----------------------------
class SepsisLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(SepsisLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


# -----------------------------
# Dataset Class
# -----------------------------
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

# -----------------------------
# Metrics Result Class
# -----------------------------
class SepsisTransformerResult:
    def __init__(self, best_thresh, f1, precision, recall, confusion_matrix, auroc, auprc, sensitivity, specificity):
        self.best_threshold = best_thresh
        self.f1_score = f1
        self.precision = precision
        self.recall = recall
        self.tn, self.fp, self.fn, self.tp = confusion_matrix
        self.auroc = auroc
        self.auprc = auprc
        self.sensitivity = sensitivity
        self.specificity = specificity

    def __str__(self):
        lines = [
            "╔" + "═" * 46 + "╗",
            "║          Sepsis LSTM Results               ║",
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
            f"  Sensitivity (Recall)    :  {self.sensitivity:.4f}",
            f"  Specificity             : {self.specificity:.4f}"
            "╚" + "═" * 46 + "╝",
        ]
        return "\n".join(lines)

# -----------------------------
# Training & Evaluation
# -----------------------------
def train_eval_lstm(model, criterion, train_set, eval_set, train_params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_params['learning_rate'])

    train_loader = DataLoader(train_set, batch_size=train_params['batch_size'], shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=train_params['batch_size'], shuffle=False)

    train_losses, eval_losses = [], []
    for epoch in range(1, train_params['num_epochs'] + 1):
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
        print(f"Epoch {epoch:2d} | train loss {train_losses[-1]:.4f} | val loss {eval_losses[-1]:.4f} | val AUROC {auroc:.4f}")

    fpr, tpr, _ = roc_curve(labels, preds)
    prec, rec, pr_thresholds = precision_recall_curve(labels, preds)
    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)

    f1s = [(f1_score(labels, (preds >= t).astype(int)), t) for t in pr_thresholds]
    f1s_only, thresholds_only = zip(*f1s)
    best_idx = np.argmax(f1s_only)
    best_f1 = f1s_only[best_idx]
    best_threshold = thresholds_only[best_idx]
    y_pred_best = (preds >= best_threshold).astype(int)
    best_precision = prec[best_idx]
    best_recall = rec[best_idx]
    tn, fp, fn, tp = confusion_matrix(labels, y_pred_best).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return (
        list(range(1, train_params['num_epochs'] + 1)),
        train_losses,
        eval_losses,
        fpr, tpr,
        prec, rec,
        best_threshold, best_f1, best_precision, best_recall,
        (tn, fp, fn, tp),
        auroc,
        auprc, 
        sensitivity,
        specificity
    )

# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('data_dir', type=str)
#     # parser.add_argument('output_dir', type=str)
#     # parser.add_argument('split_name', type=str)
#     # parser.add_argument('raw_split_csv', type=str)
#     # args = parser.parse_args()

#     args = Namespace(
#         data_dir= "/Users/appumwol/Downloads/preprocessed_transformer/AB", 
#         output_dir="/Users/appumwol/Documents/GaTech/Classes/DL/Project/Mine/output",
#         split_name="",
#         raw_split_csv="/Users/appumwol/Documents/GaTech/Classes/DL/Project/Mine/training_set_AB.csv"
#     )
    

#     split_dir = os.path.join(args.data_dir, args.split_name)
#     metadata_file = os.path.join(split_dir, 'data.json')
#     with open(metadata_file, 'r') as f:
#         metadata = json.load(f)
#     num_features = metadata['num_features']

#     train_ds = SepsisTransformerDataset(os.path.join(split_dir, 'train'), args.raw_split_csv)
#     test_ds = SepsisTransformerDataset(os.path.join(split_dir, 'test'), args.raw_split_csv)

#     os.makedirs(args.output_dir, exist_ok=True)
#     fig_dir = os.path.join(args.output_dir, args.split_name)
#     os.makedirs(fig_dir, exist_ok=True)

#     model =  SepsisLSTM(input_size=num_features)
#     train_params = {
#         'batch_size': 16,
#         'num_epochs': 30,
#         'learning_rate': 1e-4
#     }
#     criterion = nn.BCEWithLogitsLoss()

#     print(f"\n--- Training LSTM for split {args.split_name} ---\n")
#     epochs, train_loss, val_loss, fpr, tpr, prec, rec, threshold, f1, p, r, cm, auroc, auprc, sensitivity, specificity = train_eval_lstm(
#         model, criterion, train_ds, test_ds, train_params
#     )

#     result = SepsisTransformerResult(threshold, f1, p, r, cm, auroc, auprc, sensitivity, specificity)
#     print(result)

# if __name__ == '__main__':
#     main()
