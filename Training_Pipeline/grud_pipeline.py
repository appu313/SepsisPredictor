"""
grud_pipeline.py

Provides a training and evaluation pipeline for GRU-D across dataset splits.
Moves core logic out of notebooks into a reusable module.
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, roc_curve

from Model_Definitions.grud_model import GRUDCell
from Training_Pipeline.train_grud import SepsisGrudDataset, GRUDModel


def pad_collate(batch):
    Xs, Ms, Deltas, ys = zip(*batch)
    T_max = max(x.shape[0] for x in Xs)
    D = Xs[0].shape[1]
    B = len(batch)
    Xp = torch.zeros(B, T_max, D, dtype=torch.float32)
    Mp = torch.zeros(B, T_max, D, dtype=torch.float32)
    Dp = torch.zeros(B, T_max, D, dtype=torch.float32)
    y_ = torch.stack(ys)
    for i, (X, M, Delta, _) in enumerate(batch):
        L = X.shape[0]
        Xp[i, :L] = X
        Mp[i, :L] = M
        Dp[i, :L] = Delta
    return Xp, Mp, Dp, y_


def make_next_dir(base_dir, split):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir)
                if d.startswith(f"{split}-")]
    nums = [int(d.split('-')[-1]) for d in existing if '-' in d]
    idx = max(nums) + 1 if nums else 1
    out = os.path.join(base_dir, f"{split}-{idx}")
    os.makedirs(out, exist_ok=True)
    return out


def run_experiments(splits, results_base='results/GRU_D', epochs=20,
                    batch_size=32, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for split in splits:
        print(f"\n=== Split {split} ===")
        out_dir = make_next_dir(results_base, split)

        # load raw CSV to get features and mean
        csv_path = f'Data/imputed/data/data/raw/training_set_{split}.csv'
        df_raw = pd.read_csv(csv_path)
        exclude = ['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS','SepsisLabel','patient_id']
        features = [c for c in df_raw.columns if c not in exclude]
        x_mean = df_raw[features].mean().fillna(0).values

        # dataset
        npz_dir = os.path.join('Data', 'grud_inputs', split)
        full_ds = SepsisGrudDataset(npz_dir, csv_path)
        n_train = int(0.8 * len(full_ds))
        n_val   = len(full_ds) - n_train
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, collate_fn=pad_collate,
                                  pin_memory=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                shuffle=False, collate_fn=pad_collate,
                                pin_memory=True, num_workers=0)

        # model setup
        model = GRUDModel(len(features), hidden_size=128,
                          x_mean=x_mean).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        train_losses, val_losses = [], []
        for epoch in range(1, epochs+1):
            # train
            model.train()
            total_tr = 0.0
            for X, M, Delta, y in train_loader:
                X, M, Delta, y = [t.to(device) for t in (X, M, Delta, y)]
                optimizer.zero_grad()
                out = model(X, M, Delta)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_tr += loss.item() * X.size(0)
            train_losses.append(total_tr / len(train_loader.dataset))

            # validate
            model.eval()
            total_val, all_preds, all_labels = 0.0, [], []
            with torch.no_grad():
                for X, M, Delta, y in val_loader:
                    X, M, Delta, y = [t.to(device) for t in (X, M, Delta, y)]
                    out = model(X, M, Delta)
                    total_val += criterion(out, y).item() * X.size(0)
                    preds = torch.sigmoid(out).cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(y.cpu().numpy())
            val_losses.append(total_val / len(val_loader.dataset))

            labels = np.concatenate(all_labels)
            preds  = np.concatenate(all_preds)
            auc    = roc_auc_score(labels, preds)
            print(f"Epoch {epoch:2d} | train loss {train_losses[-1]:.4f}"
                  f" | val loss {val_losses[-1]:.4f} | val AUROC {auc:.4f}")

        # save plots
        epochs_range = range(1, epochs+1)
        plt.figure(); plt.plot(epochs_range, train_losses, label='Train'); plt.plot(epochs_range, val_losses, label='Val')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title(f'Loss ({split})')
        plt.savefig(os.path.join(out_dir, 'loss_curve.png')); plt.close()

        fpr, tpr, _ = roc_curve(labels, preds)
        plt.figure(); plt.plot(fpr, tpr, label=f'AUC={auc:.3f}'); plt.plot([0,1],[0,1],'--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC ({split})'); plt.legend()
        plt.savefig(os.path.join(out_dir, 'roc_curve.png')); plt.close()

        print(f"Results for {split} saved in {out_dir}")

if __name__ == '__main__':
    # default run
    run_experiments(['A','B','AB'])
