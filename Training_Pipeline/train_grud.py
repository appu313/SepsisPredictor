""
"""train_grud.py

Author: Ehsan Asadollahi
Description: Training script for GRU-D on Sepsis datasets A, B, and AB.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from Model_Definitions.grud_model import GRUDCell

class SepsisGrudDataset(Dataset):
    def __init__(self, npz_dir, labels_csv):
        self.files = sorted(glob.glob(os.path.join(npz_dir, 'patient_*.npz')))
        labels_df = pd.read_csv(labels_csv)
        self.labels = dict(zip(labels_df['patient_id'], labels_df['SepsisLabel']))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        X = torch.from_numpy(data['X']).float()        # (T, D)
        M = torch.from_numpy(data['M']).float()        # (T, D)
        Delta = torch.from_numpy(data['Delta']).float()# (T, D)
        pid = int(os.path.basename(self.files[idx]).split('_')[1].split('.')[0])
        y = torch.tensor(self.labels[pid], dtype=torch.float32)
        return X, M, Delta, y

class GRUDModel(nn.Module):
    def __init__(self, input_size, hidden_size, x_mean):
        super().__init__()
        self.cell = GRUDCell(input_size, hidden_size, x_mean)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, X, M, Delta):
        batch_size, T, D = X.size()
        device = X.device
        h = torch.zeros(batch_size, self.cell.hidden_size, device=device)
        x_prev = torch.zeros(batch_size, D, device=device)
        for t in range(T):
            h, x_prev = self.cell(
                X[:, t, :], M[:, t, :], Delta[:, t, :], h, x_prev
            )
        return self.classifier(h).squeeze(-1)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, M, Delta, y in loader:
        X, M, Delta, y = X.to(device), M.to(device), Delta.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X, M, Delta)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser(description='Train GRU-D on Sepsis dataset')
    parser.add_argument('--set', choices=['A', 'B', 'AB'], required=True,
                        help='Dataset split to use: A, B, or AB')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    # Map to CSV and NPZ directories
    csv_path = f'Data/imputed/data/data/raw/training_set_{args.set}.csv'
    npz_dir = os.path.join('Data', 'grud_inputs', args.set)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Labels file not found: {csv_path}')
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(f'Processed NPZ dir not found: {npz_dir}')

    # Determine feature columns and compute empirical means
    df_raw = pd.read_csv(csv_path)
    exclude_cols = [
        'Age', 'Gender', 'Unit1', 'Unit2',
        'HospAdmTime', 'ICULOS', 'SepsisLabel', 'patient_id'
    ]
    feature_cols = [c for c in df_raw.columns if c not in exclude_cols]
    x_mean = df_raw[feature_cols].mean().fillna(0).values

    # Create dataset & loader
    dataset = SepsisGrudDataset(npz_dir, csv_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model, optimizer, loss setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUDModel(len(feature_cols), args.hidden_size, x_mean).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f'Epoch {epoch}/{args.epochs} - Loss: {loss:.4f}')

    # Save model state
    os.makedirs('Training_Pipeline/models', exist_ok=True)
    save_path = f'Training_Pipeline/models/grud_{args.set}.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

if __name__ == '__main__':
    main()