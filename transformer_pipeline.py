import os
import glob
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import Dataset, DataLoader
from Model_Definitions import Sepsis_Predictor_Encoder, Sepsis_Predictor_Encoder_Hyperparameters
from Training_Pipeline import Train_Hyperparameters

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


def plot_roc_curve(fpr, tpr, auc, split_name, out_dir):
    plt.figure(); plt.plot(fpr, tpr, label=f'AUC={auc:.3f}'); plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC ({split_name})'); plt.legend()
    plt.savefig(os.path.join(out_dir, 'roc_curve.png')); plt.close()

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
        auc = roc_auc_score(labels, preds)
        print(f"Epoch {epoch:2d} | train loss {train_losses[-1]:.4f}"
                f" | val loss {eval_losses[-1]:.4f} | val AUROC {auc:.4f}")
        
    # save plots
    epochs_range = [e for e in range(1, train_params.num_epochs + 1)]
    fpr, tpr, _ = roc_curve(labels, preds)
    return (epochs_range, train_losses, eval_losses), (fpr, tpr), auc

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
        embedding_dim=48,
        feedforward_hidden_dim=64,
        n_heads=4,
        activation='gelu',
        n_layers=6,
        dropout_p=0,
        pos_encoding_dropout_p=0
    )
    
    train_params = Train_Hyperparameters(
        batch_size=16,
        num_epochs=30,
        learning_rate=5e-5
    )
    
    model = Sepsis_Predictor_Encoder(
        input_size=num_features,
        output_size=1,
        hyperparameters=hyperparams
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    
    print('------ Starting Training ------')
    loss_grid, roc_grid, auroc = train_eval_transformer(model, criterion, train_ds, test_ds, train_params)
    
    epochs, train_loss, eval_loss = loss_grid
    fpr, tpr = roc_grid
    
    plot_loss_curve(epochs, train_loss, eval_loss, split_name=args.split_name, out_dir=fig_dir)
    plot_roc_curve(fpr, tpr, auroc, args.split_name, fig_dir)
    
    

if __name__ == '__main__':
    main()