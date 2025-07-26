# evaluate.py (UPDATED)

import argparse
import torch
import joblib
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from Model_Definitions.TCN import TCN 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_test_data(fold_idx=0, batch_size=64):
    base = Path(f"data/preprocessed/fold_{fold_idx}")
    test_path = base / "test" / "compressed_test_dataset.pkl.z"
    return DataLoader(joblib.load(test_path), batch_size=batch_size, shuffle=False)

def evaluate(args):
    # Load test data
    test_loader = load_test_data(args.fold, args.batch_size)

    # Determine input size
    sample_X, _ = next(iter(test_loader))
    input_size = sample_X.shape[2]

    # Initialize model and load weights
    model = TCN(input_size=input_size)
    model_path = Path("models") / f"{args.model}_fold{args.fold}.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze(-1).cpu().numpy()
            preds.extend(outputs.flatten())
            trues.extend(y_batch.numpy().flatten())

    auc = roc_auc_score(trues, preds)
    print(f"Test AUC: {auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tcn", help="Model name (tcn, lstm, etc.)")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    evaluate(args)
