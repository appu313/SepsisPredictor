import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, auc
import numpy as np
from tqdm import tqdm


def roc_eval(model, dataset: TensorDataset, batch_size=32, quiet=False):
    """Evaluates the model on the provided test dataset using the ROC curve

    Args:
        model (torch.nn.Module): Untrained model, initialized with desired hyperparameters
        dataset (TensorDataset): Test dataset
        batch_size (int, optional): Data batch size. Defaults to 32.
        quiet (bool, optional): When true, disables progress bars. Defaults to False.

    Returns:
        Tuple containing
        - auroc (float): The area under the ROC curve
        - best_threshold (float): The probability threshold that maximizes the Youden J statistic
        - roc_curve (tuple[ndarray, ndarray]): Tuple of false positive rates and true positive rates for plotting the ROC curve
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, disable=quiet):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if outputs.dim() > 1 and outputs.size(1) > 1:
                probs = torch.softmax(outputs, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(outputs.squeeze())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    auroc = auc(fpr, tpr)
    

    # maximize the Youden J statistic to ensure best class separation
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    
    return auroc, best_threshold, (fpr, tpr)
