from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GroupKFold
import torch
import numpy as np
from collections import Counter

def center(data, columns):
    scaler = StandardScaler()
    scaler.fit(data[columns])
    return scaler.transform(data[columns])


def smote_oversample_to_tensor(X, y):
    """
    Oversamples the minority class(es) using SMOTE to reach 50% of the majority class count,
    then converts the resampled data to PyTorch tensors.

    Args:
        X (np.ndarray): The input features (e.g., NumPy array).
        y (np.ndarray): The target labels (e.g., NumPy array).

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The resampled features.
            - torch.Tensor: The resampled labels.
    """
    # Ensure X and y are NumPy arrays for imblearn compatibility
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    # Reshape X for SMOTE: (n_samples, n_features)
    original_shape = X.shape
    X_f = X.reshape(X.shape[0], -1)

    # Determine class counts
    class_counts = Counter(y)
    if not class_counts:
        raise ValueError("The target labels 'y' are empty or invalid.")

    majority_class_label = max(class_counts, key=class_counts.get)
    majority_class_count = class_counts[majority_class_label]

    # Calculate target samples for minority classes: 50% of majority class count
    target_minority_count = int(0.5 * majority_class_count)

    # Create sampling_strategy dictionary
    sampling_strategy = {}
    for label, count in class_counts.items():
        if label != majority_class_label:
            # Oversample minority classes to the calculated target count
            sampling_strategy[label] = target_minority_count
        else:
            # Keep the majority class as is
            sampling_strategy[label] = count

    # Initialize SMOTE with the custom sampling_strategy
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)

    # Resample
    X_r, y_r = smote.fit_resample(X_f, y)

    # Reshape X_r back to original feature dimensions
    X_r = X_r.reshape(-1, *original_shape[1:])

    # Convert to PyTorch tensors
    return torch.tensor(X_r, dtype=torch.float32), torch.tensor(y_r, dtype=torch.float32)



def train_validate_split(df, k=5, group_col='patient_id', label_col='SepsisLabel', seed=42):
    gkf = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    groups = df[group_col].values
    labels = df.groupby(group_col)[label_col].transform("max").values
    return list(gkf.split(df, labels, groups))