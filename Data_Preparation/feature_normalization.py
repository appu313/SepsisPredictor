from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GroupKFold
import torch


def center(data, columns):
    scaler = StandardScaler()
    scaler.fit(data[columns])
    return scaler.transform(data[columns])


def smote_oversample_to_tensor(X, y):
    smote = SMOTE(random_state=42)
    X_f = X.reshape(X.shape[0], -1)
    X_r, y_r, *_ = smote.fit_resample(X_f, y)
    X_r = X_r.reshape(-1, X.shape[-2], X.shape[-1])
    return torch.tensor(X_r, dtype=torch.float32), torch.tensor(
        y_r, dtype=torch.float32
    )


def train_validate_split(
    df, k=5, group_col="patient_id", label_col="SepsisLabel", seed=42
):
    gkf = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    groups = df[group_col].values
    labels = df.groupby(group_col)[label_col].transform("max").values
    return list(gkf.split(df, labels, groups))

