from sklearn.model_selection import StratifiedGroupKFold

def stratified_group_k_fold(df, k, group_col='patient_id', label_col='SepsisLabel', seed=42):
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    groups = df[group_col].values
    labels = df.groupby(group_col)[label_col].transform("max").values

    return list(sgkf.split(df, labels, groups))