"""
grud_preprocessing.py

Author: Ehsan Asadollahi
Description: GRU-D preprocessing utilities for clinical time-series data.
             Generates (X, Mask, Delta) from raw per-patient ICU records.

"""

import numpy as np
import pandas as pd

def generate_grud_input(patient_df: pd.DataFrame,
                        features: list,
                        time_col: str = 'ICULOS') -> tuple:
    """
    Generate GRU-D ready input matrices: X (values), M (mask), Δ (delta)

    Args:
        patient_df (pd.DataFrame): Time-ordered per-patient dataframe
        features (list): List of feature columns to use
        time_col (str): Name of time index column (default = 'ICULOS')

    Returns:
        X (np.ndarray): Input values with NaNs filled as 0, shape (T, D)
        M (np.ndarray): Mask matrix, 1 = observed, 0 = missing, shape (T, D)
        Δ (np.ndarray): Time since last observation per feature, shape (T, D)
    """
    # Ensure time ordering
    df = patient_df.sort_values(by=time_col).reset_index(drop=True)

    # Extract feature matrix (T, D)
    raw_values = df[features].to_numpy()  # shape (T, D)
    mask = ~np.isnan(raw_values)          # True where observed
    X = np.nan_to_num(raw_values, nan=0.0)

    # Compute delta matrix (T, D)
    T, D = X.shape
    delta = np.zeros((T, D), dtype=np.float32)

    # First time step has delta 0
    delta[0, :] = 0

    for d in range(D):
        last_observed_time = None
        for t in range(1, T):
            if mask[t-1, d]:
                last_observed_time = df.loc[t-1, time_col]
            if mask[t, d]:
                delta[t, d] = 0
                last_observed_time = df.loc[t, time_col]
            else:
                if last_observed_time is None:
                    delta[t, d] = 0
                else:
                    delta[t, d] = df.loc[t, time_col] - last_observed_time

    # Convert boolean mask to float: 1 = observed, 0 = missing
    M = mask.astype(np.float32)

    return X.astype(np.float32), M, delta
