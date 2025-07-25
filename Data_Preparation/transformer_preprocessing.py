"""
transformer_preprocessing.py

Author: Mitchell Teunissen
Description: Transformer model preprocessing utilities for clinical time-series data.
             Generates (X, Mask, Delta) from raw per-patient ICU records.

"""
import os
import glob
import numpy as np
import pandas as pd


def generate_transformer_input(patient_df: pd.DataFrame,
                                xmean_df: pd.DataFrame,
                                features: list,
                                time_col: str = 'ICULOS',
                                observation_period: int=12) -> tuple:
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
    
    # Drop patients with under 12 hour observation periods
    if len(df) < 12:
        return None
    
    # Repeat last row until observation period is satisfied
    if len(df) < observation_period:
        df = df.append(df.iloc[[-1] * (observation_period - len(df))])
    
    assert len(df) >= observation_period
    
    df = df.iloc[:observation_period]
    
    # Extract feature matrix (T, D)
    raw_values = df[features]
    mask = ~np.isnan(raw_values.to_numpy())          # True where observed
    
    # Forward/Backward fill impute
    raw_values = raw_values.ffill()
    raw_values = raw_values.bfill()
    for c in features:
        raw_values[c] = raw_values[c].fillna(xmean_df[c])

    X = raw_values.to_numpy() # shape (T, D)
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

    return np.concatenate((X.astype(np.float32), M, delta), axis=1)
