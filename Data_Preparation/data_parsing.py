"""
data_parsing.py

Team member: Mitch
Task: Data parsing and handling missing values
"""

import pandas as pd
import torch
import numpy as np
from tqdm.notebook import tqdm, trange


def parse_and_clean_data(
    df: pd.DataFrame,
    id_col="patient_id",
    temporal_col="ICULOS",
    label_col="SepsisLabel",
    missing_values="mask",
) -> torch.tensor:
    """
    Parses data from DataFrame and handles missing values.

    Args:
      df (pd.DataFrame): DataFrame containing data to parse and clean
      group_identifier (str): Column of DataFrame specifying the patient record id
      temporal_column (str): Column of DataFrame indexing time steps
      missing_values (str): Stragegy for handling missing values;

    missing_values Options:
      'mask': Fill in missing values with zeros. Add mask columns to data to indicate missingness of features in each row.
      'impute': Impute missing values using forward-fill, then backward-fill, then pop. median.
      'mask-impute': Impute missing values using forward-fill, then backward-fill, then pop. median. Add mask columns to data to indicate missingness of features in each row


    Returns:
      torch.tensor: The parsed and cleaned data, shape: (# records, # time steps, # features)
    """
    if not id_col in df.columns.to_list():
        raise ValueError("group_identifier must be a column in the DataFrame")

    patients = df[id_col].unique()
    pop_medians = df.median()

    if missing_values == "mask":
        mask = df.isnull().astype(int)
        mask = mask.drop(columns=[id_col, temporal_col, label_col]).add_suffix(
            "_missing"
        )
        df = df.fillna(0)
        return pd.concat([df, mask], axis=1)
    elif missing_values == "impute":
        records = []
        for p in tqdm(patients):
            # Chained forward and backward fill
            p_record = df[df[id_col] == p].copy(deep=True)
            p_record = p_record.sort_values(temporal_col, ascending=True)
            p_record.ffill().bfill()

            # Population median imputation to fill remaining null entries
            missing_val_cols = p_record.columns[p_record.isnull().any()].to_list()
            for col in missing_val_cols:
                p_record[col] = p_record[col].fillna(pop_medians[col])
            records.append(p_record)
        new_df = pd.concat(records)
        nan_cols = new_df.columns[new_df.isna().any()].to_list()
        new_df = new_df.drop(columns=nan_cols)
        return new_df
    elif missing_values == "mask-impute":
        mask = df.isnull().astype(int)
        mask = mask.drop(columns=[id_col, temporal_col, label_col]).add_suffix(
            "_missing"
        )
        df = pd.concat([df, mask], axis=1)
        records = []
        for p in tqdm(patients):
            # Chained forward and backward fill
            p_record = df[df[id_col] == p].copy(deep=True)
            p_record.sort_values(temporal_col, ascending=True)
            mask = p_record.isnull().astype(int)
            mask = mask.drop(columns=[id_col, temporal_col, label_col])
            p_record.ffill().bfill()

            # Population median imputation to fill remaining null entries
            missing_val_cols = p_record.columns[p_record.isnull().any()].to_list()
            for col in missing_val_cols:
                p_record[col] = p_record[col].fillna(pop_medians[col])
            records.append(p_record)
        new_df = pd.concat(records)
        nan_cols = new_df.columns[new_df.isna().any()].to_list()
        nan_cols_mask = [x + "_missing" for x in nan_cols]
        new_df = new_df.drop(columns=(nan_cols + nan_cols_mask))
        return new_df
    else:
        raise ValueError(
            "missing_values must be either 'mask', 'impute', or 'mask-impute'"
        )
