# ['HR', 'O2Sat', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel', 'patient_id']

import torch
import torch.nn as nn
import pickle
from pathlib import Path
from Training_Pipeline import (
    train_for_evaluation,
    Train_Hyperparameters
)
from Model_Definitions.TCN import TCN, TCN_Hyperparameters
import joblib


sequence_length = 6 # Number of time steps in each sequence
input_features = 40
num_classes = 2 # Sepsis (1) or No Sepsis (0)

train_dataset = joblib.load('/Users/aroudbari3/Deeplearningproj/mitch-baseline-models/data/preprocessed/time_series_data_compressed_2.pkl.z')
criterion_tcn = torch.nn.CrossEntropyLoss()
tcn_hyperparams = TCN_Hyperparameters(
    num_channels=[32, 32],
    kernel_size=2,
    dropout=0.2
)
model_tcn = TCN(input_size=input_features, output_size=num_classes, hyperparameters=tcn_hyperparams)
train_params_tcn = Train_Hyperparameters(batch_size=32, num_epochs=5, learning_rate=0.001)

tcn_model_trained, tcn_history = train_for_evaluation(
        train_set=train_dataset,
        model=model_tcn,
        criterion=criterion_tcn,
        hyperparameters=train_params_tcn,
        val_set = None
    )
print("\nTCN Training History:", tcn_history)