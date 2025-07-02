import torch
import joblib
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
from Training_Pipeline import train, Train_Hyperparameter_Grid
from torch.utils.data import TensorDataset
from Model_Definitions import (
  Baseline_GRU,
  Baseline_LSTM,
  Baseline_Model_Hyperparameter_Grid,
  Basline_Model_Hyperparameters
)


class Hyperparameter_Grid:
  def __init__(self, train_grid: Train_Hyperparameter_Grid, model_grid):
    self.train_grid = train_grid
    self.model_grid = model_grid
  
  def get_flattened_grid(self):
    flat_grid = []
    for train_param in self.train_grid.get_flattend_grid():
      for model_param in self.model_grid.get_flattened_grid():
        flat_grid.append((train_param, model_param))
    return flat_grid

def grid_search_tune(data_fold_path: Path, hyperparameter_grid: Hyperparameter_Grid, model_type, loss_function, input_size, output_size):
  if not data_fold_path.is_dir():
    raise AttributeError('Could not locate data fold directory')
  
  TRAINING_DIR = data_fold_path.joinpath('train')
  VALIDATE_DIR = data_fold_path.joinpath('validate')
  
  if not TRAINING_DIR.is_dir():
    raise AttributeError('Could not locate training sets within data fold directory')
  elif not VALIDATE_DIR.is_dir():
    raise AttributeError('Could not loacate validation sets within data fold directory')
  
  train_inner_fold_paths = [
    p for p in TRAINING_DIR.glob('*.pkl.z') if p.name.split('.pkl.z')[0].split('_')[-1] != 'full'
  ]
  val_inner_fold_paths = [
    p for p in VALIDATE_DIR.glob('*.pkl.z')
  ]
  
  inner_fold_paths = []
  for vp in val_inner_fold_paths:
    fold_id = vp.name.split('.pkl.z')[0].split('_')[-1]
    for tp in train_inner_fold_paths:
      if tp.name.split('.pkl.z')[0].split('_')[-1] == fold_id:
        inner_fold_paths.append((tp, vp))
  
  # Grid search
  best_score = None
  best_params = None
  for train_params, model_params in tqdm(hyperparameter_grid.get_flattened_grid()):
    inner_scores = []
    for train_path, val_path in inner_fold_paths:
      train_set: TensorDataset = joblib.load(train_path)
      val_set: TensorDataset = joblib.load(val_path)
      model = model_type(input_size=input_size, output_size=output_size, hyperparameters=model_params)
      criterion = loss_function()
      model, history = train(
        train_set=train_set, 
        val_set=val_set, 
        model=model, 
        criterion=criterion, 
        hyperparameters=train_params, 
        quiet=True
      )
      inner_scores.append(history['val_loss'][-1])
      del model, train_set, val_set
    avg_score = np.mean(inner_scores)
    if best_score is None or avg_score < best_score:
      best_params = (train_params, model_params)
      best_score = avg_score
  return best_params
