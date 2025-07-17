import torch
import joblib
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Training_Pipeline import train_validate, process_train_task, Train_Hyperparameter_Grid, Train_Val_Task
from torch.utils.data import TensorDataset

class Hyperparameter_Grid:
  def __init__(self, train_grid: Train_Hyperparameter_Grid, model_grid):
    self.train_grid = train_grid
    self.model_grid = model_grid
  
  def get_flattened_grid(self):
    flat_grid = []
    for train_param in self.train_grid.get_flattened_grid():
      for model_param in self.model_grid.get_flattened_grid():
        flat_grid.append((train_param, model_param))
    return flat_grid

def grid_search_tune_worker(gpu_id, task_queue, msg_queue):
  while True:
    task: Train_Val_Task | None = task_queue.get()
    if task is None:
      break
    id = task.id
    process_train_task(task=task, gpu_id=gpu_id)
    msg_queue.put(id)

def grid_search_tune_parallel(data_fold_path: Path, hyperparameter_grid: Hyperparameter_Grid, model_type, loss_function, input_size, output_size, num_gpus):
  if not data_fold_path.is_dir():
    raise AttributeError('Could not locate data fold directory')
  TRAINING_DIR = data_fold_path.joinpath('train')
  VALIDATE_DIR = data_fold_path.joinpath('validate')
  if not TRAINING_DIR.is_dir():
    raise AttributeError('Could not locate training sets within data fold directory')
  elif not VALIDATE_DIR.is_dir():
    raise AttributeError('Could not loacate validation sets within data fold directory')
  train_inner_fold_paths = [p for p in TRAINING_DIR.glob('*.pkl.z') if p.name.split('.pkl.z')[0].split('_')[-1] != 'full']
  val_inner_fold_paths = [p for p in VALIDATE_DIR.glob('*.pkl.z')]
  inner_fold_paths = []
  for vp in val_inner_fold_paths:
    fold_id = vp.name.split('.pkl.z')[0].split('_')[-1]
    for tp in train_inner_fold_paths:
      if tp.name.split('.pkl.z')[0].split('_')[-1] == fold_id:
        inner_fold_paths.append((tp, vp))
  
  # Set up worker processes
  task_queue_dict = {}
  procs = []
  gpu_ids = [id for id in range(num_gpus)]
  msg_queue = mp.Queue()
  for id in gpu_ids:
    task_q = mp.Queue()
    p = mp.Process(target=grid_search_tune_worker, args=(id, task_q, msg_queue))
    p.start()
    procs.append(p)
    task_queue_dict[id] = task_q
  
  # Grid search
  best_score = None
  best_params = None
  for train_params, model_params in enumerate(tqdm(hyperparameter_grid.get_flattened_grid())):
    res_pipe_dict = {}
    scores = []
    for fold, (train_path, val_path) in enumerate(inner_fold_paths):
      r_conn, s_conn = mp.Pipe(duplex=False)
      task = Train_Val_Task(
        train_set=train_path,
        val_set=val_path,
        train_params=train_params,
        model_params=model_params,
        model_type=model_type,
        loss_function=loss_function,
        input_size=input_size,
        output_size=output_size,
        pipe=s_conn,
        id=fold
      )
      task_queue_dict[fold % num_gpus].put(task)
      res_pipe_dict[fold] = r_conn
    completed_tasks = 0
    while completed_tasks < len(res_pipe_dict.keys()):
      f_id = msg_queue.get()
      scores.append(res_pipe_dict[f_id].recv())
      completed_tasks += 1 
    avg_score = np.mean(scores)
    if best_score is None or avg_score < best_score:
      best_params = (train_params, model_params)
      best_score = avg_score
  
  for q in task_queue_dict.values():
    q.put(None)
  
  for p in procs:
    p.join()
  
  return best_params

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
    print(f'Training and validating on {len(inner_fold_paths)} inner folds.')
    for fold, (train_path, val_path) in enumerate(inner_fold_paths):
      train_set: TensorDataset = joblib.load(train_path)
      val_set: TensorDataset = joblib.load(val_path)
      model = model_type(input_size=input_size, output_size=output_size, hyperparameters=model_params)
      criterion = loss_function()
      model, history = train_validate(
        train_set=train_set, 
        val_set=val_set, 
        model=model, 
        criterion=criterion, 
        hyperparameters=train_params,
        quiet=True,
      )
      score = history['val_loss'][-1]
      inner_scores.append(score)
      del model, train_set, val_set
    avg_score = np.mean(inner_scores.get())
    if best_score is None or avg_score < best_score:
      best_params = (train_params, model_params)
      best_score = avg_score
  return best_params
