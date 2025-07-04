import torch
from torch.utils.data import(
    TensorDataset,
    DataLoader,
    RandomSampler
)

from tqdm.notebook import tqdm

class Train_Hyperparameters:
  def __init__(self, batch_size, num_epochs, learning_rate):
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.learning_rate = learning_rate

class Train_Hyperparameter_Grid:
  """Hyperparameter grid for baseline model"""
  def __init__(self, batch_size_range, num_epochs_range, learning_rate_range):
    self.batch_size_grid = [b for b in batch_size_range]
    self.num_epochs_grid = [n for n in num_epochs_range]
    self.learning_rate_grid = [l for l in learning_rate_range]
  
  def get_flattened_grid(self) -> list[Train_Hyperparameters]:
    flat_grid = []
    for b in self.batch_size_grid:
      for n in self.num_epochs_grid:
        for l in self.learning_rate_grid:
          flat_grid.append(Train_Hyperparameters(batch_size=b, num_epochs=n, learning_rate=l))
    return flat_grid


def get_batched_data(train_set, val_set, batch_size):
    train_sampler = RandomSampler(train_set)
    val_sampler = RandomSampler(val_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler)
    return train_loader, val_loader

def train_validate(train_set: TensorDataset, val_set: TensorDataset, model, criterion, hyperparameters: Train_Hyperparameters, quiet=False, gpu_id=None):
  device = None
  if gpu_id is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu")
  
  optimizer = torch.optim.Adam(model.parameters(), hyperparameters.learning_rate)
  model.to(device)
  train_loader, val_loader = get_batched_data(train_set=train_set, val_set=val_set, batch_size=hyperparameters.batch_size)
  history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
  
  for epoch in range(hyperparameters.num_epochs):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in tqdm(train_loader, disable=quiet):
      inputs, targets = batch_x.to(device), batch_y.type(torch.LongTensor).to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs.squeeze(), targets)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    model.eval()
    with torch.no_grad():
      val_loss = 0.0
      for val_x, val_y in tqdm(val_loader, disable=quiet):
        inputs, targets = val_x.to(device), val_y.type(torch.LongTensor).to(device)
        outputs = model(inputs)
        val_loss += criterion(outputs.squeeze(), targets).item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    
    if not quiet:
      print(f"Epoch {epoch+1}/{hyperparameters.num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | ")
  return model, history


def train_for_evaluation(train_set: TensorDataset, model, criterion, hyperparameters: Train_Hyperparameters, quiet=False, gpu_id=None):
  device = None
  if gpu_id is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu")
  
  optimizer = torch.optim.Adam(model.parameters(), hyperparameters.learning_rate)
  model.to(device)
  train_loader, val_loader = get_batched_data(train_set=train_set, val_set=val_set, batch_size=hyperparameters.batch_size)
  history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
  
  for epoch in range(hyperparameters.num_epochs):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in tqdm(train_loader, disable=quiet):
      inputs, targets = batch_x.to(device), batch_y.type(torch.LongTensor).to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs.squeeze(), targets)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)
    
    if not quiet:
      print(f"Epoch {epoch+1}/{hyperparameters.num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f}")
  return model, history
