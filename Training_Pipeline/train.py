import torch
from torch._prims_common import Tensor

from torch.utils.data import(
    Sampler,
    TensorDataset,
    DataLoader,
    RandomSampler,
    SubsetRandomSampler,
    dataset
)

from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm

def train_val_split(dataset_fold, val_size):
    X = dataset_fold['X_train']
    y = dataset_fold['y_train']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    train_set = TensorDataset(X_train_tensor, y_train_tensor)
    val_set = TensorDataset(X_val_tensor, y_val_tensor)
    return train_set, val_set

def get_batched_data(train_set, val_set, batch_size):
    train_sampler = RandomSampler(train_set)
    val_sampler = RandomSampler(val_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler)
    return train_loader, val_loader

def train(dataset_fold, model, optimizer, criterion, num_epochs, batch_size=52, val_size=0.3, positive_threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_set, val_set = train_val_split(dataset_fold, val_size)
    train_loader, val_loader = get_batched_data(train_set=train_set, val_set=val_set, batch_size=batch_size)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader):
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
            for val_x, val_y in tqdm(val_loader):
                inputs, targets = val_x.to(device), val_y.type(torch.LongTensor).to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs.squeeze(), targets).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | ")
    return model, history

