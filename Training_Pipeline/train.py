import torch
from torch.utils.data import(
    TensorDataset,
    DataLoader,
    RandomSampler
)

from tqdm.notebook import tqdm

def get_batched_data(train_set, val_set, batch_size):
    train_sampler = RandomSampler(train_set)
    val_sampler = RandomSampler(val_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler)
    return train_loader, val_loader

def train(train_set: TensorDataset, val_set: TensorDataset, model, optimizer, criterion, num_epochs, batch_size=52):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
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
