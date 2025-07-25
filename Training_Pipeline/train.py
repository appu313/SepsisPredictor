import gc
import torch
import joblib
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from tqdm.notebook import tqdm


class Train_Val_Task:
    def __init__(
        self,
        train_set,
        val_set,
        train_params,
        model_params,
        input_size,
        output_size,
        model_type,
        loss_function,
        pipe,
        id,
    ):
        self.train_set = train_set
        self.val_set = val_set
        self.train_params = train_params
        self.model_params = model_params
        self.input_size = input_size
        self.output_size = output_size
        self.model_type = model_type
        self.loss_function = loss_function
        self.pipe = pipe
        self.id = id

    def post_result(self, result):
        self.pipe.send(result)
        self.pipe.close()


class Train_Hyperparameters:
    def __init__(self, batch_size, num_epochs, learning_rate):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
    def __str__(self):
        lines = [
            "╔" + "═" * 30 + "╗",
            "║        Train Hyperparameters        ║",
            "╠" + "═" * 30 + "╣",
            f"  Batch Size      : {self.batch_size}",
            f"  Number of Epochs: {self.num_epochs}",
            f"  Learning Rate   : {self.learning_rate}",
            "╚" + "═" * 30 + "╝",
        ]

        max_label_width = max(len(line.split(":")[0]) for line in lines[3:-1])
        formatted_lines = lines[:3]

        for line in lines[3:-1]:
            label, value = line.split(":")
            formatted_lines.append(f"{label.ljust(max_label_width)} :   {value.strip()}")

        formatted_lines.append(lines[-1])
        return "\n".join(formatted_lines)



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
                    flat_grid.append(
                        Train_Hyperparameters(
                            batch_size=b, num_epochs=n, learning_rate=l
                        )
                    )
        return flat_grid


def get_batched_data(train_set, val_set=None, batch_size=32):
    train_sampler = RandomSampler(train_set)
    if not val_set is None:
        val_sampler = RandomSampler(val_set)
        val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler)
    else:
        val_loader = None
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    return train_loader, val_loader


def train_validate(
    train_set: TensorDataset,
    val_set: TensorDataset,
    model,
    criterion,
    hyperparameters: Train_Hyperparameters,
    quiet=False,
    gpu_id=None,
):
    device = None
    if gpu_id is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), hyperparameters.learning_rate)
    model = model.to(device)
    train_loader, val_loader = get_batched_data(
        train_set=train_set, val_set=val_set, batch_size=hyperparameters.batch_size
    )
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(hyperparameters.num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, disable=quiet):
            inputs, targets = (
                batch_x.to(device),
                batch_y.type(torch.LongTensor).to(device),
            )
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
                inputs, targets = (
                    val_x.to(device),
                    val_y.type(torch.LongTensor).to(device),
                )
                outputs = model(inputs)
                val_loss += criterion(outputs.squeeze(), targets).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if not quiet:
            print(
                f"Epoch {epoch + 1}/{hyperparameters.num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
            )
    return model, history


def process_train_task(task: Train_Val_Task, gpu_id) -> None:
    device = None
    if gpu_id is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    model = task.model_type(
        input_size=task.input_size,
        output_size=task.output_size,
        hyperparameters=task.model_params,
    )
    model = model.to(device)

    hyperparameters = task.train_params
    optimizer = torch.optim.Adam(model.parameters(), hyperparameters.learning_rate)
    criterion = task.loss_function()

    train_set = joblib.load(task.train_set)
    val_set = joblib.load(task.val_set)
    train_loader, val_loader = get_batched_data(
        train_set=train_set, val_set=val_set, batch_size=hyperparameters.batch_size
    )
    history = []

    for _ in range(hyperparameters.num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            inputs, targets = (
                batch_x.to(device),
                batch_y.type(torch.LongTensor).to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_x, val_y in val_loader:
                inputs, targets = (
                    val_x.to(device),
                    val_y.type(torch.LongTensor).to(device),
                )
                outputs = model(inputs)
                val_loss += criterion(outputs.squeeze(), targets).item()

        avg_val_loss = val_loss / len(val_loader)
        history.append(avg_val_loss)

    task.post_result(history[-1])
    del model, train_set, val_set, train_loader, val_loader
    gc.collect()


def train_for_evaluation(
    train_set: TensorDataset,
    model,
    criterion,
    hyperparameters: Train_Hyperparameters,
    quiet=False,
    gpu_id=None,
):
    device = None
    if gpu_id is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), hyperparameters.learning_rate)
    model = model.to(device)
    train_loader, _ = get_batched_data(
        train_set=train_set, batch_size=hyperparameters.batch_size
    )
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(hyperparameters.num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, disable=quiet):
            inputs, targets = (
                batch_x.to(device),
                batch_y.type(torch.LongTensor).to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        if not quiet:
            print(
                f"Epoch {epoch + 1}/{hyperparameters.num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f}"
            )
    return model, history
