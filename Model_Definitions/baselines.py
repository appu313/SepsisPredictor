import torch
from torch.autograd import forward_ad
import torch.nn as nn


class Basline_Model_Hyperparameters:
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers


class Baseline_Model_Hyperparameter_Grid:
    """Hyperparameter grid for baseline model"""

    def __init__(self, hidden_size_range, num_layers_range):
        self.hidden_size_grid = [h for h in hidden_size_range]
        self.num_layers_grid = [l for l in num_layers_range]

    def get_flattened_grid(self) -> list[Basline_Model_Hyperparameters]:
        flat_grid = []
        for h in self.hidden_size_grid:
            for l in self.num_layers_grid:
                flat_grid.append(
                    Basline_Model_Hyperparameters(hidden_size=h, num_layers=l)
                )
        return flat_grid


class Baseline_GRU(nn.Module):
    """Baseline GRU model"""

    def __init__(
        self, input_size, output_size, hyperparameters: Basline_Model_Hyperparameters
    ):
        """
        Parameters:
        -----------
        input_size:
          Number of features in input
        hidden_size:
          Dimensionality of each hidden state h_i
        num_layers:
          Number of GRU layers
        output_size:
          Number of output categories
        """
        super(Baseline_GRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hyperparameters.hidden_size,
            num_layers=hyperparameters.num_layers,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hyperparameters.hidden_size, output_size), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        _, h = self.gru(input=x)
        return self.classifier(h[-1])


class Baseline_LSTM(nn.Module):
    """Baseline LSTM model"""

    def __init__(
        self, input_size, output_size, hyperparameters: Basline_Model_Hyperparameters
    ):
        """
        Parameters:
        -----------
        input_size:
          Number of features in input
        hidden_size:
          Dimensionality of each hidden state h_i and each cell state c_i
        num_layers:
          Number of LSTM layers
        output_size:
          Number of output categories
        """
        super(Baseline_LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hyperparameters.hidden_size,
            num_layers=hyperparameters.hidden_size,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hyperparameters.hidden_size, output_size), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Parameters:
        -----------
        x:
          Input sequence, size=(N, L, D).
          N: Batch size
          L: Length of sequence
          D: Input size
        """
        _, (h, _) = self.lstm(input=x)
        return self.classifier(h[-1])
