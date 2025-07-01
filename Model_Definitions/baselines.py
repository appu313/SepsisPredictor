import torch
from torch.autograd import forward_ad
import torch.nn as nn

class Baseline_GRU(nn.Module):
  """Baseline GRU model"""

  def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0):
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
    dropout:
      If >0, add dropout layer with probability p=dropout on output of each GRU layer.
    """
    super(Baseline_GRU, self).__init__()

    self.gru = nn.GRU(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      dropout=dropout,
      batch_first=True
    )

    self.classifier = nn.Sequential(
      nn.Linear(hidden_size, output_size),
      nn.Softmax(dim=-1)
    )
  
  def forward(self, x):
    _, h = self.gru(input=x)
    return self.classifier(h[-1])


class Baseline_LSTM(nn.Module):
  """Baseline LSTM model"""

  def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0):
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
    dropout:
      If >0, add dropout layer with probability p=dropout on output of each LSTM layer.
    """
    super(Baseline_LSTM, self).__init__()

    self.lstm = nn.LSTM(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      dropout=dropout,
      batch_first=True
    )

    self.classifier = nn.Sequential(
      nn.Linear(hidden_size, output_size),
      nn.Softmax(dim=-1)
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


