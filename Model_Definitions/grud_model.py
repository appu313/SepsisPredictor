"""
grud_model.py

Author: Ehsan Asadollahi
Description: PyTorch implementation of GRU‑D cell (Gated Recurrent Unit with Decay).
             Implements decay-based imputation and hidden-state decay as per Che et al. (2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUDCell(nn.Module):
    """
    GRU-D cell.

    Args:
        input_size (int): number of input features
        hidden_size (int): number of hidden units
        x_mean (array-like or tensor, shape [input_size]): empirical feature means
    """
    def __init__(self, input_size, hidden_size, x_mean):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # register feature-wise empirical mean for imputation
        x_mean = torch.as_tensor(x_mean, dtype=torch.float32)
        self.register_buffer('x_mean', x_mean.view(1, -1))

        # decay rates for inputs and hidden state
        self.gamma_x_layer = nn.Linear(input_size, input_size)
        self.gamma_h_layer = nn.Linear(input_size, hidden_size)

        # GRUCell consumes [x_tilde; mask] as input
        self.gru_cell = nn.GRUCell(input_size * 2, hidden_size)

    def forward(self, x, m, delta, h_prev, x_prev):
        """
        Forward pass for one time step.

        Args:
            x (Tensor): raw input at time t, shape (batch, input_size)
            m (Tensor): mask (1=observed, 0=missing), same shape as x
            delta (Tensor): time since last observation, same shape as x
            h_prev (Tensor): previous hidden state, shape (batch, hidden_size)
            x_prev (Tensor): previous imputed input, shape (batch, input_size)

        Returns:
            h_new (Tensor): new hidden state, shape (batch, hidden_size)
            x_tilde (Tensor): imputed input at time t, shape (batch, input_size)
        """
        # compute decay rates
        gamma_x = torch.exp(-F.relu(self.gamma_x_layer(delta)))
        gamma_h = torch.exp(-F.relu(self.gamma_h_layer(delta)))

        # input imputation
        x_tilde = m * x + (1.0 - m) * (
            gamma_x * x_prev + (1.0 - gamma_x) * self.x_mean
        )

        # hidden state decay
        h_tilde = gamma_h * h_prev

        # concatenate imputed input and mask for GRUCell input
        gru_input = torch.cat([x_tilde, m], dim=1)

        # GRUCell update
        h_new = self.gru_cell(gru_input, h_tilde)

        return h_new, x_tilde
