import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Sepsis_Predictor_Encoder_Hyperparameters:
    def __init__(
        self,
        embedding_dim: int,
        feedforward_hidden_dim: int,
        n_heads: int,
        activation: str,
        n_layers: int = 1,
        dropout_p: float = 0.0,
        pos_encoding_dropout_p: float = 0.0,
        interpolation_coeff: int = 24,
    ):
        """_summary_

        Args:
            embedding_dim (int): _description_
            feedforward_hidden_dim (int): _description_
            n_heads (int): _description_
            activation (str): _description_
            n_layers (int, optional): _description_. Defaults to 1.
            dropout_p (float, optional): _description_. Defaults to 0.0.
            pos_encoding_dropout_p (float, optional): _description_. Defaults to 0.0.
            interpolation_coeff (int, optional): _description_. Defaults to 24.
        """
        if not (activation == "relu" or activation == "gelu"):
            raise ValueError(
                f'Invalid activation type - {activation}: choose "relu" or "gelu"'
            )
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.feedforward_hidden_dim = feedforward_hidden_dim
        self.activation = activation
        self.dropout_p = dropout_p
        self.pos_encoding_dropout_p = pos_encoding_dropout_p
        self.n_layers = (n_layers,)
        self.interpolation_coeff = interpolation_coeff


# Shamelessly stolen straight from pytorch's docs
# Source: https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    """Positional encoding module"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize a positional encoding layer.

        Args:
            d_model (int): Model embedding dimension.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            max_len (int, optional): Maximum sequence length. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0)) / d_model
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# Inspired by : https://arxiv.org/pdf/2203.14469v1
class Dense_Interpolator(nn.Module):
    """Dense Interpolation Module

    Implements dense interpolation algorithm from (Wang, Zhao, et al)
    """

    def __init__(self, interpolation_coeff):
        super(Dense_Interpolator, self).__init__()
        self.I = interpolation_coeff

    def forward(self, x):
        _, L, _ = x.shape
        i_idx = torch.arange(1, self.I + 1).float()  # (I)
        l_idx = torch.arange(1, L + 1).float()  # (L)
        fractional_positions = i_idx.unsqueeze(1) * l_idx.unsqueeze(0) / L  # (I, L)
        coef = (
            1 - torch.abs(fractional_positions - i_idx.unsqueeze(1)) / self.I
        ) ** 2  # (I, L)
        coef = coef.unsqueeze(0).unsqueeze(3)  # (1, I, L, 1)
        x_expanded = x.unsqueeze(1)  # (N, 1, L, D)
        weighted = coef * x_expanded
        z = weighted.sum(dim=2)  # (N, I, D)
        return z


# Inspired by : https://arxiv.org/pdf/2203.14469v1
class Sepsis_Predictor_Encoder(nn.Module):
    """Transformer Encoder model"""

    def __init__(
        self,
        input_size,
        output_size,
        hyperparameters: Sepsis_Predictor_Encoder_Hyperparameters,
    ):
        """_summary_

        Args:
            input_size (_type_): _description_
            output_size (_type_): _description_
            hyperparameters (Transformer_Model_Hyperparameters): _description_
        """
        super(Sepsis_Predictor_Encoder, self).__init__()
        self.embedding_dim = hyperparameters.embedding_dim
        self.n_heads = hyperparameters.n_heads
        self.n_layers = hyperparameters.n_layers
        self.feedforward_hidden_dim = hyperparameters.feedforward_hidden_dim
        self.activation_type = hyperparameters.activation
        self.dropout_p = hyperparameters.dropout_p
        self.pos_encoding_dropout_p = hyperparameters.pos_encoding_dropout_p
        self.interpolation_coeff = hyperparameters.interpolation_coeff

        # 1x1 conv to project input time series into embedding space
        self.embedding_conv = nn.Conv1d(
            in_channels=input_size, out_channels=self.embedding_dim, kernel_size=1
        )
        self.pos_encoding = PositionalEncoding(
            d_model=self.embedding_dim, dropout=self.pos_encoding_dropout_p
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.n_heads,
            dim_feedforward=self.feedforward_hidden_dim,
            dropout=self.dropout_p,
            activation=self.activation_type,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=self.n_layers
        )
        self.interpolator = Dense_Interpolator(
            interpolation_coeff=self.interpolation_coeff
        )
        self.feedforward_classifier = nn.Sequential(
            nn.Linear(
                self.interpolation_coeff * self.embedding_dim, self.embedding_dim
            ),
            nn.ReLU() if self.activation_type == "relu" else nn.GELU(),
            nn.Linear(self.embedding_dim, output_size),
        )
        self.feedforward = nn.Linear(
            self.interpolation_coeff * self.embedding_dim, self.embedding_dim
        )
        self.classifier = nn.Linear(self.embedding_dim, output_size)

    def forward(self, x):
        """_summary_

        Args:
            x (Tensor): Input batch of sequences with shape (Batch Size, Seq Len, Input Dim)
        """
        embeddings = self.embedding_conv(torch.transpose(x, 1, 2))
        embeddings = torch.transpose(embeddings, 1, 2)
        pos_encodings = torch.transpose(self.pos_encoding(x), 0, 1)
        pos_encoded_embeddings = embeddings + pos_encodings
        encoder_output = self.encoder(pos_encoded_embeddings)  # N, L, D
        dense_rep = self.interpolator(encoder_output)
        dense_rep = dense_rep.reshape(-1, self.interpolation_coeff * self.embedding_dim)
        return self.feedforward_classifier(dense_rep)
