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
        self.n_layers = n_layers


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
        x = self.pe[: x.size(0)]
        return self.dropout(x)


# multilayer transformer encoder -> decoder layer -> binary classifier
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
        
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embedding_dim,
            nhead=self.n_heads,
            dim_feedforward=self.feedforward_hidden_dim,
            dropout=self.dropout_p,
            activation=self.activation_type,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=self.n_layers
        )
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer, num_layers=self.n_layers
        )
        
        self.classifier = nn.Linear(self.embedding_dim, output_size)

    def forward(self, x):
        """_summary_

        Args:
            x (Tensor): Input batch of sequences with shape (Batch Size, Seq Len, Input Dim)
        """
        N, _, _ = x.shape
        device = x.device
        embeddings = self.embedding_conv(torch.transpose(x, 1, 2))
        embeddings = torch.transpose(embeddings, 1, 2)
        pos_encodings = self.pos_encoding(x)
        pos_encoded_embeddings = embeddings + pos_encodings
        encoder_output = self.encoder(pos_encoded_embeddings)  # N, L, D
        tgt = torch.zeros(size=(N, 1, self.embedding_dim), device=device)
        decoder_output = self.decoder(tgt=tgt, memory=encoder_output)
        return self.classifier(decoder_output[:, -1, :])
