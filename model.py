"""ClaMIDIa model module.

This module implements a modification of MusicBERT , which is a transformer model
designed to predict masked tokens in an input music sequence. The model
consists of a transformer encoder and a set of linear layers for predicting
each element in the octuple token.

Example:
    num_tokens = [256, 128, 129, 256, 128, 32, 254, 49]
    model = MusicBERT(d_model=768, nhead=12, num_layers=12, dim_feedforward=2048, dropout=0.1, activation='gelu',
    num_tokens=num_tokens)
"""

import torch
import torch.nn as nn
from typing import List


class Clamidia(nn.Module):
    """ClaMIDIa Model class .

    This class implements a modification of a MusicBERT model, which is a transformer model
    designed to predict masked tokens in an input music sequence.

    Attributes:
        encoder: The transformer encoder.
        linear_layers: A list of linear layers for predicting each element in the octuple token.
        softmax: The softmax function for generating probability distributions.
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dropout: float,
            num_layers: int,
            dim_feedforward: int,
            num_tokens: List[int],
            activation: str):
        """Initializes the ClaMIDIa model.

        Args:
            d_model: The dimension of the input vectors.
            nhead: The number of heads in the multihead attention models.
            dropout: The dropout value
            num_layers: The number of sub-encoder-layers in the transformer encoder.
            dim_feedforward: The dimension of the feedforward network model.
            num_tokens: A list of the sizes of the vocabularies for each element type.
            activation: The activation function to use ('relu' or 'gelu').
        """
        super(Clamidia, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            dim_feedforward=dim_feedforward, activation=activation
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, num_tokens[i]) for i in range(8)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Performs a forward pass through the model.

        Args:
            x: The input tensor.

        Returns:
            A list of tensors, each representing the probability distribution
            over possible values for a corresponding element in the octuple token.
        """
        x = self.encoder(x)
        outputs = [self.softmax(linear(x)) for linear in self.linear_layers]
        return outputs
