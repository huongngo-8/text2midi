import torch
import torch.nn as nn
from typing import List
from muzic.musicbert.musicbert import *
from transformers import AutoModel



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
            in_dim: int,
            h1: int,
            h2: int,
            latent_dim: int):
        """Initializes the ClaMIDIa model.

        Args:
            in_dim: Output dimension from MusicBERT (#notes * 8 * embed_dim)
            h1: Hidden dimension for 1st linear layer
            h2: Hidden dimension for 2nd linear layer
            latent_dim: Dimension for the shared MIDI-text dimension space
        """
        super(Clamidia, self).__init__()
        self.music_enc = MusicBERTModel.from_pretrained('.', checkpoint_file = './checkpoint_last_musicbert_small_w_genre_head.pt',
                                                        user_dir='muzic/musicbert')
        self.lin1 = nn.Linear(in_dim, h1)
        self.lin2 = nn.Linear(h1, h2)
        self.lin3 = nn.Linear(h2, latent_dim)

    def forward(self, mus: torch.Tensor) -> List[torch.Tensor]:
        """Performs a forward pass through the model.

        Args:
            x: The input tensor.

        Returns:
            A list of tensors, each representing the probability distribution
            over possible values for a corresponding element in the octuple token.
        """
        mus_enc = self.music_enc.extract_features(mus)
        x = self.lin1(mus_enc)
        x = nn.GELU(self.lin2(x))
        return nn.GELU(self.lin3(x))
