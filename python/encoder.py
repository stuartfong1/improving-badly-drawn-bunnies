import numpy as np
from scipy.stats import norm
import torch
from torch import nn
from params import *

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        if conditional:
            self.input_dim = stroke_dim + Nclass
        else:
            self.input_dim = stroke_dim

        self.lstm = nn.LSTM(self.input_dim, enc_hidden_dim, bidirectional=True)

        self.fc_mu = nn.Linear(2*enc_hidden_dim, latent_dim)
        self.fc_sigma = nn.Linear(2*enc_hidden_dim, latent_dim)

    def forward(self, x):
        """
        Runs a batch of images through the encoder and returns its latent vector.
        Does not normalize values on its own.

        Parameters:
         - x: Tensor of shape [max_strokes, batch_size, num_features]
            where max_strokes is the highest number of points possible for an image in the batch.
            x should be normalized.
        - batch_size: int representing the current batch size.

        Returns:
        - mu: Tensor of shape [batch_size, 2*hidden dim] representing the mean of the distribution of values
        - sigma: Tensor of shape [batch_size, 2*hidden dim] representing the log of the distribution of values
        """

        # Get the hidden states
        hidden, cell = torch.zeros(2, (batch_size), enc_hidden_dim,device=device), torch.zeros(2, (batch_size), enc_hidden_dim,device=device)

        _, (hidden, cell) = self.lstm(x.float(), (hidden, cell))
        hidden_forward_dir, hidden_backward_dir = torch.split(hidden, 1, 0)
        hidden_concatenated = torch.cat([hidden_forward_dir.squeeze(0), hidden_backward_dir.squeeze(0)], 1)

        mu = self.fc_mu(hidden_concatenated)
        sigma = self.fc_sigma(hidden_concatenated)
        return mu, sigma

