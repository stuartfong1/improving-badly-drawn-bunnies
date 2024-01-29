import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

from encode_pen_state import encode_dataset1
from params import enc_hidden_dim,batch_size,stroke_dim,latent_dim,data,Nmax

#
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(stroke_dim, enc_hidden_dim, bidirectional=True)

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
        hidden, cell = torch.zeros(2, x.shape[1], enc_hidden_dim), torch.zeros(2, x.shape[1], enc_hidden_dim)

        _, (hidden, cell) = self.lstm(x.float(), (hidden, cell))
        hidden_forward_dir, hidden_backward_dir = torch.split(hidden, 1, 0)
        hidden_concatenated = torch.cat([hidden_forward_dir.squeeze(0), hidden_backward_dir.squeeze(0)], 1)

        mu = self.fc_mu(hidden_concatenated)
        sigma = self.fc_sigma(hidden_concatenated)
        return mu, sigma

encoder = Encoder()


def make_batch(size=batch_size):
    """
    Using the data created earlier in the code and a given batch size, randomly fetch
    that many images and return them + their lengths.

    Parameters:
        - size: the size of the batch. Default is the variable batch_size declared
            at the start of the code.

    Returns:
        - batch: a tensor of the batch of random images appended in the order they were fetched in.
        - lengths: the length of each image fetched, in the order they were fetched in.
    """

    batch_ids = np.random.choice(len(data), size)
    batch_images = [data[id] for id in batch_ids]
    lengths = [len(image) for image in batch_images]
    strokes = []
    for image in batch_images:
        new_image = np.zeros((Nmax, 3))
        new_image[:len(image), :] = image[:len(image), :] # copy over values
        new_image[len(image):, :2] = 1 # set leftover empty coordinates to 1 to indicate end
        strokes.append(new_image)

    encoded_strokes = np.stack(encode_dataset1(np.array(strokes)), 1) # don't forget to stack input along dim = 1
    batch = torch.from_numpy(encoded_strokes.astype(float))
    return batch, lengths


def train():
    encoder.train() # set it to training mode

    batch, lengths = make_batch()
    mean, logvar = encoder(batch)
    print(f" mean: {mean.size()} {mean}")
    print(f"logvar: {logvar.size()} {logvar}")


if __name__ == "__main__":
    train()