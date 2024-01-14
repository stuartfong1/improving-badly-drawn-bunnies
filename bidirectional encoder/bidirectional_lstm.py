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


# Get file path for one class of sketches
data_path = '/kaggle/input/tinyquickdraw/sketches/sketches/whale.npz'

# Load from file
dataset = np.load(data_path, encoding='latin1', allow_pickle=True)
data = dataset["train"]

lr = 2e-3
hidden_dim = 2048
latent_dim = 128
num_features = 3 # will change to 5 with pen state encoding
batch_size = 100
Nmax = max([len(i) for i in data])

class Encoder(nn.Module):
    def __init__(self, feature_number=num_features):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(feature_number, hidden_dim, bidirectional=True)
        # self.fc_mu = nn.Linear(2*hidden_dim, latent_dim)
        # self.fc_logvar = nn.Linear(2*hidden_dim, latent_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, 512)
        self.fc_2 = nn.Linear(512, latent_dim)

    def forward(self, x, batch_size):
        """
        Runs a batch of images through the encoder and returns its latent vector.
        Does not normalize values on its own.

        Parameters:
         - x: Tensor of shape [max_strokes, batch_size, num_features]
            where max_strokes is the highest number of points possible for an image in the batch.
            x should be normalized.
        - batch_size: int representing the current batch size.

        Returns:
        - z: Tensor of shape [batch_size, 2*hidden_dim] representing the latent vector of the batch.
        - mu: the mean of the distribution of values
        - logvar: log of the variance of the distribution of values
        """

        # Get the hidden states
        hidden, cell = torch.zeros(2, batch_size, hidden_dim), torch.zeros(2, batch_size, hidden_dim)

        _, (hidden, cell) = self.lstm(x.float(), (hidden, cell))
        hidden_forward_dir, hidden_backward_dir = torch.split(hidden, 1, 0)
        hidden_concatenated = torch.cat([hidden_forward_dir.squeeze(0), hidden_backward_dir.squeeze(0)], 1)

        # Get the latent vector representation of the data
        hidden_concatenated = self.fc_1(hidden_concatenated)
        hidden_concatenated = F.relu(hidden_concatenated)
        hidden_concatenated = self.fc_2(hidden_concatenated)

        return hidden_concatenated


encoder = Encoder()


def make_batch(size=batch_size):
    """
    Using the data created earlier in the code and a given batch size, randomly fetch
    that many images and return them + their lengths.

    Parameters:
        - size: the size of the batch. Default is the variable batch_size declared
            at the start of the code.

    Returns:
        - batch: the batch of random images appended in the order they were fetched in.
        - lengths: the length of each image fetched, in the order they were fetched in.
    """

    batch_ids = np.random.choice(len(data), batch_size)
    batch_images = [data[id] for id in batch_ids]
    lengths = [len(image) for image in batch_images]
    strokes = []
    for image in batch_images:
        new_image = np.zeros((Nmax, num_features))

        # need to change the following when pen state is encoded
        new_image[:len(image), :] = image[:len(image), :] # copy over values
        new_image[len(image):, :2] = 1 # set leftover empty coordinates to 1 to indicate end

        strokes.append(new_image)
    return torch.from_numpy(np.stack(strokes, 1)).float(), lengths


def train():
    encoder.train() # set it to training mode

    batch, lengths = make_batch()
    latent_vector = encoder(batch, batch_size)
    print(f"{latent_vector.size()}, {latent_vector}")


if __name__ == "__main__":
    train()