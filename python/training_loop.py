import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm

# import from other files
from decoder_lstm import distribution, Decoder
from displayData import display
from encode_pen_state import encode_pen_state, encode_dataset1, encode_dataset2
from gaussian_mixture_model import gaussian_mixture_model, sample
from normalize_data import normalize_data
from sample_from_distribution import sample_latent_vector
from SketchesDataset import SketchesDataset
from pruning import graph_values, prune_data
from encoder_lstm import Encoder, make_batch


lr = 2e-3
batch_size = 100
latent_dim = 128
n_epochs = 20

# Get file path for one class of sketches
# data_path = '/kaggle/input/tinyquickdraw/sketches/sketches/whale.npz'
data_path = 'ambulance.npz'

# Load from file
dataset = np.load(data_path, encoding='latin1', allow_pickle=True)
data = dataset["train"]

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()


    def forward(self, x):
        mean, logvar = self.encoder(x)


        sample = torch.randn(batch_size, latent_dim)
        std = torch.exp(logvar)
        z = mean + std*sample

        x = self.decoder(z)
        return x, mean, logvar


model = VAE()
# optimizer = Adam(model.parameters(), lr = lr)


def train():
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    cur_step = 0
    total_loss = 0
    for _ in range(n_epochs):
        for image, _ in tqdm(dataloader):
            # Run predictions
            output, mean, logvar = model(image)

if __name__ == "__main__":
    pass