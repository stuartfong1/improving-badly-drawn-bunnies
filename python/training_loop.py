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

from decoder_lstm import Decoder
from displayData import display
from encode_pen_state import encode_pen_state, encode_dataset1, encode_dataset2
from gaussian_mixture_model import gaussian_mixture_model, sample
# from normalize-data import normalize_data (may need to rename file, name is giving errors)
from sample_from_distribution import sample_latent_vector
from SketchesDataset import SketchesDataset

# make pruning a method and import it
# from bidirectional_encoder import Encoder

lr = 2e-3
batch_size = 128
latent_dim = 5 # originally 5
# this is the # of vector dimensions at the bottleneck
# at the latent space, we represent each image using a vector with latent_dim dimensions
# check 11/12/2023 Slides for the encoder-decoder model
n_epochs = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
