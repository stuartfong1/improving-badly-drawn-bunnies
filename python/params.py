import torch
import math
import numpy as np
# --------- VAE PARAMETERS --------------

stroke_dim = 5

# decoder https://arxiv.org/pdf/1704.03477.pdf
M = 20 # number of normal distributions for output 
T = 1# temperature parameter
# saves computation

latent_dim = 128
feature_dim = 32
dec_hyper_dim = 64
dec_hidden_dim = 2048 # dimension of cell and hidden states
enc_hidden_dim = 512 # dimension of cell and hidden states

# --------- TRAINING PARAMETERS ----------

lr = 1e-4# Used to be 2e-3 but got NaN gradients
batch_size = 2# batch_size >= 1, used to be 128

n_epochs = 20
w_kl = 0.99 # weight for loss calculation, can be tuned if needed
anneal_loss = True # True if train using annealed kl loss, False otherwise
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters for annealing loss
n_min = 0.01  # Starting Value from paper
R = 0.9999  # R is a term close to but less than 1.
KL_min = 0.2 # Value from paper (needs to be between 0.1 and 0.5)

datapaths = [
  "data/remote/apple.npz",
  "data/remote/flower.npz",
  "data/remote/cactus.npz",
  "data/remote/carrot.npz"
    ]

Nclass = len(datapaths)

datasets = [np.load(path, encoding='latin1', allow_pickle=True) for path in datapaths]

def prune_data(data, num_std=2.5):
    """
    Given a .npz file loaded in numpy arrays, prune the dataset based on its mean, std,
    and a user-specified number of std (default is 2.5).

    Parameters:
        - data: a .npz file loaded in as numpy arrays.
        - num_std: optional int for specifying how many images should be pruned.
                    As num_std increases, less images are pruned, and vice versa.
                    This should be a value between 0 and 3.

    Returns:
         - none
    """
    # get initial # of images
    print(f"Initial number of images: {np.size(data)}")

    # Get mean + std
    data_val = np.zeros(0)
    for i in data:
        data_val = np.append(data_val, [len(i)])
    mean = np.mean(data_val)
    std = np.std(data_val)

    lower = math.floor(mean - std*num_std)  # the lower bound of points allowed
    upper = math.ceil(mean + std*num_std) # the upper bound of points allowed

    i = 0
    while i < np.size(data):
        num_points = np.shape(data[i])[0] # gets number of points
        if num_points < lower or num_points > upper:
            data = np.delete(data, i)
        else:
            i += 1

    # get final # of images
    print(f"Number of images after pruning: {np.size(data)}")

for dataset in datasets:
  prune_data(dataset["train"],3)
  prune_data(dataset["test"],3)

Nmax = max([
  max([len(i) for i in dataset["train"]] + [len(i) for i in dataset["test"]]) 
  for dataset in datasets
  ])

