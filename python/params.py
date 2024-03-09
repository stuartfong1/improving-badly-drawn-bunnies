import torch
from dataloading import datapaths, datasets, Nclass

# --------- VAE PARAMETERS --------------
stroke_dim = 5

# decoder https://arxiv.org/pdf/1704.03477.pdf
M = 20 # number of normal distributions for output 
T = 0.1 # temperature parameter
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

conditional = True
Nmax = 200
