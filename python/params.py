import numpy as np
import torch
from math import sqrt

# --------- VAE PARAMETERS --------------

latent_dim = 128
stroke_dim = 5

# decoder 
M = 10 # number of normal distributions for output 
T = 0.1# temperature parameter
sqrtT = sqrt(T) # saves computation
dec_hidden_dim = 2048 # dimension of cell and hidden states

# encoder
enc_hidden_dim = 2048 # dimension of cell and hidden states

# --------- TRAINING PARAMETERS ----------

lr = 2e-3 # Used to be 2e-3 but got NaN gradients
batch_size = 128 # modification here requires modification in decoder_lstm.py
latent_dim = 128
n_epochs = 150
w_kl = 0.7 # weight for loss calculation, can be tuned if needed
anneal_loss = True # True if train using annealed kl loss, False otherwise
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrained = False
# --------- DATA LOADING -----------------

# Get file path for one class of sketches
# data_path = '/kaggle/input/tinyquickdraw/sketches/sketches/whale.npz'
#for debugging purposes, comment the above line and decomment the below line
# data_path = 'python/ambulance.npz' # ambulance.npz is stored alongside the files on my computer, feel free to change
data_path = 'ambulance.npz'
dataset = np.load(data_path, encoding='latin1', allow_pickle=True)
data = dataset["train"]
Nmax = max([len(i) for i in data])

