import numpy as np
from math import sqrt

# --------- VAE PARAMETERS --------------

latent_dim = 128
stroke_dim = 5

# decoder 
M = 10 # number of normal distributions for output 
T = 0.5 # temperature parameter
sqrtT = sqrt(T) # saves computation
dec_hidden_dim = 2048 # dimension of cell and hidden states

# encoder
enc_hidden_dim = 2048 # dimension of cell and hidden states

# --------- TRAINING PARAMETERS ----------

lr = 2e-3 # Used to be 2e-3 but got NaN gradients
batch_size = 50 # modification here requires modification in decoder_lstm.py
latent_dim = 128
n_epochs = 20
w_kl = 0.5 # weight for loss calculation, can be tuned if needed
anneal_loss = False # True if train using annealed kl loss, False otherwise

# --------- DATA LOADING -----------------

# Get file path for one class of sketches
# data_path = '/kaggle/input/tinyquickdraw/sketches/sketches/whale.npz'
#for debugging purposes, comment the above line and decomment the below line
data_path = 'python/ambulance.npz'
dataset = np.load(data_path, encoding='latin1', allow_pickle=True)
data = dataset["train"]
Nmax = max([len(i) for i in data])

