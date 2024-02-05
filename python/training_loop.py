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
import os
from datetime import datetime

# import from other files
from SketchesDataset import SketchesDataset
from encode_pen_state import encode_dataset1
from autoencoder import VAE
from normalize_data import normalize_data

# import parameters
from params import *

# TODO: Solve y = x problem

# # Taken from encoder_lstm.py
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
        new_image[len(image):, 2] = 1 # set leftover empty coordinates to 1 to indicate end
        strokes.append(new_image)

    encoded_strokes = np.stack(encode_dataset1(np.array(strokes),lengths), 1) # don't forget to stack input along dim = 1
    batch = torch.from_numpy(encoded_strokes.astype(float))
    return batch, torch.tensor(lengths)

normalize_data()

# based on display(imageNum) from displayData.py
def display_encoded_image(image):
    """
    For some image tensor, draw the image using matplotlib.

    Parameters:
        - image: some [n*5] tensor representing a sketch.
    Returns:
        - none
    """
    #Xplot and Yplot are array of points that will be plotted
    Xplot = [0]
    Yplot = [0]
    #Keeps track of the current point that is being drawn
    xpos = 0
    ypos = 0
    #For loop to go through data and plot points
    i=0
    for i in range(len(image)):
        xpos += float(image[i,0])
        ypos += float(image[i,1])
        Xplot.append(-xpos)
        Yplot.append(-ypos)
        if image[i,3] == 0:
            plt.plot(Xplot, Yplot,color='black')
            Xplot.clear()
            Yplot.clear()
        # elif image[i, 4] == 1:
    plt.show()

# Taken from KL_loss.py
def kl_loss(sigma_hat, mu):
    # torch.sum is added to sum over all the dimensions of the vectors
    return (-0.5 / latent_dim) * torch.sum(1 + sigma_hat - torch.square(mu) - torch.exp(sigma_hat))


# Taken from anneal_kl_loss.py
def anneal_kl_loss(num_training_steps, reconstruction_loss, kl_loss):
    # Hyperparameters
    n_min = 0.01  # Starting Value from paper
    R = 0.9995  # R is a term close to but less than 1.
    KL_min = 0.1 # Value from paper (needs to be between 0.1 and 0.5)

    # Initialize
    n_step = n_min
    total_loss = 0

    # Training loop
    for step in range(num_training_steps):
        # Calculate n_step
        n_step = 1 - (1 - n_min) * R**step

        # Calculate the total weighted loss
        step_loss = reconstruction_loss + w_kl * n_step * max(kl_loss, KL_min)
        total_loss += step_loss

    return total_loss

def train():
    print("Training loop running...\n")
    
    for epoch in range(n_epochs):
        batch, lengths = make_batch(batch_size)
        
        batch = batch.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        
        # Run predictions - [n * batch * 5] fed in, similar shape should come out
        output, l_s,l_p, mean, logvar = model(batch,lengths) 
        
        #pen_state = torch.Tensor(params[-1]).squeeze()
        #mixture_weights, mean_x, mean_y, std_x, std_y, corr_xy = torch.stack(params[:-1], 1).squeeze(0)
        
        l_r = l_s+l_p
        l_kl = kl_loss(mean, logvar)
    
        # get the total loss from the model forward(), add it to our kl

        if anneal_loss:
            # until dataloader is added, epochs and steps are the same.

            # Hyperparameters
            n_min = 0.01  # Starting Value from paper
            R = 0.9995  # R is a term close to but less than 1.
            KL_min = 0.1 # Value from paper (needs to be between 0.1 and 0.5)

            # Calculate n_step
            n_step = 1 - (1 - n_min) * R**epoch

            # Calculate the total weighted loss
            loss = l_r + w_kl * n_step * max(l_kl, KL_min)
        else:
            loss = l_r + w_kl * l_kl # had an incident w/ epoch 3 having over 1k loss - investigate?
        
        loss.backward()
        
        grad_threshold = 1.0 # tunable parameter, prevents exploding gradient
        nn.utils.clip_grad_norm_(model.encoder.parameters(), grad_threshold)
        nn.utils.clip_grad_norm_(model.decoder.parameters(), grad_threshold)

        # update encoder and decoder parameters using adam algorithm
        optimizer.step()

        # print(model.encoder_optimizer.param_groups) # More Debug Code 
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
        print(f"l_kl: {l_kl:.4f} l_s: {l_s:.4f} l_p: {l_p:.4f}") 
        print("---------------------------------------------------------\n")
        
        # if epoch % 5 == 0:
        #     save_model()
        
        if epoch % 5 == 0:
           #draw image
           display_encoded_image(output[:, 0, :])
           display_encoded_image(batch[:, 0, :])

def save_model():
    '''
    Saves model parameters to "model/temp".  An epoch is saved as t_{n}, where
    n enumerates the parameter files.  Note that these get VERY big.   
    '''
    
    dir = "model/temp/"
    
    if not os.path.isdir(dir):
        os.mkdir(dir)
    
    filename = f"{str(datetime.now())}.pt"
    files = os.listdir(dir)
    
    while len(files) >= 3:
        os.remove(dir + files.pop(0))
        
    torch.save({
            "model": model.state_dict(),
            "opt": optimizer.state_dict()
        }, dir + filename)

if __name__ == "__main__":
    
    model = VAE().to(device)
    optimizer = Adam(model.parameters(), lr = lr)    
    # weights = os.listdir("model/final") # directory does not exist on my pc atm, feel free to change
    
    if pretrained and not len(os.listdir("model/final")) == 0:
        weights = os.listdir("model/final")
        loaded_state = torch.load(f"model/final/{weights[0]}", map_location=device)
        model.load_state_dict(loaded_state['model'])
        optimizer.load_state_dict(loaded_state['opt'])
    
    train()
    
    
