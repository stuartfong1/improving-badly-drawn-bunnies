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
from decoder_lstm import Decoder, M
from encode_pen_state import encode_dataset1
from SketchesDataset import SketchesDataset
from encoder_lstm import Encoder
from pen_reconstruction_loss import pen_reconstruction_loss
from offset_reconstruction_loss import offset_reconstruction_loss

# import parameters
from params import *

# TODO: Solve NaN gradients problem

# Taken from pruning.py
def graph_values():
    """
    Given a .npz file loaded in numpy arrays, graph a histogram and
    a bell curve of the number of features. Returns and prints the
    mean and std.

    Parameters:
        - data: a .npz file loaded in as numpy arrays.

    Returns:
        - (mean, std): a 2x1 tuple representing the mean and std of the data.
    """

    # Graphing values
    data_val = np.zeros(0)
    for i in data:
        data_val = np.append(data_val, [len(i)])

    plt.hist(data_val, bins=25, density=True, alpha=0.6, color='b') # graph a histogram of our values

    plt.show()

    mean = np.mean(data_val)
    std = np.std(data_val)

    plt.plot(data_val, norm.pdf(data_val, mean, std)) # pdf is probability, density function
    plt.show()

    print("Mean: ", mean)
    print("Standard deviation: ", std)


# Taken from pruning.py
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


# Taken from encoder_lstm.py
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


# Original function taken from normalize_data.py
def normalize_data():

    total_length = 0

    for element in data:
        total_length += (len(element))


    coordinate_list = np.empty((total_length,2))

    i = 0

    for element in data:
        coordinate_list[i:i+len(element),:] = element[:,0:2]
        i+=len(element)

    data_std = np.std(coordinate_list)

    for i, element in enumerate(data):
        data[i] = data[i].astype(np.float32)
        data[i][:,0:2] = element[:,0:2].astype(np.float32)/data_std


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
        if image[i,3] == 1:
            plt.plot(Xplot, Yplot,color='black')
            Xplot.clear()
            Yplot.clear()
        # elif image[i, 4] == 1:
    plt.show()


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder_optimizer = Adam(self.encoder.parameters(), lr = lr)
        self.decoder_optimizer = Adam(self.encoder.parameters(), lr = lr)


    def forward(self, x, N_s = torch.full((batch_size,1),Nmax)):
        mean, logvar = self.encoder(x)

        sample = torch.randn(batch_size, latent_dim)
        std = torch.exp(logvar/2) # logvar / 2 should be a float
        z = mean + std*sample

        # if torch.isnan(z)[0, 0]: # debug code
        #     print() # put a breakpoint here - will let you view variables in the case gradient becomes NaN

        # Code from old run_decoder method

        # Data for all strokes in output sequence
        strokes = torch.zeros(Nmax + 1, batch_size, 5)
        strokes[0,:] = torch.tensor([0,0,1,0,0])

        # Batch of samples from latent space distributions
        z = z.view(batch_size,latent_dim)

        #THIS IS ONLY A PLACEHOLDER SKETCH LENGTH.  IT MUST NOT BE Nmax FOR THE FINAL MODEL
        # replace with equivalent of lengths array


        # Obtain initial hidden and cell states by splitting result of fc_in along column axis
        self.decoder.hidden_cell = torch.split(F.tanh(self.decoder.fc_in(z).view(1,batch_size,2*dec_hidden_dim)),
                                          [dec_hidden_dim, dec_hidden_dim],
                                          dim = 2)
        pen_loss = 0
        offset_loss = 0

        mask, dx, dy, p = make_target(x, N_s)

        # For each timestep, pass the batch of strokes through LSTM and compute
        # the output.  Output of the previous timestep is used as input.
        for i in range(1,Nmax):

            #params will be used for computing loss
            strokes[i], params = self.decoder(z,strokes[i-1])

            input_stroke = x[i]

            # calculate loss at each timestep

            #params[6] is pen_state, input_stroke is the input data
            pen_loss += pen_reconstruction_loss(Nmax,input_stroke[:,2:],params[6])/Nmax
            # size of dx and mu_x (and dy and mu_y) do not match.

            offset_params = [params[i].view(batch_size,M) for i in range(6)]

            offset_loss += offset_reconstruction_loss(
                dx[i].view(batch_size,1),
                dy[i].view(batch_size,1),
                *offset_params,
                mask[i].view(batch_size,1)
            )/Nmax

            #print(f"Loss at step {i} (offset,pen): {offset_loss.item()}, {pen_loss.item()}")

            #for strokes in generated sequence past sequence length, set to [0,0,0,0,1]
            stroke_mask = (i > N_s) # boolean mask set to false when i is larger than sketch size

            for index in range(len(stroke_mask)):
                # stroke_mask[0] = True # Debug Code
                if stroke_mask[index] == True:
                    empty_stroke = torch.tensor([0,0,0,0,1],dtype=torch.float32)
                    strokes[i,index,:] = empty_stroke
    
        return strokes[1:], offset_loss,pen_loss, mean, logvar



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
    w_KL = 1.0  # Weight for the KL divergence part of the loss (can tune as needed, 1.0 is standard)

    # Initialize
    n_step = n_min
    total_loss = 0

    # Training loop
    for step in range(num_training_steps):
        # Calculate n_step
        n_step = 1 - (1 - n_min) * R**step

        # Calculate the total weighted loss
        step_loss = reconstruction_loss + w_KL * n_step * max(kl_loss, KL_min)
        total_loss += step_loss

    return total_loss


def make_target(batch, lengths):
    mask = torch.zeros((Nmax + 1, batch.size()[1]))
    for index, num_strokes in enumerate(lengths):
        mask[:num_strokes, index] = 1

    dx = batch[:, :, 0]
    dy = batch[:, :, 1]
    # copy + append together pen state values
    p = torch.stack([batch.data[:, :, 2], batch.data[:, :, 3], batch.data[:, :, 4]], 2)

    return mask, dx, dy, p


def train():
    print("Training loop running...\n")
    
    cur_step = 0
    total_loss = 0
    for epoch in range(n_epochs):
        batch, lengths = make_batch(batch_size)

        model.encoder_optimizer.zero_grad()
        model.decoder_optimizer.zero_grad()

        # Run predictions - [n * batch * 5] fed in, similar shape should come out
        output, l_s,l_p, mean, logvar = model(batch)

        #pen_state = torch.Tensor(params[-1]).squeeze()
        #mixture_weights, mean_x, mean_y, std_x, std_y, corr_xy = torch.stack(params[:-1], 1).squeeze(0)
        
        l_r = l_s+l_p
        l_kl = kl_loss(logvar, mean)
    
        # get the total loss from the model forward(), add it to our kl

        if anneal_loss:
            # the first parameter is num_training_steps, which is tunable
            loss = anneal_kl_loss(20, l_r, l_kl)
        else:
            loss = l_r + w_kl * l_kl # had an incident w/ epoch 3 having over 1k loss - investigate?

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}\n")
        print(f"l_kl: {l_kl:.4f}\n l_s: {l_s:.4f}\n l_r: {l_p:.4f}\n")     
        
        print("Backpropagating error...")
        loss.backward()
        
        grad_threshold = 1.0 # tunable parameter, prevents exploding gradient
        nn.utils.clip_grad_norm_(model.encoder.parameters(), grad_threshold)
        nn.utils.clip_grad_norm_(model.decoder.parameters(), grad_threshold)

        # update encoder and decoder parameters using adam algorithm
        model.encoder_optimizer.step()
        model.decoder_optimizer.step()

        # print(model.encoder_optimizer.param_groups) # More Debug Code 
        print("Done!")
        print("---------------------------------------------------------")
        #
        # if n_epochs % 5 == 0:
        #     # draw image
        #     display_encoded_image(output[:, np.random.randint(batch_size), :])



model = VAE()

if __name__ == "__main__":
    train()