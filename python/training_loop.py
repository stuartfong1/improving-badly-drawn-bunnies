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
from encode_pen_state import encode_dataset1
from SketchesDataset import SketchesDataset
from encoder_lstm import Encoder
from anneal_kl_loss import anneal_kl_loss



# Get file path for one class of sketches
# data_path = '/kaggle/input/tinyquickdraw/sketches/sketches/whale.npz'
data_path = 'ambulance.npz'

# Load from file
dataset = np.load(data_path, encoding='latin1', allow_pickle=True)
data = dataset["train"]

lr = 2e-3
batch_size = 128
latent_dim = 128
n_epochs = 20
Nmax = max([len(i) for i in data])

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()


    def forward(self, x):
        mean, logvar = self.encoder(x)

        sample = torch.randn(batch_size, latent_dim)
        std = torch.exp(logvar/2) # logvar / 2 should be a float
        z = mean + std*sample

        x = self.decoder(z)
        return x, mean, logvar


# Taken from strokes_reconstruction_loss.py
def bivariate_normal_pdf(dx, dy, mu_x, mu_y, std_x, std_y, corr_xy):
    """
    Return N(dx, dy | mu_x, mu_y, std_x, std_y, corr_xy)
    """
    z_x = (dx - mu_x) / std_x
    z_y = (dy - mu_y) / std_y
    exponent = -(z_x ** 2 - 2 * corr_xy * z_x * z_y + z_y ** 2) / 2 * (1 - corr_xy ** 2)
    norm = 1 / (2 * np.pi * std_x * std_y * torch.sqrt(1-corr_xy ** 2))
    return norm * torch.exp(exponent)


# Taken from strokes_reconstruction_loss.py
def reconstruction_loss(dx, dy, mu_x, mu_y, std_x, std_y, corr_xy, pi, mask):
    """
    pi: The mixture probabilities
    mask: 1 if the point is not after the final stroke, 0 otherwise

    Returns the reconstruction loss for the strokes, L_s
    """
    pdf = bivariate_normal_pdf(dx, dy, mu_x, mu_y, std_x, std_y, corr_xy)
    return -(1/(Nmax * batch_size)) * torch.sum(mask * torch.log(torch.sum(pi * pdf, axis=0)))


# Taken from strokes_reconstruction_loss.py
def vae_loss():
    l_r = reconstruction_loss()
    l_kl = 0
    return l_s + l_kl


model = VAE()
optimizer = Adam(model.parameters(), lr = lr) # make sure the model learns from the loss functions


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


def train():
    cur_step = 0
    total_loss = 0
    for _ in range(n_epochs):
        batch, _ = make_batch(batch_size)
        # Run predictions - [n * batch * 5] fed in, similar shape should come out
        output, mean, logvar = model(batch)
        print(f"output: {output.shape}") # [num_strokes, num_images, num_features]
        print(f"mean: {mean.shape}")
        print(f"logvar: {logvar.shape}")
        if n_epochs % 5 == 0:
            # draw image
            for i in range(batch_size):
                display_encoded_image(output[:, i, :])


if __name__ == "__main__":
    train()