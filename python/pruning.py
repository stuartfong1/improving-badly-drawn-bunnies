import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math


def graph_values(data):
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

if __name__ == "__main__":
    
    # Get file path for one class of sketches
    data_path = '/kaggle/input/tinyquickdraw/sketches/sketches/whale.npz'
    
    # Load from file
    dataset = np.load(data_path, encoding='latin1', allow_pickle=True)
    data = dataset["train"]

    graph_values(data)
    prune_data(data)