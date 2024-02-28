# Load the data in here before anything else so the right values for Nclass and Nmax is
# generated, if that is relevant. 

import math
import numpy as np

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

do_pruning = False

datapaths = [
  "data/remote/apple.npz",
  "data/remote/flower.npz",
  "data/remote/cactus.npz",
  "data/remote/carrot.npz"
    ]

Nclass = len(datapaths)

datasets = [np.load(path, encoding='latin1', allow_pickle=True) for path in datapaths]

if do_pruning:
    for dataset in datasets:
      prune_data(dataset["train"],3)
      prune_data(dataset["test"],3)

# Nmax = max([
#   max([len(i) for i in dataset["train"]] + [len(i) for i in dataset["test"]]) 
#   for dataset in datasets
#   ])