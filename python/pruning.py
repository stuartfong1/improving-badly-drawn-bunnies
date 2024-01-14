import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

# Get file path for one class of sketches
data_path = '/kaggle/input/tinyquickdraw/sketches/sketches/whale.npz'

# Load from file
dataset = np.load(data_path, encoding='latin1', allow_pickle=True)
data = dataset["train"]

# Graphing values
data_val = np.zeros(0)
for i in data:
    data_val = np.append(data_val, [len(i)])

plt.hist(data_val, bins=25, density=True, alpha=0.6, color='b') # graph a histogram of our values

plt.show()

mean = np.mean(data_val)
sd = np.std(data_val)

plt.plot(data_val, norm.pdf(data_val, mean, sd)) # pdf is probability, density function
plt.show()

print("Mean: ", mean)
print("Standard deviation: ", sd)

# get initial # of images
print(np.size(data))

num_st = 2.5 # should be a value in (0, 3) 

lower = math.floor(mean - sd*num_st)  # the lower bound of points allowed
upper = math.ceil(mean + sd*num_st) # the upper bound of points allowed

i = 0
while i < np.size(data):
    num_points = np.shape(data[i])[0] # gets number of points
    if num_points < lower or num_points > upper:
        data = np.delete(data, i) 
    else:
        i += 1
        
# get final # of images
print(np.size(data))

