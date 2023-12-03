import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import torch
from torch import nn
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
# Birdirectional Encoder starts at line 62.

# Get file path for one class of sketches
data_path = '/kaggle/input/tinyquickdraw/sketches/sketches/whale.npz'

# Load from file
dataset = np.load(data_path, encoding='latin1', allow_pickle=True)
data = dataset["train"]

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
        data = np.delete(data, i) # np.delete(array, index) returns the array, minus the object at the index  
    else:
        i += 1
        
# get final # of images
print(np.size(data))

# get input dimensions
for i in range(10):
    print(np.shape(data[i]))
    
    
lr = 2e-3
hidden_dim = 128
# TODO: Get the model to train w/ data + display results
# A successful model should be able to take in an image and output a matrix for the hidden state + cell state.

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, hidden_dim, bidirectional=True)
        
        # The activation function
        self.fc = nn.Linear(hidden_dim, 5) 

        # the short term memory of the LSTM
        # starts in a state of all 0's - tuple of 2 three dimensional arrays
        self.hidden_cell = (torch.zeros(1, 1, hidden_dim),
                            torch.zeros(1, 1, hidden_dim)) 
        
    def forward(self, x):
        # changes x to a c x 5 tensor (c being some integer we don't know)
        # also ensures out data matches our LSTM's requirements
        x = x.view(-1, 5)
        
        # updating hidden cell and passing input along
        x, self.hidden_cell = self.lstm(x, self.hidden_cell) 
        
        # self.fc() describes a layer
        # this flattens x, feeds it into a fully connected layer, and gets x as the output
        x = self.fc(x.view(len(x), -1)) 
        
        # get the final output of feeding x into the model
        return x[-1] 
    
model = Model()

optimizer = Adam(model.parameters(), lr=lr)