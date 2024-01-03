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

class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__() 

        
        self.fc1 = nn.Linear(input_size, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, hidden_dim*2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        
        print(x)
        
        mean_logvar = x.view(-1, 2, latent_dim)
        mean = mean_logvar[:, 0, :]
        logvar = mean_logvar[:, 1, :]
        
        return mean, logvar
    

def train():
    cur_step = 0
    total_loss = 0
        
    for _ in range(n_epochs):
        for image in data:
            print(np.shape(image)[0])
            encoder = Encoder(np.shape(image)[0])
            mean, logvar = encoder(torch.from_numpy(image))
            
            cur_step += 1

        print(f"Mean: {mean} Logvar: {logvar}")
        total_loss = 0
        cur_step = 0

train()