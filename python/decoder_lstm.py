import numpy as np
import torch
from torch import nn

M = 10 # number of normal distributions for output 

batch_size = 128
N_max = 10 # maximum number of strokes for sketch in dataset
hidden_dim = 2048 # dimension of cell and hidden states
latent_dim = 128 
stroke_dim = 5
input_dim = latent_dim + stroke_dim # + 5 for size of stroke data: (x,y,p1,p2,p3)
output_dim = 6*M + 3

def output_dist(y):
    return y

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # generates initial hidden and cell states from latent vector
        self.fc_in = nn.Linear(latent_dim,2*hidden_dim)

        # Fully connected layer for reducing dimensionality of hidden state before
        # being used for distribution parameters.
        self.fc_proj = nn.Linear(hidden_dim,output_dim)

        self.lstm = nn.LSTM(input_dim,hidden_dim)

        self.hidden = torch.zeros(1,batch_size,hidden_dim)
        self.cell = torch.zeros(1,batch_size,hidden_dim)

    def forward(self, z): 
        # Batch amples from latent space distributions
        z = z.view(batch_size,latent_dim)
        
        # Obtain initial hidden and cell states by splitting result of fc_in along column axis
        self.hidden_cell = torch.split(torch.tanh(self.fc_in(z)), 
                                       [hidden_dim, hidden_dim], 
                                       dim = 1)

        # Data for all strokes in output sequence
        strokes = torch.zeros(N_max + 1, batch_size, stroke_dim)
        strokes[:,0] = torch.tensor([0,0,1,0,0])
  
        # For each stroke 
        for i in range(0,N_max):

            x = torch.cat((z,strokes[i,:,:]),dim = 1)

            out, self.hidden_cell = self.lstm(x,self.hidden_cell)
            strokes[i+1] = output_dist(self.fc_proj(out))
            
        return strokes[1:,:,:] #ignore first stroke
    