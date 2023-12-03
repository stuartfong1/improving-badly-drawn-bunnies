import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt

M = 10 # number of normal distributions for output 
T = 0.5 # temperature parameter
sqrtT = sqrt(T) # 

batch_size = 128
N_max = 10 # maximum number of strokes for sketch in dataset
hidden_dim = 2048 # dimension of cell and hidden states
latent_dim = 128 
stroke_dim = 5
input_dim = latent_dim + stroke_dim # + 5 for size of stroke data: (x,y,p1,p2,p3)

output_dim = 6*M + 3


def distribution(decoder_output):
    """
    Input: 
        decoder_output (6M + 3): Decoder LSTM output
    Return:
        mixture_weights (M): Mixture weights (probability of a point being in distribution i)
        mean_x (M): The x-values of the means of each distribution
        mean_y (M): The y-values of the means of each distribution
        std_x (M): The standard deviations of the x-values
        std_y (M): The standard deviations of the y-values
        corr_xy (M): The correlation coefficients of the x and y values
        q (3): The predicted pen state (pen_down, pen_up, <EOS>)
    """
    # Split the decoder output into 
    # [pi, mean_x, mean_y, std_x, std_y, rho_xy] and [q1,q2,q3]
    parameters = torch.split(decoder_output, 6, 1)

    # Chunk the parameters together, then stack them 
    # so that each column defines a distribution
    mixture_parameters = torch.stack(parameters[:-1])

    # Split mixture parameters into each parameter
    mixture_weights, mean_x, mean_y, std_x, std_y, corr_xy = torch.split(mixture_parameters, 1, 2)

    # The 3 leftover parameters are for the pen state
    pen_state = parameters[-1]

    mixture_weights = F.softmax(mixture_weights/T)  # Each weight must be in [0, 1] and all must sum to 1
    std_x = std_x.exp(std_x)*sqrtT  # Standard deviation must be positive
    std_y = std_y.exp(std_x)*sqrtT  # Standard deviation must be positive
    corr_xy = F.tanh(corr_xy)  # Correlation coefficient must be in [-1, 1]
    pen_state = F.softmax(pen_state/T)  # Each probability must be in [0, 1] and all must sum to 1


    return mixture_weights, mean_x, mean_y, std_x, std_y, corr_xy, pen_state

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # generates initial hidden and cell states from latent vector
        self.fc_in = nn.Linear(latent_dim,2*hidden_dim)

        # Fully connected layer for reducing dimensionality of hidden state before
        # being used for distribution parameters.
        self.fc_proj = nn.Linear(hidden_dim,output_dim)

        # Input has dimension latent_dim + 5 for latent vector and initial stroke
        self.lstm = nn.LSTM(input_dim,hidden_dim)

        # We can adjust dimensions of these
        self.hidden_cell = (torch.zeros(1,batch_size,hidden_dim),
                            torch.zeros(1,batch_size,hidden_dim))

    def forward(self, z): 
        # Batch samples from latent space distributions
        z = z.view(batch_size,latent_dim)
        
        # Obtain initial hidden and cell states by splitting result of fc_in along column axis
        self.hidden_cell = torch.split(F.tanh(self.fc_in(z)), 
                                       [hidden_dim, hidden_dim], 
                                       dim = 1)

        # Data for all strokes in output sequence
        strokes = torch.zeros(N_max + 1, batch_size, stroke_dim)
        strokes[:,0] = torch.tensor([0,0,1,0,0])
  
        # For each timestep, pass the batch of strokes through LSTM cell and compute 
        # the output.  Output of the previous timestep is used as input.
        for i in range(0,N_max):

            x = torch.cat((z,strokes[i,:,:]),dim = 1)

            out, self.hidden_cell = self.lstm(x,self.hidden_cell)

            # Sample from output distribution. If temperature parameter is small,
            # this becomes deterministic.
            params = distribution(self.fc_proj(out),T)
            strokes[i+1] = sample(params);
            
        return strokes[1:,:,:] #ignore first stroke
    