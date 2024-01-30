import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
from gaussian_mixture_model import sample
from pen_reconstruction_loss import pen_reconstruction_loss

from params import M, T, sqrtT, batch_size, dec_hidden_dim, latent_dim,stroke_dim,device

input_dim = latent_dim + stroke_dim # z | (x,y,p1,p2,p3)
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
    parameters = torch.split(decoder_output, 6, 2)

    # Chunk the parameters together, then stack them 
    # so that each column defines a distribution
    mixture_parameters = torch.stack(parameters[:-1],1)

    # Split mixture parameters into each parameter
    mixture_weights, mean_x, mean_y, std_x, std_y, corr_xy = torch.split(mixture_parameters, 1, 3)

    # The 3 leftover parameters are for the pen state
    pen_state = parameters[-1]

    mixture_weights = F.softmax(mixture_weights/T,dim=3)  # Each weight must be in [0, 1] and all must sum to 1
    std_x = torch.exp(std_x)*sqrtT  # Standard deviation must be positive
    std_y = torch.exp(std_y)*sqrtT  # Standard deviation must be positive
    corr_xy = F.tanh(corr_xy)  # Correlation coefficient must be in [-1, 1]
    pen_state = F.softmax(pen_state/T,dim=2)  # Each probability must be in [0, 1] and all must sum to 1

    return mixture_weights, mean_x, mean_y, std_x, std_y, corr_xy, pen_state

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # generates initial hidden and cell states from latent vector
        self.fc_in = nn.Linear(latent_dim,2*dec_hidden_dim)

        # Fully connected layer for reducing dimensionality of hidden state before
        # being used for distribution parameters.
        self.fc_proj = nn.Linear(dec_hidden_dim,output_dim)

        # Input has dimension latent_dim + 5 for latent vector and initial stroke
        self.lstm = nn.LSTM(input_dim,dec_hidden_dim)

        # We can adjust dimensions of these as we see fit
        self.hidden = torch.zeros(1,batch_size,dec_hidden_dim,device=device)
        self.cell = torch.zeros(1,batch_size,dec_hidden_dim,device=device)
    
    def forward(self, z, stroke):
        """
        Parameters:
            z - Tensor of size  (batch_size, latent_dim), with latent vector samples.

            stroke - Tensor of size (batch_size, stroke dim); previous stroke


        Returns:
            Tensor of size (N_max, batch_size, stroke dim), as the next stroke
        """
        x = torch.cat((z,stroke),dim = 1).view(1,batch_size,input_dim)

        out, self.hidden_cell = self.lstm(x,(self.hidden,self.cell))

        # Sample from output distribution. If temperature parameter is small,
        # this becomes deterministic.
        with torch.device(device):
            params = distribution(self.fc_proj(out))
            stroke_next = sample(*params)

        return stroke_next, params

if __name__ == "__main__":
    print("Running tests...\n")
    decoder = Decoder()
   
    print("Decoder successfully initialiazed ✅\n")

    z = torch.ones(batch_size,latent_dim) 
    #out, params = run_decoder(decoder,z)
    print('Dimension test passed ✅\n')
    print("Output (first sketch in batch):\n")
    #print(out[:,0])

