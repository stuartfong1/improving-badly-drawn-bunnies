import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
#from gaussian_mixture_model import sample

M = 10 # number of normal distributions for output 
T = 0.5 # temperature parameter
sqrtT = sqrt(T) # 

batch_size = 128
N_max = 10 # maximum number of strokes for sketch in dataset
hidden_dim = 2048 # dimension of cell and hidden states
latent_dim = 128 
stroke_dim = 5
input_dim = latent_dim + stroke_dim # z | (x,y,p1,p2,p3)

output_dim = 6*M + 3

def gaussian_mixture_model(mixture_weights, mean_x, mean_y, std_x, std_y, corr_xy):
    """
    Input:
        mixture_weights: Mixture weights (probability of a point being in distribution i)
        mean_x: The x-values of the means of each distribution
        mean_y: The y-values of the means of each distribution
        std_x: The standard deviations of the x-values
        std_y: The standard deviations of the y-values
        corr_xy: The correlation coefficients of the x and y values
        
    Return: 
        The sampled x and y offsets
    """
    # Choose which distribution to sample from
    mixture_weights = mixture_weights.squeeze().transpose(0, 1).contiguous()
    
    # Index for each batch
    i = torch.searchsorted(mixture_weights.cumsum(0), torch.rand(batch_size, 1)).squeeze()
    
    # Sample from bivariate normal distribution i
    rand_x, rand_y = torch.randn(batch_size), torch.randn(batch_size)
    
    mean_x = torch.take(mean_x, i)
    mean_y = torch.take(mean_y, i)
    std_x = torch.take(std_x, i)
    std_y = torch.take(std_y, i)
    corr_xy = torch.take(corr_xy, i)
    
    # Alternatively torch.distributions.multivariate_normal.MultivariateNormal?
    offset_x = mean_x + std_x * rand_x
    offset_y = mean_y + std_y * (corr_xy * offset_x + torch.sqrt(1 - corr_xy ** 2) * rand_y)
    return offset_x.unsqueeze(0), offset_y.unsqueeze(0)

def sample(mixture_weights, mean_x, mean_y, std_x, std_y, corr_xy, pen_state):
    offset_x, offset_y = gaussian_mixture_model(mixture_weights, mean_x, mean_y, std_x, std_y, corr_xy)
    
    pen_state = pen_state.squeeze()
    pen_state = torch.searchsorted(pen_state.cumsum(1), torch.rand(batch_size, 1)).squeeze()
    next_point = torch.cat((offset_x, offset_y, torch.zeros(3, batch_size)))
    next_point = torch.cat((offset_x, offset_y, torch.eye(3)[pen_state].transpose(0, 1)))
    
    return next_point.transpose(0, 1)

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
        self.fc_in = nn.Linear(latent_dim,2*hidden_dim)

        # Fully connected layer for reducing dimensionality of hidden state before
        # being used for distribution parameters.
        self.fc_proj = nn.Linear(hidden_dim,output_dim)

        # Input has dimension latent_dim + 5 for latent vector and initial stroke
        self.lstm = nn.LSTM(input_dim,hidden_dim)

        # We can adjust dimensions of these as we see fit
        self.hidden_cell = (torch.zeros(1,batch_size,hidden_dim),
                            torch.zeros(1,batch_size,hidden_dim))
    
    def forward(self, z, N_s): 
        """
        Parameters:
            z - Tensor of size  (batch_size, latent_dim), with latent vector samples.

            N_s - Tensor of size (batch_size), with lengths of each sketch sequence.
            When generating new sketches, this is not needed.

        Returns:
            Tensor of size (N_max, batch_size, stroke dim), with generated sketch sequences.
        """
        # Batch samples from latent space distributions
        z = z.view(batch_size,latent_dim)
        N_s = N_s.view(batch_size)
        
        # Obtain initial hidden and cell states by splitting result of fc_in along column axis
        self.hidden_cell = torch.split(F.tanh(self.fc_in(z).view(1,latent_dim,2*hidden_dim)), 
                                       [hidden_dim, hidden_dim], 
                                       dim = 2)

        # Data for all strokes in output sequence
        strokes = torch.zeros(N_max + 1, batch_size, stroke_dim)
        strokes[0,:] = torch.tensor([0,0,1,0,0])
  
        # For each timestep, pass the batch of strokes through LSTM cell and compute 
        # the output.  Output of the previous timestep is used as input.
        for i in range(0,N_max):

            x = torch.cat((z,strokes[i,:,:]),dim = 1).view(1,batch_size,input_dim)

            out, self.hidden_cell = self.lstm(x,self.hidden_cell)

            # Sample from output distribution. If temperature parameter is small,
            # this becomes deterministic.
            params = distribution(self.fc_proj(out))
            strokes[i+1] = sample(batch_size, *params)

            #for strokes in generated sequence past sequence length, set to [0,0,0,0,1]
            mask = (i > N_s)
            empty_stroke = torch.tensor([0,0,0,0,1],dtype=torch.float32)
            strokes[i+1,mask] = empty_stroke
            
        return strokes[1:,:,:] #ignore first stroke
    
def dim_test(decoder):
    z = torch.ones(batch_size,latent_dim) 
    out = decoder.forward(z,torch.tensor(N_max/2).repeat(batch_size))
    print('Dimension test passed ✅\n')
    print("Output (first sketch in batch):\n")
    print(out[:,0])

if __name__ == "__main__":
    print("Running tests...\n")
    decoder = Decoder()
    print("Decoder successfully initialiazed ✅\n")
    dim_test(decoder)
