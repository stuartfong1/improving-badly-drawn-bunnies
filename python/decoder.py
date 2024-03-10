import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
from params import *
from hyper_lstm import HyperLSTM

input_dim = latent_dim + stroke_dim + Nclass # z | (x,y,p1,p2,p3) | (c1, c2, ... c_Nclass)
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
    mixture_weights = torch.reshape(mixture_weights,(batch_size,M)).contiguous() 
     
    # Index for each batch
    i = torch.searchsorted(mixture_weights.cumsum(0), torch.rand(batch_size, 1)).squeeze()
    
    # Sample from bivariate normal distribution i
    rand_x, rand_y = torch.randn(batch_size), torch.randn(batch_size)

    mean_x = torch.gather(mean_x.squeeze(), 0, i.unsqueeze(0)).squeeze()
    mean_y = torch.gather(mean_y.squeeze(), 0, i.unsqueeze(0)).squeeze()
    std_x = torch.gather(std_x.squeeze(), 0, i.unsqueeze(0)).squeeze()
    std_y = torch.gather(std_y.squeeze(), 0, i.unsqueeze(0)).squeeze()
    corr_xy = torch.gather(corr_xy.squeeze(), 0, i.unsqueeze(0)).squeeze()
    
    # Alternatively torch.distributions.multivariate_normal.MultivariateNormal?
    offset_x = mean_x + std_x * rand_x
    offset_y = mean_y + std_y * (corr_xy * offset_x + torch.sqrt(1 - corr_xy ** 2) * rand_y)
    return offset_x.unsqueeze(0), offset_y.unsqueeze(0)

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
    sqrtT = sqrt(T)
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

def sample(mixture_weights, mean_x, mean_y, std_x, std_y, corr_xy, pen_state):
    offset_x, offset_y = gaussian_mixture_model(mixture_weights, mean_x, mean_y, std_x, std_y, corr_xy)
    
    pen_state = pen_state.squeeze()
    pen_state = torch.searchsorted(pen_state.cumsum(1), torch.rand(batch_size, 1)).squeeze()
    next_point = torch.cat((offset_x, offset_y, torch.zeros(3, batch_size)))
    next_point = torch.cat((offset_x, offset_y, torch.eye(3)[pen_state].transpose(0, 1)))
    return next_point.transpose(0, 1)


#DECODER..

output_dim = 6*M + 3

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

       # input dimension is z | (x,y,p1,p2,p3) | (c1, c2, ... c_Nclass)
        if conditional:
            self.input_dim = latent_dim + stroke_dim + Nclass
        else:
            self.input_dim = latent_dim + stroke_dim

        # generates initial hidden and cell states from latent vector
        self.fc_in = nn.Linear(latent_dim,2*dec_hidden_dim + 2*dec_hyper_dim)

        # Fully connected layer for reducing dimensionality of hidden state before
        # being used for distribution parameters.
        self.fc_proj = nn.Linear(dec_hidden_dim,output_dim)

        # Input has dimension latent_dim + 5 for latent vector and initial stroke
        self.lstm = HyperLSTM(self.input_dim,dec_hidden_dim,dec_hyper_dim,feature_dim,1)

        self.hidden = torch.zeros(batch_size,dec_hidden_dim,device=device)
        self.cell = torch.zeros(batch_size,dec_hidden_dim,device=device)
        
        self.hidden_h = torch.zeros(batch_size,dec_hyper_dim,device=device)
        self.cell_h = torch.zeros(batch_size,dec_hyper_dim,device=device)
    
    def dec_forward1(self, z, strokes, classifier = None):
        params = [torch.zeros(Nmax,M,batch_size).to(device) for  _ in range(6)] + [torch.zeros(Nmax,batch_size,3).to(device)]
        # used when loop goes beyond input sketch length
        empty_stroke = torch.tensor([0,0,0,0,1]).to(torch.float32).to(device)
        # For each timestep, pass the batch of strokes through LSTM and compute
        # the output.  Output of the previous timestep is used as input.
        for i in range(1,Nmax+1):
            if classifier == None:
                input = strokes[i-1]
            else:
                input = torch.cat((strokes[i-1],classifier),dim = 1)
            
            x = torch.cat((input,z),dim = 1).view(1,batch_size,self.input_dim)
            
            (h,c,h_hat,c_hat) = self.lstm.cells[0](x.float().squeeze(), 
                                    self.hidden.squeeze(), 
                                    self.cell.squeeze(), 
                                    self.hidden_h.squeeze(), 
                                    self.cell_h.squeeze())
            self.hidden = h.unsqueeze(0)
            self.cell = c.unsqueeze(0)
            self.hidden_h = h_hat.unsqueeze(0)
            self.cell_h = c_hat.unsqueeze(0)    
            
            output = self.hidden[-1].unsqueeze(0)
            with torch.device(device):
                #params will be used for computing loss
                param_list = distribution(self.fc_proj(output))
                strokes[i] = sample(*param_list).to(device)
                for j in range(7):
                    params[j][i-1,:,:] = param_list[j].squeeze()
            
            stroke_mask = (strokes[i,:,4] == 1).to(device) # boolean mask set to true when p3 == 1
            
            strokes[i,stroke_mask.squeeze(),:] = empty_stroke
            
        return params
    
    def dec_forward2(self, z, strokes):
        self.hidden, self.cell, self.hidden_h, self.cell_h = torch.split(
            F.tanh(self.fc_in(z).unsqueeze(0)),
            [dec_hidden_dim, dec_hidden_dim, dec_hyper_dim, dec_hyper_dim],
            dim = 2)
        out, (self.hidden, self.cell,self.hidden_h,self.cell_h) = self.lstm(strokes.float(),(self.hidden.contiguous(), 
                                           self.cell.contiguous(),
                                           self.hidden_h.contiguous(),
                                           self.cell_h.contiguous()))
        params = distribution(self.fc_proj(out))
        return params
    
    def dec_forward3(self,z,strokes,classifier):
        params = self.dec_forward2(z,strokes)
        N_s = strokes.shape[0]
        new_strokes = torch.zeros(N_s, batch_size, stroke_dim,device=device)
        with torch.device(device):
            for i in range(N_s):
                new_strokes[i,:,:] = sample(*[j[i] for j in params])
        
        generated_strokes = [new_strokes[-1,:,:]]
        i = 1
        while True:
            if classifier == None:
                input = generated_strokes[i-1]
            else:
                input = torch.cat((generated_strokes[i-1],classifier),dim = 1)
            x = torch.cat((input,z),dim = 1).view(1,batch_size,self.input_dim)
            
            (h,c,h_hat,c_hat) = self.lstm.cells[0](x.float().squeeze(), 
                                    self.hidden.squeeze(), 
                                    self.cell.squeeze(), 
                                    self.hidden_h.squeeze(), 
                                    self.cell_h.squeeze())
            self.hidden = h.unsqueeze(0)
            self.cell = c.unsqueeze(0)
            self.hidden_h = h_hat.unsqueeze(0)
            self.cell_h = c_hat.unsqueeze(0)    
            
            output = self.hidden[-1].unsqueeze(0)
            with torch.device(device):
                #params will be used for computing loss
                params = distribution(self.fc_proj(output))
                generated_strokes.append(sample(*params).to(device))
            
            if generated_strokes[i][0,4].item() == 1 or i >= Nmax - N_s:
                break
            i = i + 1
            
        return torch.stack(generated_strokes[1:],dim = 0)
    
    
    def forward(self, z, strokes, mode = 'train',classifier = None):
        """
        Parameters:
            z - Tensor of size  (batch_size, latent_dim), with latent vector samples.

            stroke - Tensor of size (batch_size, stroke dim); previous stroke

        Returns:
            Tensor of size (N_max, batch_size, stroke dim), as the next stroke
        """
        
        if mode == 'generate':
            params = self.dec_forward1(z,strokes,classifier)
            return params
        
        elif mode == 'train':
            params = self.dec_forward2(z,strokes)
            return params
            
        elif mode == 'finish-sketch':
            params
        
        

if __name__ == "__main__":
    print("Running tests...\n")
    decoder = Decoder()
   
    print("Decoder successfully initialiazed ✅\n")

    z = torch.ones(batch_size,latent_dim) 
    strokes = torch.ones(24,batch_size,input_dim)
    params = decoder(z, strokes, False)
    
    print('Dimension test passed ✅\n')


