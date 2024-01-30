
import torch
from torch import nn
import torch.nn.functional as F

from decoder_lstm import Decoder
from encoder_lstm import Encoder

from pen_reconstruction_loss import pen_reconstruction_loss
from offset_reconstruction_loss import offset_reconstruction_loss

from params import enc_hidden_dim,batch_size,stroke_dim,latent_dim,data,Nmax,device
from params import M, T, sqrtT, batch_size, Nmax,dec_hidden_dim, latent_dim,stroke_dim,device


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x, N_s = torch.full((batch_size,1),Nmax)):
        mean, logvar = self.encoder(x)
        
        sample = torch.randn(batch_size, latent_dim,device = device)
        std = torch.exp(logvar/2) # logvar / 2 should be a float
        z = mean + std*sample

        if torch.isnan(z)[0, 0]: # debug code
             print() # put a breakpoint here - will let you view variables in the case gradient becomes NaN

        # Code from old run_decoder method

        # Data for all strokes in output sequence
        strokes = torch.zeros(Nmax + 1, batch_size, 5,device=device)
        strokes[0,:] = torch.tensor([0,0,1,0,0])

        # Batch of samples from latent space distributions
        z = z.view(batch_size,latent_dim)

        # Obtain initial hidden and cell states by splitting result of fc_in along column axis
        self.decoder.hidden_cell = torch.split(
            F.tanh(self.decoder.fc_in(z).view(1,batch_size,2*dec_hidden_dim)),
            [dec_hidden_dim, dec_hidden_dim],
            dim = 2)
        
        pen_loss = 0
        offset_loss = 0

        mask, dx, dy, p = make_target(x, N_s)

        # used when loop goes beyond input sketch length
        empty_stroke = torch.tensor([0,0,0,0,1],dtype=torch.float32,device=device)
        
        # For each timestep, pass the batch of strokes through LSTM and compute
        # the output.  Output of the previous timestep is used as input.
        for i in range(1,Nmax):

            #params will be used for computing loss
            strokes[i], params = self.decoder(z,strokes[i-1])

            input_stroke = x[i]

            # calculate loss at each timestep

            #params[6] is pen_state, input_stroke is the input data
            pen_loss += pen_reconstruction_loss(input_stroke[:,2:],params[6])
            # size of dx and mu_x (and dy and mu_y) do not match.

            offset_params = [params[i].view(batch_size,M) for i in range(6)]

            offset_loss += offset_reconstruction_loss(
                dx[i].view(batch_size,1),
                dy[i].view(batch_size,1),
                *offset_params,
                mask[i].view(batch_size,1)
            )
            #for strokes in generated sequence past sequence length, set to [0,0,0,0,1]
            stroke_mask = (i > N_s) # boolean mask set to false when i is larger than sketch size

            
            strokes[i,stroke_mask.squeeze(),:] = empty_stroke
    
        return strokes[1:], offset_loss, pen_loss, mean, logvar

def make_target(batch, lengths):
    with torch.device(device):
        mask = torch.zeros((Nmax + 1, batch.size()[1]))
        for index, num_strokes in enumerate(lengths):
            mask[:num_strokes, index] = 1

        dx = batch[:, :, 0]
        dy = batch[:, :, 1]
        # copy + append together pen state values
        p = torch.stack([batch.data[:, :, 2], batch.data[:, :, 3], batch.data[:, :, 4]], 2)

        return mask, dx, dy, p