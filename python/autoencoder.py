
import torch
from torch import nn
import torch.nn.functional as F
from loss import *
from params import *

from decoder import Decoder, sample, distribution, input_dim
from encoder import Encoder

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.generate = False
    
    def run_decoder_generate(self,batch,lengths,z,classifier,compute_loss = True):
        '''
        Generates sketches conditioned only on latent vector and a classification vector
        
        Parameters:
            classifer (1,Nclass) - One-hot encoded (usually) vector to indicate sketch type.
            
            compute_loss - Set this to False if loss is not needed.
        '''
       
        # Obtain initial hidden and cell states by splitting result of fc_in along column axis
        self.decoder.hidden, self.decoder.cell, self.decoder.hidden_h, self.decoder.cell_h = torch.split(
                F.tanh(self.decoder.fc_in(z).unsqueeze(0)),
                [dec_hidden_dim, dec_hidden_dim, dec_hyper_dim, dec_hyper_dim],
                dim = 2)
        
        # Data for all strokes in output sequence
        strokes = torch.zeros(Nmax + 1, batch_size, stroke_dim,device=device)
        strokes[0,:] = torch.tensor([0,0,1,0,0]).to(device)
        
        
        params = self.decoder(z,strokes,'generate',classifier=classifier)
                   
        #mask, dx, dy, p = make_target(batch, lengths)
        #offset_params = [params[i].squeeze().transpose(0, 1) for i in range(6)]
        #l_p = pen_reconstruction_loss(p, params[6]) 
        #l_s = offset_reconstruction_loss(dx, dy, *offset_params, mask[:-1])
                
        return strokes[1:], params

    def run_decoder_train(self,batch,lengths,z):

        empty = torch.zeros(stroke_dim + Nclass)
        empty[2] = 1
        start_stroke = torch.stack([empty] * batch_size).unsqueeze(0).to(device)
        strokes = torch.cat([start_stroke, batch[:-1]], 0)
        zs = torch.stack([z] * (Nmax))
        #IMPORTANT: Must always ensure that this is concatenated in the same order as the
        #generation mode in the decoder. 
        strokes = torch.cat([strokes,zs], 2)

        params = self.decoder(z, strokes,'train')
        
        output = torch.zeros(Nmax, batch_size, 5,device=device)
        with torch.device(device):
            empty_stroke = torch.tensor([0,0,0,0,1]).to(torch.float32).to(device)
            for i in range(Nmax):
                output[i] = sample(*[j[i] for j in params])
                stroke_mask = (i > lengths).squeeze()
                output[i,stroke_mask,:] = empty_stroke
                           

        return output, params
    
    def complete_sketch(self,input):
        mean, logvar = self.encoder(input)
        z = mean + torch.exp(logvar/2)*torch.randn(batch_size, latent_dim, device = device)
        empty = torch.zeros(stroke_dim + Nclass)
        empty[2] = 1
        start_stroke = torch.stack([empty] * batch_size).unsqueeze(0).to(device)
        strokes = torch.cat([start_stroke, input[:-1]], 0)
        zs = torch.stack([z] * (input.shape[0]))
        #IMPORTANT: Must always ensure that this is concatenated in the same order as the
        #generation mode in the decoder. 
        strokes = torch.cat([strokes,zs], 2)
        
        output = self.decoder.dec_forward3(z,strokes,classifier = input[0,:,5:].clone())
        
        return torch.cat((input[:,:,:5],output),dim = 0)
    
    def forward(self, batch, lengths, anneal_loss=True, step=0): 
        batch = batch.to(device)
        lengths = lengths.to(device) 
        
        mean, logvar = self.encoder(batch)
        
        # sample latent vector from encoder output
        random_sample = torch.randn(batch_size, latent_dim, device = device)
        std = torch.exp(logvar/2) # logvar / 2 should be a float
        z = mean + std*random_sample
        
        if self.generate:
            classifier = batch[0,:,5:].clone()
            output, params = self.run_decoder_generate(batch,lengths,z,classifier)
        else:
            output, params = self.run_decoder_train(batch,lengths,z)
            
        mask, dx, dy, p = make_target(batch, lengths)
        
        #compute loss
        offset_params = [params[i].squeeze().transpose(0, 1) for i in range(6)]
        l_p = pen_reconstruction_loss(p, params[6]) 
        l_s = offset_reconstruction_loss(dx, dy, *offset_params, mask[:-1])

        l_r = l_p + l_s
        l_kl = kl_loss(mean, logvar)
        
        if anneal_loss:
            # Calculate n_step
            n_step = 1 - (1 - n_min) * R**step

            # Calculate the total weighted loss
            loss = l_r + w_kl * n_step * max(l_kl, KL_min)
        else:
            loss = l_r + w_kl * l_kl
    
        return output, loss, l_kl, l_s, l_p
    
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