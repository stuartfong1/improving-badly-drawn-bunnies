import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def pen_reconstruction_loss(batch_size, N_max, input_pen_state, output):
    """
    Parameters:

        N_max (int) - Maximum sketch sequence length
        
        input_pen_state (batch_size,3) - Pen state data for a stroke

        output (batch_size, 3)- Generated pen state logit values.

    Returns:
        Reconstruction loss for pen state.
    """    

    return -1/(N_max*batch_size) * torch.sum(input_pen_state*torch.log(output.view(128,3)))


    
    