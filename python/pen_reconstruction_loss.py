import torch
from params import batch_size

def pen_reconstruction_loss(N_max, input_pen_state, output):
    """
    Parameters:

        N_max (int) - Maximum sketch sequence length
        
        input_pen_state (batch_size,3) - Pen state data for a stroke

        output (batch_size, 3)- Generated pen state logit values.

    Returns:
        Reconstruction loss for pen state.
    """    

    return -1/(batch_size) * torch.sum(input_pen_state*torch.log(1e-5+output.view(batch_size,3)))


    
    