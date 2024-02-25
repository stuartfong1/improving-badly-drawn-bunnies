from params import *


def bivariate_normal_pdf(dx, dy, mu_x, mu_y, std_x, std_y, corr_xy):
    """
    Return N(dx, dy | mu_x, mu_y, std_x, std_y, corr_xy)
    """
    z_x = (dx - mu_x) / std_x
    z_y = (dy - mu_y) / std_y
    exponent = -(z_x ** 2 - 2 * corr_xy * z_x * z_y + z_y ** 2) / (2 * (1 - corr_xy ** 2))
    norm = 2 * np.pi * std_x * std_y * torch.sqrt(1-corr_xy ** 2)
    return torch.exp(exponent) / norm


def offset_reconstruction_loss(dx, dy, pi, mu_x, mu_y, std_x, std_y, corr_xy, mask):
    """
    pi: The mixture probabilities
    mask: 1 if the point is not after the final stroke, 0 otherwise

    Returns the reconstruction loss for the strokes, L_s
    """
    pdf = bivariate_normal_pdf(dx, dy, mu_x, mu_y, std_x, std_y, corr_xy)
    
    return -torch.sum(mask * torch.log(1e-5 + torch.sum(pi*pdf,axis=0))) / (batch_size*Nmax) 


def pen_reconstruction_loss(input_pen_state, output):
    """
    Parameters:

        N_max (int) - Maximum sketch sequence length
        
        input_pen_state (batch_size,3) - Pen state data for a stroke

        output (batch_size, 3)- Generated pen state logit values.

    Returns:
        Reconstruction loss for pen state.
    """    

    return -torch.sum(input_pen_state*torch.log(1e-5+output)) / (batch_size*Nmax)

def kl_loss(sigma_hat, mu):
    # torch.sum is added to sum over all the dimensions of the vectors
    return (-0.5 / (latent_dim * batch_size)) * torch.sum(1 + sigma_hat - torch.square(mu) - torch.exp(sigma_hat))