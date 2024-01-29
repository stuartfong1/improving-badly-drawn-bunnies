import torch

# Number of dimensions in latent vector
from params import latent_dim
def kl_loss(sigma_hat, mu):
    # torch.sum is added to sum over all the dimensions of the vectors
    return (-0.5 / latent_dim) * torch.sum(1 + sigma_hat - torch.square(mu) - torch.exp(sigma_hat))
