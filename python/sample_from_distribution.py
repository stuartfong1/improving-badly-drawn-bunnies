# mu is the mean of the latent space distribution
# sigma is the standard deviation of the latent space distribution
def sample_latent_vector(mu, sigma_hat):
    # Convert sigma_hat to sigma using an exponential operation
    sigma = torch.exp(sigma_hat / 2.0)
    # Sample epsilon from a standard normal distribution
    epsilon = torch.randn(sigma.shape)
    # Calculate the latent vector z
    z = mu + sigma * epsilon
    return z
