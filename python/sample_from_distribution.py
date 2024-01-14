# mu is the mean of the latent space distribution
# sigma is the standard deviation of the latent space distribution
def sample_latent_vector(mu, sigma_hat):
    # Convert sigma_hat to sigma using an exponential operation
    sigma = np.exp(sigma_hat / 2.0)
    # Sample epsilon from a standard normal distribution
    epsilon = np.random.randn(mu.shape[0])
    # Calculate the latent vector z
    z = mu + np.multiply(sigma, epsilon)
    return z
