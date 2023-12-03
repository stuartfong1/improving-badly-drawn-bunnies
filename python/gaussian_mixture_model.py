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
    i = np.random.choice(n_dist, p=mixture_weights)
    # Sample from bivariate normal distribution i
    rand_x, rand_y = torch.randn(2)
    
    # Sample from bivariate normal distribution
    # Alternatively torch.distributions.multivariate_normal.MultivariateNormal?
    offset_x = mean_x[i] + std_x[i] * rand_x
    offset_y = mean_y[i] + std_y[i] * (corr_xy[i] * offset_x + torch.sqrt(1 - corr_xy[i] ** 2) * rand_y)
    
    return offset_x, offset_y