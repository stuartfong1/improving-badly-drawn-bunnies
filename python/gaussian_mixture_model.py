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
    mixture_weights = mixture_weights.squeeze().transpose(0, 1).contiguous()
    
    # Index for each batch
    i = torch.searchsorted(mixture_weights.cumsum(0), torch.rand(batch_size, 1)).squeeze()
    
    # Sample from bivariate normal distribution i
    rand_x, rand_y = torch.randn(batch_size), torch.randn(batch_size)
    
    mean_x = torch.take(mean_x, i)
    mean_y = torch.take(mean_y, i)
    std_x = torch.take(std_x, i)
    std_y = torch.take(std_y, i)
    corr_xy = torch.take(corr_xy, i)
    
    # Alternatively torch.distributions.multivariate_normal.MultivariateNormal?
    offset_x = mean_x + std_x * rand_x
    offset_y = mean_y + std_y * (corr_xy * offset_x + torch.sqrt(1 - corr_xy ** 2) * rand_y)
    return offset_x.unsqueeze(0), offset_y.unsqueeze(0)

def sample(mixture_weights, mean_x, mean_y, std_x, std_y, corr_xy, pen_state):
    offset_x, offset_y = gaussian_mixture_model(mixture_weights, mean_x, mean_y, std_x, std_y, corr_xy)
    
    pen_state = pen_state.squeeze()
    pen_state = torch.searchsorted(pen_state.cumsum(1), torch.rand(batch_size, 1)).squeeze()
    next_point = torch.cat((offset_x, offset_y, torch.zeros(3, batch_size)))
    next_point = torch.cat((offset_x, offset_y, torch.eye(3)[pen_state].transpose(0, 1)))
    
    return next_point.transpose(0, 1)