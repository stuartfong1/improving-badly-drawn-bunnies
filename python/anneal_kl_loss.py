# Function that does the annealing of the KL term
def anneal_kl_loss(num_training_steps, compute_reconstruction_loss, compute_kl_divergence, prediction, target, mu, sigma_hat):
    # Hyperparameters
    n_min = 0.01  # Starting Value from paper
    R = 0.9995  # R is a term close to but less than 1.
    KL_min = 0.1 # Value from paper (needs to be between 0.1 and 0.5)
    w_KL = 1.0  # Weight for the KL divergence part of the loss (can tune as needed, 1.0 is standard)

    # Initialize
    n_step = n_min
    total_loss = 0

    # Training loop
    for step in range(num_training_steps):
    
        # Compute reconstruction loss and KL divergence
        reconstruction_loss = compute_reconstruction_loss(prediction, target) # Might need to change depending on how this function is implemented
        kl_divergence = compute_kl_divergence(mu, sigma_hat)
    
        # Calculate n_step
        n_step = 1 - (1 - n_min) * R**step
    
        # Calculate the total weighted loss
        step_loss = reconstruction_loss + w_KL * n_step * max(kl_divergence, KL_min)
        total_loss += step_loss
        
    return total_loss
