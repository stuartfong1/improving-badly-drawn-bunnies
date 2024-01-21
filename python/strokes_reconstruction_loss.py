def bivariate_normal_pdf(dx, dy, mu_x, mu_y, std_x, std_y, corr_xy):
    """
    Return N(dx, dy | mu_x, mu_y, std_x, std_y, corr_xy)
    """
    z_x = (dx - mu_x) / std_x
    z_y = (dy - mu_y) / std_y
    exponent = -(z_x ** 2 - 2 * corr_xy * z_x * z_y + z_y ** 2) / 2 * (1 - corr_xy ** 2)
    norm = 1 / (2 * np.pi * std_x * std_y * torch.sqrt(1-corr_xy ** 2))
    return norm * torch.exp(exponent)

def reconstruction_loss(dx, dy, pi, mu_x, mu_y, std_x, std_y, corr_xy, mask):
    """
    pi: The mixture probabilities
    mask: 1 if the point is not after the final stroke, 0 otherwise
    
    Returns the reconstruction loss for the strokes, L_s
    """
    pdf = bivariate_normal_pdf(dx, dy, mu_x, mu_y, std_x, std_y, corr_xy)
    return -(1/(Nmax * batch_size)) * torch.sum(mask * torch.log(torch.sum(pi * pdf, axis=0)))

if __name__ == '__main__':
    print("Testing strokes reconstruction loss...")
    dx = torch.randn((batch_size, 1))
    dy = torch.randn((batch_size, 1))
    mu_x = torch.randn((batch_size, M))
    mu_y = torch.randn((batch_size, M))
    std_x = torch.randn((batch_size, M)).exp()
    std_y = torch.randn((batch_size, M)).exp()
    corr_xy = F.tanh(torch.randn((batch_size, M)))
    pi = F.softmax(torch.randn(batch_size, M), dim=1)

    # Uncomment for testing, out is from the testing part of the decoder
    # mask = (out != torch.Tensor([0, 0, 0, 0, 1])).any(2).transpose(0, 1)
    mask = torch.randn((batch_size, M))

    print(reconstruction_loss(dx, dy, mu_x, mu_y, std_x, std_y, corr_xy, pi, mask))
    print("Success!")