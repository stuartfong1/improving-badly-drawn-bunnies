from params import *
from data_processing import *
from autoencoder import VAE
from torch.utils.data import DataLoader
from torch.optim import Adam
from training import run_tests

train_dataset = SketchesDataset(
        datasets=datasets,
        mode='train',
        transform=make_image
    )
test_dataset = SketchesDataset(
        datasets=datasets,
        mode='test',
        transform=make_image
    )
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

# Initialize model and weights
model = VAE().to(device)
optimizer = Adam(model.parameters(), lr = lr) 

load_weights(model,optimizer,"model/final/remote/fruit.pt")   

T = 0.2
model.generate = True
run_tests(test_dataloader,model,5)

# INTERPOLATION EXAMPLE
def latent_lerp(model, S1, S2, nstep):
    mean, logvar = model.encoder(S1)
    z1 = mean + torch.exp(logvar/2)*torch.randn(batch_size, latent_dim, device = device)
    mean, logvar = model.encoder(S2)
    z2 = mean + torch.exp(logvar/2)*torch.randn(batch_size, latent_dim, device = device)
    
    c1 = S1[0,:,5:].clone()
    c2 = S2[0,:,5:].clone()
    
    step_size = 1/(nstep - 1)
    
    z_interp = [torch.lerp(z1, z2, step_size*i) for i in range(nstep)]
    c_interp = [torch.lerp(c1, c2, step_size*i) for i in range(nstep)]
    
    return [model.run_decoder_generate(None, None, z, c, compute_loss = False)[0] for z, c in zip(z_interp, c_interp)]


def get_sketch(dataloader):
    return next(iter(dataloader))[0].squeeze(2).transpose(0, 1).to(device)

T = 0.2
S1 = get_sketch(test_dataloader)
S2 = get_sketch(test_dataloader)
display_encoded_image(S1[:, 0, :])
display_encoded_image(S2[:, 0, :])

nstep = 10

interp = latent_lerp(model,S1,S2,nstep)

for i in range(nstep):
    display_encoded_image(interp[i][:, 0, :])