import numpy as np
import torch
from torch.optim import Adam

from autoencoder import VAE
from params import device, T
from data_processing import load_weights, encode_dataset1
from display_data import display

# Normalize data
# We need the standard deviation of each class
classes = ['apple', 'flower', 'cactus', 'carrot']
cls = 0  # CHANGE THIS TO THE INDEX OF THE CLASS YOU WANT TO TEST
data_stds = {'apple': 30.20803366869141, 'flower': 42.27573114632625, 'cactus': 29.853817345732597, 'carrot': 42.56584982524036}
data_std = data_stds[classes[cls]]
Nmax = 200

def normalize(data):
    data = data.astype(np.float32)
    data[:,0:2] = data[:,0:2].astype(np.float32)/data_std
    return data

def add_class(sketch):
    sketch_class = np.full((len(sketch), 1), cls)
    sketch = np.concatenate([sketch, sketch_class], 1)
    return sketch

def make_image(image):
    length = len(image)
    new_image = np.zeros((Nmax, 4))
    new_image[:len(image), :] = image[:len(image), :] # copy over values

    encoded_strokes = np.stack(encode_dataset1(np.array([new_image]),[length]), 1) # don't forget to stack input along dim = 1
    batch = torch.from_numpy(encoded_strokes.astype(float))
    return batch, torch.tensor(length)

def predict(model, batch, lengths):
    batch = batch.squeeze(2).transpose(0, 1).to(device)
    with torch.no_grad():
        mean, logvar = model.encoder(batch)
        z = mean + torch.exp(logvar/2)*torch.randn(2, 128, device = device) * T
        c = batch[0,:,5:].clone()
        return model.run_decoder_generate(None, None, z, c, compute_loss = False)[0]

def run_model(my_array):
    my_array = normalize(my_array)
    my_array = add_class(my_array)
    batch, length = make_image(my_array)
    batch = batch.unsqueeze(0)
    batch = torch.cat([batch, batch], 0)
    length = torch.Tensor([length.item(), length.item()])

    model = VAE().to(device)
    optimizer = Adam(model.parameters()) 
    load_weights(model,optimizer,"model/final/remote/fruit.pt") 

    model.generate = True

    output = predict(model, batch, length)[:, 0, :3]
    return output


