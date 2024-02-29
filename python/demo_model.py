import numpy as np
import torch

from params import device, T
from data_processing import encode_dataset1
from display_data import display

# Normalize data
# We need the standard deviation of each class

Nmax = 200

def normalize(data, cls):
    data_stds = {
        'Apple': 30.20803366869141, 
        'Flower': 42.27573114632625, 
        'Cactus': 29.853817345732597, 
        'Carrot': 42.56584982524036
    }
    data_std = data_stds[cls]
    data = data.astype(np.float32)
    data[:,0:2] = data[:,0:2].astype(np.float32)/data_std
    return data

def add_class(sketch, cls):
    classes = ['Apple', 'Flower', 'Cactus', 'Carrot']
    sketch_class = np.full((len(sketch), 1), classes.index(cls))
    sketch = np.concatenate([sketch, sketch_class], 1)
    return sketch

def make_image(image):
    length = len(image)
    new_image = np.zeros((Nmax, 4))
    new_image[:len(image), :] = image[:len(image), :] # copy over values

    encoded_strokes = np.stack(encode_dataset1(np.array([new_image]),[length]), 1) # don't forget to stack input along dim = 1
    batch = torch.from_numpy(encoded_strokes.astype(float))
    return batch, torch.tensor(length)

def process(model, batch, length):
    batch = batch.squeeze(2).transpose(0, 1).to(device)
    with torch.no_grad():
        mean, logvar = model.encoder(batch)
        z = mean + torch.exp(logvar/2)*torch.randn(2, 128, device = device) * T
        c = batch[0,:,5:].clone()
        return model.run_decoder_generate(None, None, z, c, compute_loss = False)[0]

def complete(model, batch, length):
    batch = batch.squeeze(2).transpose(0, 1).to(device)
    batch = batch[:int(length[0].item()), :, :]
    
    # Uncomment this for the last stroke to be pen down
    # batch[-1, :, 2] = 0
    # batch[-1, :, 3] = 1

    with torch.no_grad():
        mean, logvar = model.encoder(batch)
        z = mean + torch.exp(logvar/2)*torch.randn(2, 128, device = device) * T
        empty = torch.zeros(5 + 4)  # stroke_dim + Nclass
        empty[2] = 1
        start_stroke = torch.stack([empty] * 2).unsqueeze(0).to(device)
        strokes = torch.cat([start_stroke, batch[:-1]], 0)
        zs = torch.stack([z] * (batch.shape[0]))
        #IMPORTANT: Must always ensure that this is concatenated in the same order as the
        #generation mode in the decoder. 
        strokes = torch.cat([strokes,zs], 2)
        
        output = model.decoder.dec_forward3(z,strokes,classifier = batch[0,:,5:].clone())
        
        return torch.cat((batch[:,:,:5],output),dim = 0)

def run_model(model, my_array, cls, mode='process'):
    my_array = normalize(my_array, cls)
    my_array = add_class(my_array, cls)
    batch, length = make_image(my_array)
    batch = batch.unsqueeze(0)
    batch = torch.cat([batch, batch], 0)
    length = torch.Tensor([length.item(), length.item()])

    if mode == 'process':
        output = process(model, batch, length)[:, 0, :3]
    elif mode == 'complete':
        output = complete(model, batch, length)[:, 0, :3]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    return output


