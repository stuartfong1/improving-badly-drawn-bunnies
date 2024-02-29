import numpy as np                                                                                                                                  
import matplotlib.pyplot as plt
import math
import torch
from torch import nn
from torch.utils.data import dataloader
import os
from datetime import datetime
from params import *

def encode_sketch(sketch,length,N,do_offset = True): 
    ''' 
    One-hot encode pen state by adding additional columns for pen up and end of stroke.

    Parameters: 
        sketch (ndarray): n*4 array with format (x,y,p1,class), representing sketch data

    Returns: 
        ndarray: n*5 array with format (x,y,p1,p2,p3, c1,...,c_N), where p2 = 1-p1 and p3 is 1 at 
        end of the sketch, 0 otherwise.
    '''
    if do_offset:
        sketch[:,:2] *= np.random.uniform(0.9, 1.1, size=(Nmax, 2))


    shape = sketch.shape
    pen_up = (np.ones(shape[0]) - sketch[:,2]).reshape(shape[0],1)
    category = np.zeros((shape[0],N))
    
    category[:,int(sketch[0][-1])] = 1
    
    end_stroke = np.zeros((shape[0],1))
    end_stroke[length:] = 1 
    pen_up[length:] = 0
    sketch[:,2][length:] = 0
    sketch[-1][2] = 0
    
    return np.concatenate((sketch[:,:-1],pen_up,end_stroke,category),axis=1)

def encode_dataset1(data,lengths,do_offset = True):
    """
    Encode pen states by creating a new array of sketch data.
    
    Parameters:
        data (iterable): object containing data for each sketch
        
    Returns:
        ndarray: object array containing encoded data for each sketch
    """
    # new_data = np.empty(data.size,dtype=object)
    new_data = np.empty((data.shape[0], data.shape[1], stroke_dim+Nclass), dtype=object)

    for i, sketch in enumerate(data):
        new_data[i] = encode_sketch(sketch,lengths[i],Nclass,do_offset)

    return new_data


def normalize_data(data):

    total_length = 0
    

    for element in data:
        total_length += (len(element))


    coordinate_list = np.empty((total_length,2))

    i = 0

    for element in data:
        coordinate_list[i:i+len(element),:] = element[:,0:2]
        i+=len(element)

    data_std = np.std(coordinate_list)

    for i, element in enumerate(data):
        data[i] = data[i].astype(np.float32)
        data[i][:,0:2] = element[:,0:2].astype(np.float32)/data_std
    
    return data

def display_encoded_image(image):
    """
    For some image tensor, draw the image using matplotlib.

    Parameters:
        - image: some [n*5] tensor representing a sketch.
    Returns:
        - none
    """
    #Xplot and Yplot are array of points that will be plotted
    Xplot = [0]
    Yplot = [0]
    #Keeps track of the current point that is being drawn
    xpos = 0
    ypos = 0
    #For loop to go through data and plot points
    i=0
    for i in range(len(image)):
        xpos += float(image[i,0])
        ypos += float(image[i,1])
        Xplot.append(-xpos)
        Yplot.append(-ypos)
        if image[i,3] == 0:
            plt.plot(Xplot, Yplot,color='black')
            Xplot.clear()
            Yplot.clear()
        # elif image[i, 4] == 1:
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.show()


class SketchesDataset():
    def __init__(self, datasets, mode, transform=None):
        """
        data_path: The path to the data
        mode: Either 'train' or 'test'
        """
        self.transform = transform
        self.mode = mode
        self.data_set = []
        for i, dataset in enumerate(datasets):
            dataset = dataset[mode]
            dataset = normalize_data(dataset)
            for j, sketch in enumerate(dataset):
                sketch_class = np.full((len(sketch), 1), i)
                sketch = np.concatenate([sketch, sketch_class], 1)
                self.data_set.append(sketch)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        sketch = self.data_set[idx]
        if self.transform:
            sketch, length = self.transform(sketch)
        return sketch, length
    
def make_image(image):
    length = len(image)
    new_image = np.zeros((Nmax, 4))
    new_image[:len(image), :] = image[:len(image), :] # copy over values

    encoded_strokes = np.stack(encode_dataset1(np.array([new_image]),[length]), 1) # don't forget to stack input along dim = 1
    batch = torch.from_numpy(encoded_strokes.astype(float))
    return batch, torch.tensor(length)
    

def save_weights(model,optimizer,dir,filename = None, max_count = 2**31 - 1):
    try:
        os.makedirs(dir)
    except:
        pass
    
    if filename == None:
        filename = f"{str(datetime.now())}.pt"
        
    files = os.listdir(dir)
    
    while len(files) >= max_count:
        os.remove(dir + files.pop(0))
        
    torch.save({
            "model": model.state_dict(),
            "opt": optimizer.state_dict()
        }, dir + filename)
    
def load_weights(model,optimizer,dir,file_index = None):
    if not file_index == None:
        files = os.listdir(dir)
        dir = dir + files[file_index]
        
    loaded_state = torch.load(dir, map_location=device)
    model.load_state_dict(loaded_state['model'])
    optimizer.load_state_dict(loaded_state['opt'])
    