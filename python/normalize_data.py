# import numpy as np
# 
# data_path = '/kaggle/input/tinyquickdraw/sketches/sketches/whale.npz'
# 
# dataset = np.load(data_path, encoding='latin1', allow_pickle=True)
# data = dataset["train"]
# 
# def normalize_data(): 
#     
#     total_length = 0
#     
#     for element in data:
#         total_length += (len(element))
#     
#     coordinate_list = np.empty((total_length,2)) 
#     
#     i = 0
#     
#     for element in data:
#         coordinate_list[i:i+len(element),:] = element[:,0:2] 
#         i+=len(element)
# 
#     data_std = np.std(coordinate_list) 
#     
#     for i, element in enumerate(data):
#         data[i] = data[i].astype(np.float32)
#         data[i][:,0:2] = element[:,0:2].astype(np.float32)/data_std

from params import data, np

def normalize_data():

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


