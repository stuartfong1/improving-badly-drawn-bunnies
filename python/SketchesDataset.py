import numpy as np

class SketchesDataset():
    def __init__(self, data_path, transform=None):
        # data_path is path to one of the npz files
        # transform is not currently functional
        self.data_path = data_path
        self.data_set = np.load(data_path, encoding='latin1', allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data_set['train'])

    def __getitem__(self, idx):
        sketch = self.data_set['train'][idx]
        if self.transform:
            sketch = self.transform(sketch)
        return sketch, sketch.shape[0]
    

if __name__ == "__main__":
    data = SketchesDataset(data_path="python/ambulance.npz")
    sketch, len = data.__getitem__(0)
    assert(sketch[len-1][2] == 1)
    print("Tests passed!")
            
