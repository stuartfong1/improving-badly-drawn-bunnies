import numpy as np

class SketchesDataset():
    def __init__(self, data_path, transform=None):
        # data_path is path to one of the npz files
        # transform is not currently used
        self.data_path = data_path
        self.data_set = np.load(data_path, encoding='latin1', allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.sketchs_file_path)

    def __getitem__(self, idx):
        sketch = self.data_set['train'][idx]
        if self.transform:
            sketch = self.transform(sketch)
        return sketch