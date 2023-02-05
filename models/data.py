import os
import random
from glob import glob
import numpy as np
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, data_dir, split, length):
        if split not in ['train', 'validation', 'test']:
            raise ValueError('Invalid partition.')
        self.length = length
        self.filepaths = glob(os.path.join(self.data_dir, split, "**/*.npy"), recursive=True)
        print(f'[Loaded dataset] [{data_dir} - {split}]')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        array = np.load(self.filepaths[index])
        if array.shape[0] < self.length:
            return np.pad(array, (0, self.length - array.shape[0]), 'constant', constant_values=(0, 0))
        else:
            i = random.randint(0, array.shape[0] - self.length)
            return array[i:i + self.length]
