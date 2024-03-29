import os
import random

import numpy as np
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, split, style, representation, length):
        if split not in ['train', 'validation', 'test']:
            raise ValueError('Invalid partition.')
        self.split = split
        self.length = length
        self.path = f'./dataset/{style}/{representation}/{length}'
        self.len = sum([len(files) for r, d, files in os.walk(os.path.join(self.path, split))])
        print(f'Loading dataset: {style} - {split} - {length}\n\tSize: {self.len}')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        array = np.load(f'{self.path}/{self.split}/{index // 1000}/{index}.npy')
        if array.shape[0] <= self.length:
            return np.pad(array, (0, self.length - array.shape[0]), 'constant', constant_values=(0, 0))
        else:
            i = random.randint(0, array.shape[0] - self.length)
            return array[i:i + self.length]
