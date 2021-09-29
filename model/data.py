import os
import numpy as np
from pathlib import Path
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, split, style, representation, length):
        if split not in ['train', 'validation', 'test']:
            raise ValueError('Invalid partition.')
        self.split = split
        self.path = f'./dataset/{style}/{representation}/{length}'
        self.len = sum([len(files) for r, d, files in os.walk(os.path.join(self.path, split))])
        print(f'Loading dataset: {style} - {split} - {length}\n\tSize: {self.len}')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        array = np.load(f'{self.path}/{self.split}/{index // 1000}/{index}.npy')
        return array
