import os
import numpy as np
from pathlib import Path
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, split, style, representation, length):
        self.split = split
        self.path = Path.cwd() / "model/dataset/{}/{}/{}".format(style, representation, length)
        self.len = sum([len(files) for r, d, files in os.walk(os.path.join(self.path, split))])
        if split not in ['train', 'validation', 'test']:
            raise ValueError('Invalid partition.')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        array = np.load((self.path / self.split / "{}/{}.npy".format(index // 1000, index)).as_posix())
        return array
