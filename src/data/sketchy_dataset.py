import os
import torch.utils.data as data
import numpy as np
from ..options import Options

class GAN_DataGeneratorPaired_for_sketchy(data.Dataset):
    def __init__(self, photo_dir, sketch_dir, train_or_test,
                 transform_photo, transform_sketch):
        self.photo_dir = photo_dir
        self.sketch_dir = sketch_dir
        self.train_or_test = train_or_test
        self.transform_photo = transform_photo
        self.transform_sketch = transform_sketch
        self.train_list = os.listdir(self.photo_dir)
        self.sub_sketch_dirs = ['tx_000000000000', 'tx_000100000000', 'tx_000000000010', 'tx_000000000110', 'tx_000000001110',
                         'tx_000000001010']
        self.sub_photo_dirs = ['tx_000000000000', 'tx_000100000000']
        self.

    def __getitem__(self, item):

        if self.train_or_test == 'train':








