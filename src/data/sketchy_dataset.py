import os
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('..')
from src.data.utils import read_clslist_for_sketchy, get_file_list,default_image_loader
from src.options import Options
import numpy as np
from torchvision import transforms
import torch
from torch.autograd import Variable


class GAN_DataGeneratorPaired_for_sketchy(Dataset):
    def __init__(self, photo_dir, sketch_dir, train_or_test,
                 transform_photo=None, transform_sketch=None):
        self.photo_dir = photo_dir
        self.sketch_dir = sketch_dir
        self.loader = default_image_loader
        self.train_or_test = train_or_test
        self.transform_photo = transforms.Compose([transforms.ToTensor()])
        self.transform_sketch = transforms.Compose([transforms.ToTensor()])
        if self.train_or_test == 'train':
            self.cls_list = read_clslist_for_sketchy(train=True)
        else:
            self.cls_list = read_clslist_for_sketchy(train=False)
        self.sub_sketch_dirs = ['tx_000000000000', 'tx_000100000000', 'tx_000000000010',
                                'tx_000000000110', 'tx_000000001110', 'tx_000000001010']
        self.sub_photo_dirs = ['tx_000000000000', 'tx_000100000000']

        self.fnames_photos, self.cls_photos = get_file_list(os.path.join(self.photo_dir, self.sub_photo_dirs[0]), self.cls_list, 'images')

    def __getitem__(self, index):
        # random_sub_dir_sk = np.random.choice(self.sub_sketch_dirs, 1)[0]
        # random_sub_dir_im = np.random.choice(self.sub_photo_dirs, 1)[0]
        random_sub_dir_sk = self.sub_sketch_dirs[0]
        random_sub_dir_im = self.sub_photo_dirs[0]
        fname_photo = os.path.join(self.photo_dir, random_sub_dir_im, self.cls_photos[index],
                                   self.fnames_photos[index])
        fname_sketch_list = os.listdir(os.path.join(self.sketch_dir, random_sub_dir_sk,
                                                    self.cls_photos[index]))
        fname_ske = np.random.choice([fname for fname in fname_sketch_list if self.fnames_photos[index].split('.')[0] in fname], 1)[0]
        fname_sketch = os.path.join(self.sketch_dir, random_sub_dir_sk, self.cls_photos[index],
                                    fname_ske)

        return self.transform_photo(self.loader(fname_photo, True)), self.transform_photo(self.loader(fname_sketch, False))





    def __len__(self):
        return len(self.fnames_photos)







if __name__ == '__main__':
    args = Options().parse()
    testdemo = GAN_DataGeneratorPaired_for_sketchy(args.photo_dir, args.sketch_dir, 'train')
    d = DataLoader(dataset=testdemo, batch_size=32, shuffle=True, num_workers=2)
    for epoch in range(2):
        for i, data in enumerate(d):
            img, sketch = data
            img = Variable(img)
            sketch = Variable(sketch)

            print(epoch, i, img.data.size(), sketch.data.size())