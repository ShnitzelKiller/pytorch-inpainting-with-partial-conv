import numpy as np
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter, ImageMorph
import os,sys
import random
import itertools
from scipy.misc import imread


class Dataset(torch.utils.data.Dataset):
    def __init__(self, flist, mask_flist, img_transform, mask_transform, train=True):
        super(Dataset, self).__init__()
        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        print('data: ', len(self.data))
        print('mask data: ' , len(self.mask_data))
        self.N_mask = len(self.mask_data)
        self.train = train
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        gt = Image.open(self.data[index])
        gt = self.img_transform(gt.convert('RGB'))
        if self.train:
            mask = Image.open(self.mask_data[random.randint(0, self.N_mask-1)])
        else:
            mask = Image.open(self.mask_data[index])
        mask = 1-self.mask_transform(mask.convert('RGB'))
        img = gt * mask
        return img, mask, gt

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []
