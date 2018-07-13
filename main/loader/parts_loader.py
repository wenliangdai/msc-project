import collections
import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from PIL import Image, ImageMath
from torch.utils import data

from main import get_data_path
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

sys.path.append('/home/wenlidai/sunets-reproduce/main/loader')
from BaseLoader import Loader

class PASCAL_PARTS_LOADER(Loader):
    def __init__(self, mode, n_classes, transform=None, target_transform=None, img_size=512, ignore_index=255, do_transform=False):
        super(PASCAL_PARTS_LOADER, self).__init__(
            mode, 
            n_classes, 
            transform, 
            target_transform, 
            img_size, 
            ignore_index, 
            do_transform)

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('P')

        if self.do_transform:
            img, mask = self.further_transform(img, mask)
        else:
            img, mask = self.crop(img, mask)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def get_pascal_labels(self):
        # 7 classes (background, head, torso, upper/lower arms, upper/lower legs)
        return np.asarray([
            [0,0,0], [128,0,0], [0,128,0], [128,128,0], 
            [0,0,128], [128,0,128], [0,128,128]
        ])

    def preprocess(self, mode):
        assert mode in ['train', 'val', 'test']
        items = []
        data_path = get_data_path('pascalparts')
        
        if mode == 'train':
            img_path = os.path.join(data_path, 'JPEGImages')
            mask_path = os.path.join(data_path, 'ImageSets', 'Person', 'gt')
            data_list = [l.strip('\n') for l in open(os.path.join(
                data_path, 'ImageSets', 'Person', 'train.txt')).readlines()]
            for it in data_list:
                item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
                items.append(item)
        elif mode == 'val':
            img_path = os.path.join(data_path, 'JPEGImages')
            mask_path = os.path.join(data_path, 'ImageSets', 'Person', 'gt')
            data_list = [l.strip('\n') for l in open(os.path.join(
                data_path, 'ImageSets', 'Person', 'val.txt')).readlines()]
            for it in data_list:
                item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
                items.append(item)
        
        return items
