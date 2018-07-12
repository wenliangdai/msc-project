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

class VOC_Loader(Loader):
    def __init__(self, mode, n_classes, transform=None, target_transform=None, img_size=512, ignore_index=255, do_transform=False):
        super(VOC_Loader, self).__init__(
            mode, 
            n_classes, 
            transform, 
            target_transform, 
            img_size, 
            ignore_index, 
            do_transform)

    def __getitem__(self, index):
        img = None
        mask = None

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if mask_path.split('.')[-1] == 'mat':
            mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        else:
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
        # 21 classes
        return np.asarray([
            [0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
            [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
            [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0,64,0], [128,64,0],
            [0,192,0], [128,192,0], [0,64,128]
        ])

    def preprocess(self, mode):
        assert mode in ['train', 'val', 'test']
        items = []
        sbd_path = get_data_path('sbd')
        sbd_img_path = os.path.join(sbd_path, 'dataset', 'img')
        sbd_mask_path = os.path.join(sbd_path, 'dataset', 'cls')
        voc_path = get_data_path('pascal')
        voc_test_path = get_data_path('pascal_test')
        voc_img_path = os.path.join(voc_path, 'JPEGImages')
        voc_mask_path = os.path.join(voc_path, 'SegmentationClass')
        
        # Train data = VOC_train + SBD_train + SBD_val
        if mode == 'train':
            sbd_data_list = [l.strip('\n') for l in open(os.path.join(
                sbd_path, 'dataset', 'trainval.txt')).readlines()]
            for it in sbd_data_list:
                item = (os.path.join(sbd_img_path, it + '.jpg'), os.path.join(sbd_mask_path, it + '.mat'))
                items.append(item)
            
            voc_data_list = [l.strip('\n') for l in open(os.path.join(
                voc_path, 'ImageSets', 'Segmentation', 'train.txt')).readlines()]
            for it in voc_data_list:
                item = (os.path.join(voc_img_path, it + '.jpg'), os.path.join(voc_mask_path, it + '.png'))
                items.append(item)
        # Val data = VOC_val
        elif mode == 'val':
            data_list = [l.strip('\n') for l in open(os.path.join(
                voc_path, 'ImageSets', 'Segmentation', 'val.txt')).readlines()]
            for it in data_list:
                item = (os.path.join(voc_img_path, it + '.jpg'), os.path.join(voc_mask_path, it + '.png'))
                items.append(item)
        # Test data = VOC_test
        else:
            img_path = os.path.join(voc_test_path, 'JPEGImages')
            data_list = [l.strip('\n') for l in open(os.path.join(
                voc_path, 'ImageSets', 'Segmentation', 'test.txt')).readlines()]
            for it in data_list:
                items.append((img_path, it))
        return items
