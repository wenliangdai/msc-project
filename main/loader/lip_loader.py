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

class LIP_LOADER(Loader):
    def __init__(self, mode, n_classes, transform=None, target_transform=None, img_size=512, ignore_index=255, do_transform=False):
        super(LIP_LOADER, self).__init__(
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
        # 20 classes
        return np.asarray([
            [0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
            [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
            [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0,64,0], [128,64,0],
            [0,192,0], [128,192,0]
        ])

    def preprocess(self, mode):
        assert mode in ['train', 'val', 'test']
        items = []
        data_path = get_data_path('lip')
        
        if mode == 'train':
            img_path = os.path.join(data_path, 'multi-person', 'Training', 'Images')
            mask_path = os.path.join(data_path, 'multi-person', 'Training', 'Category_ids')
            data_list = [l.strip('\n') for l in open(os.path.join(
                data_path, 'multi-person', 'Training', 'train_id.txt')).readlines()]
            for it in data_list:
                item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
                items.append(item)
        elif mode == 'val':
            img_path = os.path.join(data_path, 'multi-person', 'Validation', 'Images')
            mask_path = os.path.join(data_path, 'multi-person', 'Validation', 'Category_ids')
            data_list = [l.strip('\n') for l in open(os.path.join(
                data_path, 'multi-person', 'Validation', 'val_id.txt')).readlines()]
            for it in data_list:
                item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
                items.append(item)
        
        return items

    # mask_obj = sio.loadmat(mask_path)
    # person_class_index = None
    # for i, class_name in enumerate(mask_obj['anno']['objects'][0,0]['class'][0]):
    #     if class_name[0] == 'person':
    #         person_class_index = i

    # for i, part in enumerate(mask_obj['anno']['objects'][0,0]['parts'][0, person_class_index][0]):
    #     part_name = part[0][0]
    #     part_index = self.get_part_index(part_name)
    #     if i == 0:
    #         mask = part[1] * part_index
    #     else:
    #         mask = mask + part[1] * part_index
    # mask = Image.fromarray(mask.astype(np.uint8)).convert('P')


    # def get_part_index(self, part_name):
    #     '''
    #     coarse partition:
    #     head = 1
    #     torso = 2
    #     arm = 3
    #     leg = 4
    #     (background = 0)
    #     There are 24 finer parts in total
    #     '''
    #     if part_name in ['head','leye','reye','lear','rear','lebrow','rebrow','nose','mouth','hair']:
    #         return 1
    #     if part_name in ['torso','neck']:
    #         return 2
    #     if part_name in ['llarm','luarm','lhand','rlarm','ruarm','rhand']:
    #         return 3
    #     if part_name in ['llleg','luleg','lfoot','rlleg','ruleg','rfoot']:
    #         return 4
