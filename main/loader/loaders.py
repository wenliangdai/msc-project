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

sys.path.append('/home/wenlidai/msc-project/main/loader')
from BaseLoader import Loader

class SEMSEG_LOADER(Loader):
    def __init__(self, mode, n_classes, transform=None, target_transform=None, img_size=512, ignore_index=255, do_transform=False, portion=1):
        super(SEMSEG_LOADER, self).__init__(
            mode, 
            n_classes, 
            transform, 
            target_transform, 
            img_size, 
            ignore_index, 
            do_transform, portion=portion)

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
        '''
        21 classes:
            - Person: person
            - Animal: bird, cat, cow, dog, horse, sheep
            - Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
            - Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor
        '''
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
            # SBD dataset contains some of the voc_val samples, so we have to remove them
            voc_val_data_list = [l.strip('\n') for l in open(os.path.join(
                voc_path, 'ImageSets', 'Segmentation', 'val.txt')).readlines()]
            sbd_data_list = list(set(sbd_data_list) - set(voc_val_data_list))

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

class PASCAL_PARTS_LOADER(Loader):
    def __init__(self, mode, n_classes, transform=None, target_transform=None, img_size=512, ignore_index=255, do_transform=False, portion=1):
        super(PASCAL_PARTS_LOADER, self).__init__(
            mode, 
            n_classes, 
            transform, 
            target_transform, 
            img_size, 
            ignore_index, 
            do_transform, portion=portion)

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
        data_path = get_data_path('pascalparts')
        if mode == 'train':
            data_list = [l.strip('\n') for l in open(os.path.join(
                data_path, 'ImageSets', 'Person', 'train.txt')).readlines()]
        elif mode == 'val':
            data_list = [l.strip('\n') for l in open(os.path.join(
                data_path, 'ImageSets', 'Person', 'val.txt')).readlines()]

        items = []
        img_path = os.path.join(data_path, 'JPEGImages')
        mask_path = os.path.join(data_path, 'ImageSets', 'Person', 'gt')
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)

        return items

class LIP_LOADER(Loader):
    def __init__(self, mode, n_classes, transform=None, target_transform=None, img_size=256, ignore_index=255, do_transform=False, portion=1):
        super(LIP_LOADER, self).__init__(
            mode, 
            n_classes, 
            transform, 
            target_transform, 
            img_size, 
            ignore_index, 
            do_transform, portion=portion)

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
        return items[0:11716]

class PASCAL_HUMAN_SEMSEG_LOADER(Loader):
    def __init__(self, mode, n_classes, transform=None, target_transform=None, img_size=512, ignore_index=255, do_transform=False, portion=1):
        super(PASCAL_HUMAN_SEMSEG_LOADER, self).__init__(
            mode, 
            n_classes, 
            transform, 
            target_transform, 
            img_size, 
            ignore_index, 
            do_transform, portion=portion)
    
    def __getitem__(self, index):
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
        return np.asarray([
            [0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
            [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
            [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0,64,0], [128,64,0],
            [0,192,0], [128,192,0], [0,64,128]
        ])
    
    def preprocess(self, mode):
        assert mode in ['train', 'val', 'test']
        pascal_data_path = get_data_path('pascal')
        sbd_data_path = get_data_path('sbd')
        items = []

        if mode == 'train':
            p = open(os.path.join(pascal_data_path, 'ImageSets', 'Person', 'train.txt')).readlines()
            s = open(os.path.join(sbd_data_path, 'dataset', 'train.txt')).readlines()
            lines = list(set(p).intersection(s))
            data_list = [l.strip('\n') for l in lines]
        elif mode == 'val':
            p = open(os.path.join(pascal_data_path, 'ImageSets', 'Person', 'val.txt')).readlines()
            s = open(os.path.join(sbd_data_path, 'dataset', 'val.txt')).readlines()
            lines = list(set(p).intersection(s))
            data_list = [l.strip('\n') for l in lines]
        
        img_path = os.path.join(sbd_data_path, 'dataset', 'img')
        mask_path = os.path.join(sbd_data_path, 'dataset', 'cls')
        
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.mat'))
            items.append(item)

        return items

class PASCAL_HUMAN_PARTS_LOADER(Loader):
    def __init__(self, mode, n_classes, transform=None, target_transform=None, img_size=512, ignore_index=255, do_transform=False, portion=1):
        super(PASCAL_HUMAN_PARTS_LOADER, self).__init__(
            mode, 
            n_classes, 
            transform, 
            target_transform, 
            img_size, 
            ignore_index, 
            do_transform, portion=portion)
    
    def __getitem__(self, index):
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
        return np.asarray([
            [0,0,0], [128,0,0], [0,128,0], [128,128,0], 
            [0,0,128], [128,0,128], [0,128,128]
        ])
    
    def preprocess(self, mode):
        assert mode in ['train', 'val', 'test']
        pascal_data_path = get_data_path('pascal')
        sbd_data_path = get_data_path('sbd')
        items = []

        if mode == 'train':
            p = open(os.path.join(pascal_data_path, 'ImageSets', 'Person', 'train.txt')).readlines()
            s = open(os.path.join(sbd_data_path, 'dataset', 'train.txt')).readlines()
            lines = list(set(p).intersection(s))
            data_list = [l.strip('\n') for l in lines]
        elif mode == 'val':
            p = open(os.path.join(pascal_data_path, 'ImageSets', 'Person', 'val.txt')).readlines()
            s = open(os.path.join(sbd_data_path, 'dataset', 'val.txt')).readlines()
            lines = list(set(p).intersection(s))
            data_list = [l.strip('\n') for l in lines]
        
        img_path = os.path.join(sbd_data_path, 'dataset', 'img')
        mask_path = os.path.join(pascal_data_path, 'ImageSets', 'Person', 'gt')

        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)

        return items

class PASCAL_HUMAN_LOADER(Loader):
    def __init__(self, mode, n_classes, transform=None, target_transform=None, img_size=512, ignore_index=255, do_transform=False, portion=1):
        super(PASCAL_HUMAN_LOADER, self).__init__(
            mode, 
            n_classes, 
            transform, 
            target_transform, 
            img_size, 
            ignore_index, 
            do_transform, portion=portion)
    
    def __getitem__(self, index):
        img_path, sbd_mask_path, lip_mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if sbd_mask_path.split('.')[-1] == 'mat':
            sbd_mask = sio.loadmat(sbd_mask_path)['GTcls']['Segmentation'][0][0]
            sbd_mask = Image.fromarray(sbd_mask.astype(np.uint8)).convert('P')
        else:
            sbd_mask = Image.open(sbd_mask_path).convert('P')
        if lip_mask_path.split('.')[-1] == 'mat':
            lip_mask = sio.loadmat(lip_mask_path)['GTcls']['Segmentation'][0][0]
            lip_mask = Image.fromarray(lip_mask.astype(np.uint8)).convert('P')
        else:
            lip_mask = Image.open(lip_mask_path).convert('P')

        if self.do_transform:
            img_transformed, sbd_mask = self.further_transform(img, sbd_mask)
            _, lip_mask = self.further_transform(img, lip_mask)
        else:
            img_transformed, sbd_mask = self.crop(img, sbd_mask)
            _, lip_mask = self.crop(img, lip_mask)

        if self.transform is not None:
            img_transformed = self.transform(img_transformed)
        if self.target_transform is not None:
            sbd_mask = self.target_transform(sbd_mask)
            lip_mask = self.target_transform(lip_mask)
        
        return img_transformed, sbd_mask, lip_mask

    def get_pascal_labels(self):
        return np.asarray([
            [0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
            [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
            [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0,64,0], [128,64,0],
            [0,192,0], [128,192,0], [0,64,128]
        ])
    
    def decode_segmap(self, temp, plot=False, task=0):
        label_colours = self.get_pascal_labels()
        if task == 1:
            label_colours = label_colours[0:7]
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes[task]):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb
    
    def preprocess(self, mode):
        assert mode in ['train', 'val', 'test']
        pascal_data_path = get_data_path('pascal')
        sbd_data_path = get_data_path('sbd')
        items = []

        if mode == 'train':
            p = open(os.path.join(pascal_data_path, 'ImageSets', 'Person', 'train.txt')).readlines()
            s = open(os.path.join(sbd_data_path, 'dataset', 'train.txt')).readlines()
            lines = list(set(p).intersection(s))
            data_list = [l.strip('\n') for l in lines]
        elif mode == 'val':
            p = open(os.path.join(pascal_data_path, 'ImageSets', 'Person', 'val.txt')).readlines()
            s = open(os.path.join(sbd_data_path, 'dataset', 'val.txt')).readlines()
            lines = list(set(p).intersection(s))
            data_list = [l.strip('\n') for l in lines]
        
        img_path = os.path.join(sbd_data_path, 'dataset', 'img')
        semseg_mask_path = os.path.join(sbd_data_path, 'dataset', 'cls')
        parts_mask_path = os.path.join(pascal_data_path, 'ImageSets', 'Person', 'gt')
        
        for it in data_list:
            item = (
                os.path.join(img_path, it + '.jpg'), 
                os.path.join(semseg_mask_path, it + '.mat'), 
                os.path.join(parts_mask_path, it + '.png')
            )
            items.append(item)
        return items

class SBD_LIP_LOADER(Loader):
    def __init__(self, mode, n_classes, transform=None, target_transform=None, img_size=512, ignore_index=255, do_transform=False, portion=1):
        self.sbd_loader = SEMSEG_LOADER(
            mode, 
            n_classes[0], 
            transform, 
            target_transform, 
            img_size, 
            ignore_index, 
            do_transform, portion=portion)
        self.lip_loader = LIP_LOADER(
            mode, 
            n_classes[1], 
            transform, 
            target_transform, 
            img_size, 
            ignore_index, 
            do_transform, portion=portion)
        super(SBD_LIP_LOADER, self).__init__(
            mode, 
            n_classes, 
            transform, 
            target_transform, 
            img_size, 
            ignore_index, 
            do_transform, portion=1)
        
    def __getitem__(self, index):
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

        if 'person' in img_path:
            task = 1
        else:
            task = 0
        return img, mask, task

    def get_pascal_labels(self, task):
        if task == 0:
            return np.asarray([
                [0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
                [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0,64,0], [128,64,0],
                [0,192,0], [128,192,0], [0,64,128]
            ])
        else:
            return np.asarray([
                [0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
                [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0,64,0], [128,64,0],
                [0,192,0], [128,192,0]
            ])
    
    def decode_segmap(self, temp, task=0, plot=False):
        label_colours = self.get_pascal_labels(task)
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes[task]):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def preprocess(self, mode):
        sbd_items = self.sbd_loader.imgs
        lip_items = self.lip_loader.imgs

        lip_items = lip_items[0:len(sbd_items)]

        return sbd_items + lip_items


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