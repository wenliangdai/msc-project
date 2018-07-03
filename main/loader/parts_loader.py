import os
import random
import collections
import torch
import numpy as np
# import scipy.misc as m
import scipy.io as sio

import matplotlib.pyplot as plt
from PIL import Image, ImageMath

from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

from main import get_data_path


class VOC_parts(data.Dataset):
    def __init__(self, mode, transform=None, target_transform=None, img_size=512, ignore_index=255, do_transform=False):
        self.imgs = self.preprocess(mode=mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.ignore_index = ignore_index
        self.do_transform = do_transform
        self.filler = [0, 0, 0]
        self.n_classes = 7 # head, torso, upper/lower arm, upper/lower leg, background

    def __getitem__(self, index):
        img = None
        mask = None

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('P')

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

        if self.do_transform:
            img, mask = self.further_transform(img, mask)
        else:
            img, mask = self.crop(img, mask)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)
    
    def further_transform(self, img, mask):
        img, mask = self.scale(img, mask)
        img, mask = self.crop(img, mask)
        img, mask = self.flip(img, mask)
        img, mask = self.rotate(img, mask)

        return img, mask
    
    def scale(self, img, mask, low=0.5, high=2.0):
        # 对图像造成 0.5 - 2 倍的缩放
        w, h = img.size
        resize = random.uniform(low, high)
        new_w, new_h = int(resize * w), int(resize * h)
        image_transform = Resize(size=(new_h, new_w))
        label_transform = Resize(size=(new_h, new_w), interpolation=Image.NEAREST)

        return (image_transform(img), label_transform(mask))

    def crop(self, img, mask):
        w, h = img.size
        th, tw = self.img_size

        # 如果图像尺寸小于要求尺寸
        if w < tw or h < th:
            padw, padh = max(tw - w, 0), max(th - h, 0)
            w += padw
            h += padh
            im = Image.new(img.mode, (w, h), tuple(self.filler))
            im.paste(img, (int(padw/2),int(padh/2)))
            l = Image.new(mask.mode, (w, h), self.ignore_index)
            l.paste(mask, (int(padw/2),int(padh/2)))
            img = im
            mask = l

        if w == tw and h == th:
            return img, mask

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))
    
    def flip(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

    def rotate(self, img, mask):
        angle = random.uniform(-10, 10)

        mask = np.array(mask, dtype=np.int32) - self.ignore_index
        mask = Image.fromarray(mask)
        img = tuple([ImageMath.eval("int(a)-b", a=j, b=self.filler[i]) for i, j in enumerate(img.split())])

        mask = mask.rotate(angle, resample=Image.NEAREST)
        img = tuple([k.rotate(angle, resample=Image.BICUBIC) for k in img])

        mask = ImageMath.eval("int(a)+b", a=mask, b=self.ignore_index)
        img = Image.merge(mode='RGB', bands=tuple(
            [ImageMath.eval("convert(int(a)+b,'L')", a=j, b=self.filler[i]) for i, j in enumerate(img)]))
        
        return img, mask

    def decode_segmap(self, temp, plot=False):
        label_colours = self.get_pascal_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
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

    def get_pascal_labels(self):
        # 7 classes
        return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128], [0,128,128]])

    def preprocess(self, mode):
        assert mode in ['train', 'val', 'test']
        items = []
        data_path = get_data_path('parts')
        
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

    def get_part_index(self, part_name):
        '''
        coarse partition:
        head = 1
        torso = 2
        arm = 3
        leg = 4
        (background = 0)
        There are 24 finer parts in total
        '''
        if part_name in ['head','leye','reye','lear','rear','lebrow','rebrow','nose','mouth','hair']:
            return 1
        if part_name in ['torso','neck']:
            return 2
        if part_name in ['llarm','luarm','lhand','rlarm','ruarm','rhand']:
            return 3
        if part_name in ['llleg','luleg','lfoot','rlleg','ruleg','rfoot']:
            return 4
