import collections
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from PIL import Image, ImageMath
from torch.utils import data

from main import get_data_path
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class Loader(data.Dataset):
    def __init__(self, mode, n_classes, transform=None, target_transform=None, img_size=512, ignore_index=255, do_transform=False):
        super(Loader, self).__init__()
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
        self.n_classes = n_classes

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.imgs)

    def get_pascal_labels(self):
        raise NotImplementedError

    def preprocess(self, mode):
        raise NotImplementedError
    
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


