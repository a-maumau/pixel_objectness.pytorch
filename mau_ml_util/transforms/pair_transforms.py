"""
    wrapping the code of Pytorch
    to perform pair consistent of random values.
    I only wrapped the things I needed.
"""

import torch
import torchvision.transforms.functional as F

import math
import random
import os
from PIL import Image, ImageOps
import numbers

class PairCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_img, target_img):
        for t in self.transforms:
            input_img, target_img = t(input_img, target_img)

        return input_img, target_img

class PairResize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, input_img, target_img):
        return F.resize(input_img, self.size, Image.BILINEAR), F.resize(target_img, self.size, Image.NEAREST)

class PairRandomHorizontalFlip(object):
    def __call__(self, input_img, target_img):
        if random.random() < 0.5:
            return input_img.transpose(Image.FLIP_LEFT_RIGHT), target_img.transpose(Image.FLIP_LEFT_RIGHT)

        return input_img, target_img

class PairRandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target_img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img), F.vflip(target_img)
        return img, target_img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class PairCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, input_img, target_img):
        return F.center_crop(input_img, self.size), F.center_crop(target_img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class PairRandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, input_img, target_img):
        """
            not thinking that img and mask are not same size
        """
        if self.padding is not None:
            input_img = F.pad(input_img, self.padding, self.fill, self.padding_mode)
            target_img = F.pad(target_img, self.padding, 0, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            input_img = F.pad(input_img, (self.size[1] - input_img.size[0], 0), self.fill, self.padding_mode)
            target_img = F.pad(targe_timg, (self.size[1] - input_img.size[0], 0), 0, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and input_img.size[1] < self.size[0]:
            input_img = F.pad(input_img, (0, self.size[0] - input_img.size[1]), self.fill, self.padding_mode)
            target_img = F.pad(target_img, (0, self.size[0] - input_img.size[1]), 0, self.padding_mode)

        i, j, h, w = self.get_params(input_img, self.size)

        return F.crop(input_img, i, j, h, w), F.crop(target_img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)
    """
    old version
    def __init__(self, size, padding=0):
        ## size = (width, height)
        
        if isinstance(size, numbers.Number):
            self.crop_sizew = int(size)
            self.crop_sizeh = int(size)
        else:
            self.crop_sizew = int(size[0])
            self.crop_sizeh = int(size[1])

        self.padding = padding

    def __call__(self, input_img, target_img):
        if self.padding > 0:
            input_img = ImageOps.expand(input_img, border=self.padding, fill=0)
            target_img = ImageOps.expand(target_img, border=self.padding, fill=0)

        # assuming input_img and target_img has same size
        w, h = input_img.size
        if w == self.crop_sizew and h == self.crop_sizeh:
            return input_img, target_img
        if w-self.crop_sizew < 0 or h-self.crop_sizeh < 0:
            add_size = w-self.crop_sizew if w-self.crop_sizeh < h-self.crop_sizeh else h-self.crop_sizeh
            input_img = input_img.resize((self.crop_sizew-add_size, self.crop_sizeh-add_size), Image.BILINEAR)
            target_img = target_img.resize((self.crop_sizew-add_size, self.crop_sizeh-add_size), Image.BILINEAR)
            w -= add_size
            h -= add_size

        x1 = random.randint(0, w - self.crop_sizew)
        y1 = random.randint(0, h - self.crop_sizeh)
        return input_img.crop((x1, y1, x1 + self.crop_sizew, y1 + self.crop_sizeh)), target_img.crop((x1, y1, x1 + self.crop_sizew, y1 + self.crop_sizeh))
    """

class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, input_img, target_img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(input_img, self.scale, self.ratio)
        return F.resized_crop(input_img, i, j, h, w, self.size, self.interpolation), F.resized_crop(target_img, i, j, h, w, self.size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

class PairRandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, input_img, target_img):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return input_img.rotate(rotate_degree, Image.BILINEAR), target_img.rotate(rotate_degree, Image.NEAREST)
