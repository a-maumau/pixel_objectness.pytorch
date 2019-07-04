"""
    wrapping the code of Pytorch
    to perform pair consistent of random values.
    I only wrapped the things I needed.
"""
import os
import math
import random

import torch
import torchvision.transforms.functional as F

import numbers
import numpy as np
from PIL import Image, ImageOps, ImageFilter

class AddGaussianNoise(object):
    def __init__(self, percent=5):
        # this percent should calc. at running?
        # like, math.sqrt(self.percent)*img_std or math.sqrt((self.percent*img_std)^2)
        self.noise_coeff = math.sqrt(percent/100.0)

    def __call__(self, input_img):
        return self._process(input_img)

    def _process(self, input_img):
        """
            input_img: PIL.Image.Image
        """

        img_array = np.asarray(input_img, dtype="int16", order="F") # in case it over/under flow

        # channel wise, but should be calc. with all of pixels?
        img_std_ch0 = np.std(img_array[:,:,0])/255.0
        img_std_ch1 = np.std(img_array[:,:,1])/255.0
        img_std_ch2 = np.std(img_array[:,:,2])/255.0

        noise_ch0 = np.random.normal(0, self.noise_coeff*img_std_ch0, img_array.shape[:2])*255
        noise_ch1 = np.random.normal(0, self.noise_coeff*img_std_ch1, img_array.shape[:2])*255
        noise_ch2 = np.random.normal(0, self.noise_coeff*img_std_ch2, img_array.shape[:2])*255

        img_array[:,:,0] += noise_ch0.astype("int16")
        img_array[:,:,1] += noise_ch1.astype("int16")
        img_array[:,:,2] += noise_ch2.astype("int16")

        img_array = np.clip(img_array, 0, 255)


        return Image.fromarray(np.uint8(img_array))

# stochastically, it might has a non-changed image in AddGaussianNoise process
# for explicitly having non-changed image in the processing
class RanmdomAddGaussianNoise(AddGaussianNoise):
    def __init__(self, percent=5, prob=0.5):
        super(RanmdomAddGaussianNoise, self).__init__(percent)

        self.prob = prob

    def __call__(self, input_img):
        if random.random() < self.prob:
            return self._process(input_img)

        return input_img

class GaussianBlur(object):
    def __init__(self, radius):
        self.radius = radius

    def _blur(self, input_img, radius):
        return input_img.filter(ImageFilter.GaussianBlur(radius))

    def __call__(self, input_img):
        return self._blur(input_img, self.radius)

# add prob. and scaling to GaussianFilter
class RanmdomGaussianBlur(GaussianBlur):
    def __init__(self, radius=2, prob=0.5, scale=(0.8, 1.2)):
        super(RanmdomGaussianBlur, self).__init__(radius)

        self.prob = prob
        self.scale_min = min(0.0, self.prob*scale[0])
        self.scale_max = self.prob*scale[1]

    def __call__(self, input_img):
        if random.random() < self.prob:
            radius = random.uniform(self.scale_min, self.scale_max)

            return self._blur(input_img, radius)

        return input_img

class LowpassFilter(object):
    def __init__(self, pass_size=0.8):
        self.pass_size = pass_size

    # lowpass_filter is borrowed from
    # https://algorithm.joho.info/programming/python/opencv-fft-low-pass-filter-py/ 
    def _lowpass_filter(self, img_array, size):
        """
            img_array: numpy.ndarray:uint8
                shape must be (width, height)
        """

        # FFT in 2dim
        src = np.fft.fft2(img_array)
        h, w = img_array.shape
       
        # image center
        cy, cx =  int(h/2), int(w/2)
        # filter size
        rh, rw = int(size*cy), int(size*cx)

        # swap 1st quadrant and 3rd quadrant、2nd quadrant and 4th quadrant
        fsrc = np.fft.fftshift(src)

        fdst = np.zeros(src.shape, dtype=complex)
        # only hold the value of around center.
        fdst[cy-rh:cy+rh, cx-rw:cx+rw] = fsrc[cy-rh:cy+rh, cx-rw:cx+rw]
        
        # swap back the quadrants to the original
        fdst =  np.fft.fftshift(fdst)

        # inverse FFT
        dst = np.fft.ifft2(fdst)
       
        # take only real part
        return np.uint8(dst.real)
        
    def __call__(self, input_img):
        img_array = np.asarray(input_img, dtype="uint8")
        img_array.flags.writeable = True

        img_array[:,:,0] = self._lowpass_filter(img_array[:,:,0], self.pass_size)
        img_array[:,:,1] = self._lowpass_filter(img_array[:,:,1], self.pass_size)
        img_array[:,:,2] = self._lowpass_filter(img_array[:,:,2], self.pass_size)

        return Image.fromarray(np.uint8(np.clip(img_array, 0, 255)))

# add prob. and scaling to GaussianFilter
class RanmdomLowpassFilter(LowpassFilter):
    def __init__(self, pass_size=0.8, prob=0.5, scale=(0.9, 1.1)):
        super(RanmdomGaussianFilter, self).__init__(pass_size)

        self.prob = prob
        self.scale_min = min(0.0, self.prob*scale[0])
        self.scale_max = max(1.0, self.prob*scale[1])

    def __call__(self, input_img):
        if random.random() < self.prob:
            size = random.uniform(self.scale_min, self.scale_max)

            return self._lowpass_filter(input_img, size)

        return input_img

class Random90degreeRotation(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, input_img):
        if random.random() < self.prob:
            if random.random() < 0.5:
                return input_img.rotate(90)
            else:
                return input_img.rotate(-90)

        return input_img

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

class RandomSizedCropResize(object):
    """
        It first crop in size of [crop_size*crop_scale[0], crop_size*crop_scale[1]]
        then 
        then resize to size of resize_size
    """
    def __init__(self, crop_size, resize_size, crop_scale=(0.8, 1.2), crop_ratio=(3. / 4., 4. / 3.), padding=None, pad_if_needed=False, fill=0, padding_mode='constant', interpolation=Image.BILINEAR):
        if isinstance(size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = tuple(crop_size)

        if isinstance(size, numbers.Number):
            self.resize_size = (int(resize_size), int(resize_size))
        else:
            self.resize_size = tuple(resize_size)

        self.scale = scale
        self.ratio = ratio
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.interpolation = interpolation

        self.max_crop_scaling = max(self.crop_scale)
        self.max_crop_ratio = max(self.crop_ratio+tuple([1.0]))
        self.max_crop_size = self.max_crop_scaling*self.max_crop_ratio

    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = self.crop_size[0] * self.crop_size[1]
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
            assuming the input_img and target_img has same size

            at this moment, I'm lazy to think if crop size has different value in w, h
            so consideringing as same size
        """

        # crop max size is crop_size * self.max_crop_size
        # thinking the worst case

        if self.padding is not None:
            input_img = F.pad(input_img, self.padding, self.fill, self.padding_mode)
            target_img = F.pad(target_img, self.padding, 0, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]*self.max_crop_size:
            input_img = F.pad(input_img, (self.size[1]*self.max_crop_size - input_img.size[0], 0), self.fill, self.padding_mode)
            target_img = F.pad(targe_timg, (self.size[1]*self.max_crop_size - input_img.size[0], 0), 0, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and input_img.size[1] < self.size[0]*self.max_crop_size:
            input_img = F.pad(input_img, (0, self.size[0]*self.max_crop_size - input_img.size[1]), self.fill, self.padding_mode)
            target_img = F.pad(target_img, (0, self.size[0]*self.max_crop_size - input_img.size[1]), 0, self.padding_mode)

        i, j, h, w = self.get_params(input_img, self.scale, self.ratio)
        
        return F.resized_crop(input_img, i, j, h, w, self.resize_size, self.interpolation), F.resized_crop(target_img, i, j, h, w, self.resize_size, Image.NEAREST)

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
