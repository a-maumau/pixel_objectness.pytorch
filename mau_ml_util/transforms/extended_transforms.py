import math
import random
import numpy as np
from PIL import Image, ImageFilter

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

# not yet.
"""
class MedianFilter(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, pic):

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        # if image has only 2 channels make them 3
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)

        pic = ndimage.median_filter(pic, size=self.size)
        return pic
"""

if __name__ == '__main__':
    img = Image.open("test.jpg")

    t = AddGaussianNoise(30)
    n_img = t(img)
    n_img.save("gaussian_noised.png")

    t = LowpassFilter(0.05)
    n_img = t(img)
    n_img.save("gaussian_filtered.png")

    t = GaussianBlur(2.0)
    n_img = t(img)
    n_img.save("gaussian_blured.png")
