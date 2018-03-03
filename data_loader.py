import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import os

import numpy as np
from PIL import Image

# segmentation template class
class _segmentation(data.Dataset):
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        
        # Using PIL to open images is very slow? i think... if you don't have enough memory you should use here
        #_img = Image.open(os.path.join(self.img_root,self.images[index]["file_name"])).convert('RGB')
        #_mask_img = Image.open( os.path.join(self.mask_root,re.sub(r'.jpg', "",self.images[index]["file_name"])+".png"))
        
        if self.pair_transform is not None:
                #_img, _mask_img = self.pair_transform(_img, _mask_img)
            _img, _mask_img = self.pair_transform(self.img[index], self.mask_img[index])
                
        if self.input_transform is not None:
            _img = self.input_transform(_img)
                
        if self.target_transform is not None:
            _mask_img = self.target_transform(_mask_img)
        else:
            _mask_img = torch.from_numpy(np.asarray(_mask_img)).type(torch.LongTensor)
                
        return _img, _mask_img

    def __len__(self):
        return self.data_num

class VOC12Seg(_segmentation):
    def __init__(self, file_list_path, img_root, mask_root, pair_transform=None, input_transform=None, target_transform=None):
        self.img_root = img_root
        self.mask_root = mask_root
        
        # VOC12 seems to be an .txt file that has a per line style.
        with open(os.path.join(file_list_path), "r") as file:
            self.image_names = file.readlines()
                
        # equals to map(lamda x: x.rstrip(\n), self.image)
        self.image_names = [img_name.rstrip("\n") for img_name in self.image_names]

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.pair_transform = pair_transform
        #self.data_num = len(self.images_names)
        self.img = []
        self.mask_img = []

        for img_name in self.image_names:
            try:
                # save as num py array
                _img = np.asarray(Image.open(os.path.join(self.img_root, img_name+".jpg")).convert('RGB')) # not thinking there is a empty input...
                # I dont know is this neede...
                _img.flags.writeable = True
                _img = Image.fromarray(np.uint8(_img))
                
                # same file name but it is .png
                _mask_img = np.asarray(Image.open(os.path.join(self.mask_root, img_name+".png")).convert('P'))
                _mask_img.flags.writeable = True
                _mask_img = Image.fromarray(np.uint8(_mask_img))

                self.img.append(_img)
                self.mask_img.append(_mask_img)

            except Exception as e:
                print(e)
                print("pass {}".format(img_name))

            self.data_num = len(self.img)

class PODataLoader(_segmentation):
    """
        opening the two datasets, there is same file name between both.
        so need to be distinguishable.
    """
    def __init__(self, VOC_list, SBD_list, voc_img_root, sbd_img_root, voc_mask_root, sbd_mask_root, pair_transform=None, input_transform=None, target_transform=None):
        self.voc_img_root = voc_img_root
        self.voc_mask_root = voc_mask_root
        self.sbd_img_root = sbd_img_root
        self.sbd_mask_root = sbd_mask_root
        
        # VOC12 seems to be an .txt file that has a per line style.
        with open(os.path.join(VOC_list), "r") as file:
            self.voc_image_names = file.readlines()


        with open(os.path.join(SBD_list), "r") as file:
            self.sbd_image_names = file.readlines()
                
        self.voc_image_names = [img_name.rstrip("\n") for img_name in self.voc_image_names]
        self.sbd_image_names = [img_name.rstrip("\n") for img_name in self.sbd_image_names]

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.pair_transform = pair_transform

        self.image_names = []
        self.img = []
        self.mask_img = []

        # this part is redundant for the input images...
        # load voc dataset
        for img_name in self.voc_image_names:
            try:
                # save as num py array
                _img = np.asarray(Image.open(os.path.join(self.voc_img_root, img_name+".jpg")).convert('RGB')) # not thinking there is a empty input...
                # I dont know is this neede...
                _img.flags.writeable = True
                _img = Image.fromarray(np.uint8(_img))
                
                # same file name but it is .png
                _mask_img = np.asarray(Image.open(os.path.join(self.voc_mask_root, img_name+".png")).convert('P'))
                _mask_img.flags.writeable = True
                _mask_img = Image.fromarray(np.uint8(_mask_img))

                self.image_names.append(img_name)
                self.img.append(_img)
                self.mask_img.append(_mask_img)

            except Exception as e:
                print(e)
                print("pass {}".format(img_name))

        # load sbd dataset
        for img_name in self.sbd_image_names:
            try:
                # save as num py array
                _img = np.asarray(Image.open(os.path.join(self.sbd_img_root, img_name+".jpg")).convert('RGB')) # not thinking there is a empty input...
                # I dont know is this neede...
                _img.flags.writeable = True
                _img = Image.fromarray(np.uint8(_img))
                
                # same file name but it is .png
                _mask_img = np.asarray(Image.open(os.path.join(self.sbd_mask_root, img_name+".png")).convert('P'))
                _mask_img.flags.writeable = True
                _mask_img = Image.fromarray(np.uint8(_mask_img))

                self.image_names.append(img_name)
                self.img.append(_img)
                self.mask_img.append(_mask_img)

            except Exception as e:
                print(e)
                print("pass {}".format(img_name))

        self.data_num = len(self.img)

class TestDataLoader(data.Dataset):
    def __init__(self, img_dir, input_transform=None):
        self.img_dir = img_dir                
        # VOC12 seems to be an .txt file that has a per line style.
        images_list = os.listdir(self.img_dir)

        self.input_transform = input_transform
        self.image_names = []
        self.img = []

        # this part is redundant for the input images...
        # load voc dataset
        for img_name in images_list:
            try:
                # save as num py array
                _img = np.asarray(Image.open(os.path.join(self.img_dir, img_name+".jpg")).convert('RGB')) # not thinking there is a empty input...
                # I dont know is this neede...
                _img.flags.writeable = True
                _img = Image.fromarray(np.uint8(_img))
                
                self.image_names.append(img_name)
                self.img.append(_img)

            except Exception as e:
                print(e)
                print("pass {}".format(img_name))

        self.data_num = len(self.img)

    def __getitem__(self, index):
        _img = self.img[index]
        
        if self.input_transform is not None:
            _img = self.input_transform(_img)

        return _img, self.image_names[index]

    def __len__(self):
        return self.data_num

def collate_fn(data):
    _img, _mask_img = zip(*data)
    _img = torch.stack(_img, 0)
    _mask_img = torch.stack(_mask_img, 0)
    
    return _img, _mask_img


def get_loader( data_set, batch_size, shuffle, num_workers):
    data_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader

def get_test_loader( data_set, batch_size, shuffle, num_workers):
    data_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)

    return data_loader
