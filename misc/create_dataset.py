"""
    for VOC and SBD dataset
"""
import os
import argparse

import pickle
import scipy.io
import numpy as np
from PIL import Image
from tqdm import tqdm

class DatasetParser(object):
    img_extension = '.jpg'
    mask_extension = '.png'
    mat_file_extension = '.mat'

    def __init__(self, VOC2012dataset_root="./VOCdevkit", SBDdataset_root="./benchmark_RELEASE"):
        self.voc_path = os.path.join(VOC2012dataset_root, "VOC2012")
        self.sbd_path = os.path.join(SBDdataset_root, "dataset")

        self.train_data = []
        self.val_data = []

        self.voc_train_names = []
        self.voc_val_names = []

    def create_dataset(self):
        # VOC
        self.__load_voc_data()

        # SBD
        self.__load_sbd_data()

        print("train: {} data".format(len(self.train_data)))
        print("val  : {} data".format(len(self.val_data)))

    def read_txt(self, txt_file="train.txt"):
        with open(os.path.join(txt_file), "r") as f:
            image_names = f.readlines()
        
        image_names = [img_name.rstrip("\n") for img_name in image_names]
        return image_names

    def binarize(self, mask):
        """
            binarize an image.
            inplace
        """

        return mask[mask > 0] = 1

    def read_mat(self, mat_filename, key='GTcls'):
            mat = scipy.io.loadmat(mat_filename, mat_dtype=True, squeeze_me=True, struct_as_record=False)
            return mat[key].Segmentation

    def convert_mat2nparray(self, mat_name):
            return np.uint8(self.read_mat(mat_name, key='GTcls'))

    def __load_voc_data(self):
        train_name_list = self.read_txt(os.path.join(self.voc_path, "ImageSets/Segmentation/train.txt"))
        for img_name in tqdm(train_name_list, desc="VOC training"):
            img = np.asarray(Image.open(os.path.join(self.voc_path, "JPEGImages/{}{}".format(img_name, self.img_extension)))).copy()
            mask = np.asarray(Image.open(os.path.join(self.voc_path, "SegmentationClass/{}{}".format(img_name, self.mask_extension)))).copy()
            #mask[mask == 255] = 0
            mask = self.binarize(mask)

            self.train_data.append({"image":img, "mask":mask, "image_name":img_name, "dataset_type":"VOC"})

        """
        train_name_list = self.read_txt(os.path.join(self.voc_path, "ImageSets/Segmentation/trainval.txt"))
        for img_name in train_name_list:
            img = np.asarray(Image.open(os.path.join(self.voc_path, "JPEGImages/{}".format(img_name)))).copy()
            mask = np.asarray(Image.open(os.path.join(self.voc_path, "Segmentation/{}".format(img_name)))).copy()
            mask[mask == 255] = 0

            self.train_data.append({"image":img, "mask":mask, "image_name":img_name})
        """

        val_name_list = self.read_txt(os.path.join(self.voc_path, "ImageSets/Segmentation/val.txt"))
        for img_name in tqdm(val_name_list, desc="VOC validation"):
            img = np.asarray(Image.open(os.path.join(self.voc_path, "JPEGImages/{}{}".format(img_name, self.img_extension)))).copy()
            mask = np.asarray(Image.open(os.path.join(self.voc_path, "SegmentationClass/{}{}".format(img_name, self.mask_extension)))).copy()
            #mask[mask == 255] = 0
            mask = self.binarize(mask)

            self.val_data.append({"image":img, "mask":mask, "image_name":img_name, "dataset_type":"VOC"})

        self.voc_train_names = train_name_list
        self.voc_val_names = val_name_list

    def __load_sbd_data(self):
        """
            it will erase the 255 value pixel in this function.
            well, I don't remember it containg a 255 value....
        """

        train_name_list = self.read_txt(os.path.join(self.sbd_path, "train.txt"))
        for img_name in tqdm(train_name_list, desc="SBD training"):
            # it seems same image in VOC and SBD, so I will skip it.
            if img_name in self.voc_train_names:
                tqdm.write("skipped {}".format(img_name))
                continue

            img = np.asarray(Image.open(os.path.join(self.sbd_path, "img/{}{}".format(img_name, self.img_extension)))).copy()
            mask = self.convert_mat2nparray(os.path.join(self.sbd_path, "cls/{}{}".format(img_name, self.mat_file_extension))).copy()
            #mask[mask == 255] = 0
            mask = self.binarize(mask)

            self.train_data.append({"image":img, "mask":mask, "image_name":img_name, "dataset_type":"SBD"})

        val_name_list = self.read_txt(os.path.join(self.sbd_path, "val.txt"))
        for img_name in tqdm(val_name_list, desc="SBD validation"):
            # it seems same image in VOC and SBD, so I will skip it.
            if img_name in self.voc_val_names:
                tqdm.write("skipped {}".format(img_name))
                continue

            img = np.asarray(Image.open(os.path.join(self.sbd_path, "img/{}{}".format(img_name, self.img_extension)))).copy()
            mask = self.convert_mat2nparray(os.path.join(self.sbd_path, "cls/{}{}".format(img_name, self.mat_file_extension))).copy()
            #mask[mask == 255] = 0
            mask = self.binarize(mask)

            self.val_data.append({"image":img, "mask":mask, "image_name":img_name, "dataset_type":"SBD"})

    def save_dataset(self, save_dir):
        with open("{}".format(os.path.join(save_dir, "train.pkl")), "wb") as f:
            pickle.dump(self.train_data, f)

        with open("{}".format(os.path.join(save_dir, "val.pkl")), "wb") as f:
            pickle.dump(self.val_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--voc_root', type=str, default='./VOCdevkit', help='path of VOCdevkit')
    parser.add_argument('--sbd_root', type=str, default='./benchmark_RELEASE', help='path of benchmark_RELEASE')
    parser.add_argument('--output_dir', type=str, default='./', help='output dir.')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    dp = DatasetParser(args.voc_root, args.sbd_root)
    dp.create_dataset()
    dp.save_dataset(args.output_dir)
