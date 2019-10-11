import os
import math
import random

import torch
import torchvision.transforms as transforms

import mau_transforms
from dataset import SegmentationDataSet, PredictionLoader

# data loader for dataset
def get_loader(dataset, batch_size=64, shuffle=True, num_workers=8):
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)

    return data_loader

def get_train_loader(args, normalize_param=([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])):
    pair_transform_content = [mau_transforms.PairRandomHorizontalFlip()]
    input_transform_content = []
    target_transform_content = None

    # in the paper,it only using horizontal flip
    pair_transform_content.append(mau_transforms.PairRandomRotate(args.rotate_deg))
    pair_transform_content.append(mau_transforms.PairResize(args.resize_size))
    pair_transform_content.append(mau_transforms.PairRandomCrop(args.crop_size))

    # input transforms #####################################################################
    #input_transform_content.append(mau_transforms.RanmdomGaussianBlur(radius=args.blur_radius, prob=args.blur_prob, scale=(args.blur_scale_min, args.blur_scale_max)))

    #transforms.Normalize(mean=(0.452, 0.431, 0.399),std=(0.277, 0.273, 0.285))
    input_transform_content.append(transforms.ToTensor())

    #if args.normalize:
    input_transform_content.append(transforms.Normalize(*normalize_param))


    pair_transform = mau_transforms.PairCompose(pair_transform_content)
    input_transform = transforms.Compose(input_transform_content)
    target_transform = None

    dataset_args = {"pickle_path":args.train_dataset,
                    "pair_transform":pair_transform,
                    "input_transform":input_transform,
                    "target_transform":target_transform,
                    "return_original":False,
                }

    return get_loader(SegmentationDataSet(**dataset_args),
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=args.num_workers
                    )

def get_val_loader(args, normalize_param=([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])):
    val_pair_transform = mau_transforms.PairCompose([mau_transforms.PairResize(args.resize_size),
                                                      mau_transforms.PairCenterCrop(args.crop_size)])
    val_input_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(*normalize_param)])
    val_target_transform = None

    dataset_val_args = {
                        "pickle_path":args.val_dataset,
                        "pair_transform":val_pair_transform,
                        "input_transform":val_input_transform,
                        "target_transform":val_target_transform,
                        "return_original":True
                    }

    return get_loader(SegmentationDataSet(**dataset_val_args),
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.num_workers
                    )

def get_pred_loader(args, normalize_param=([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])):
    # if you want predict in original size, you should set batch_size to 1 and delete the crop code
    input_transform = transforms.Compose([transforms.Scale(args.resize_size),
                                          transforms.CenterCrop(args.resize_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(*normalize_param)])

    dataset_args = {
                        "img_root": args.image_dir,
                        "input_transform":input_transform,
                    }

    return get_loader(PredictionLoader(**dataset_args),
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.num_workers
                    )
