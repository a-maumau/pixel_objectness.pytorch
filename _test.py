"""
    this program is for checking other modules is eorking correctly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import argparse
from datetime import datetime
from tqdm import tqdm

from model import POVGG16
from data_loader import get_loader, VOC12Seg, PODataLoader
import pair_transforms
import metric

def train(args):
    with torch.cuda.device(args.gpu_device_num):
        def to_var(x, volatile=False):
            if torch.cuda.is_available():
                x = x.cuda()
            return Variable(x, volatile=volatile)

        model = POVGG16().cuda()

        if args.trained_path is not None:
            try:
                chkp = torch.load(args.trained_path)
                model_dict = model.state_dict()
                    
                pretrained_dict = {k: v for k, v in chkp.items() if k in model_dict}
                #print(pretrained_dict.keys())
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(e)
                print("cannot load pretrained data.")

        print(args)
        for idx, m in enumerate(model.modules()):
            print(idx, '->', m)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-5)
        batch_batch_count = 0

        pair_transform = pair_transforms.PairCompose([pair_transforms.PairRandomCrop(args.crop_size),
                                                      pair_transforms.PairRandomHorizontalFlip()])

        val_pair_transform = pair_transforms.PairCompose([pair_transforms.PairRandomCrop(args.crop_size)])

        input_transform = transforms.Compose([transforms.ToTensor()])
        
        # file_list_path, img_root, mask_root, pair_transform=None, input_transform=None, target_transform=None
        train_data_set = PODataLoader(VOC_list=args.voc_train_image_list,
                                  SBD_list=args.sbd_train_image_list,
                                  voc_img_root=args.voc_image_dir,
                                  sbd_img_root=args.sbd_image_dir,
                                  voc_mask_root=args.voc_mask_dir,
                                  sbd_mask_root=args.sbd_mask_dir,
                                  pair_transform=pair_transform,
                                  input_transform=input_transform,
                                  target_transform=None)

        trainval_data_set = PODataLoader(VOC_list=args.voc_trainval_image_list,
                                  SBD_list=args.sbd_val_image_list,
                                  voc_img_root=args.voc_image_dir,
                                  sbd_img_root=args.sbd_image_dir,
                                  voc_mask_root=args.voc_mask_dir,
                                  sbd_mask_root=args.sbd_mask_dir,
                                  pair_transform=pair_transform,
                                  input_transform=input_transform,
                                  target_transform=None)

        train_loader= get_loader(train_data_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        trainval_loader= get_loader(trainval_data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        epoch_total_loss = 0.0
        _train_loader = tqdm(train_loader, ncols=80)

        # check train
        for img, mask in _train_loader:
            images = to_var(img, volatile=False)
            masks = to_var(mask, volatile=False)

            #optimizer.zero_grad()
            
            #outputs = model(images)

            #outputs = F.upsample(outputs, size=[args.crop_size, args.crop_size], mode='bilinear')

            #batch_loss = model.loss(masks, masks)
            
            #epoch_total_loss += batch_loss.data[0]
            
            #batch_loss.backward()
            #optimizer.step()
            
            #_train_loader.set_description("train batch loss: {:5.5f}".format(batch_loss.data[0]))

            #epochs.set_description("train epoch loss: {:5.5f}".format(epoch_total_loss))

        # check train validation
        model.eval()
        _trainval_loader = tqdm(train_loader, ncols=80)
        _trainval_loader.set_description("train val")
        trainval_total_loss = 0.0
        pix_acc = 0.0
        batch_count = 0
        for img, mask in _trainval_loader:
            images = to_var(img, volatile=False)
            #masks = to_var(mask, volatile=False)

            #outputs = model.inference(images)
            #outputs = F.upsample(outputs, size=[args.crop_size, args.crop_size], mode='bilinear')
            acc_tmp = metric.pix_acc(mask, mask)
            pix_acc += acc_tmp
            
            batch_count += 1
            
            _trainval_loader.set_description("trainval mean pix acc: {:5.5f}".format(acc_tmp))

        tqdm.write("trainval total: mean pix acc. {:5.5f}".format(pix_acc/batch_count))
        model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--voc_image_dir', type=str, default='./dataset/voc/img', help='directory for train images')
    parser.add_argument('--voc_mask_dir', type=str, default='./dataset/voc/mask', help='directory for train mask images')
    parser.add_argument('--sbd_image_dir', type=str, default='.', help='directory for train images')
    parser.add_argument('--sbd_mask_dir', type=str, default='.', help='directory for train mask images')

    parser.add_argument('--voc_train_image_list', type=str, default='./dataset/voc/train.txt', help='directory of image list of train')
    parser.add_argument('--voc_trainval_image_list', type=str, default='./dataset/voc/trainval.txt', help='directory of image list of trainval')
    parser.add_argument('--voc_val_image_list', type=str, default='./dataset/voc/val.txt', help='directory of image list of validation')
    parser.add_argument('--sbd_train_image_list', type=str, default='./dataset/sbd/train.txt', help='directory of image list of train')
    parser.add_argument('--sbd_val_image_list', type=str, default='./dataset/sbd/val.txt', help='directory of image list of trainval')

    parser.add_argument('--crop_size', type=int, default=321, help='size for image after processing')
    parser.add_argument('--save_dir', type=str, default="./log/", help='size for image after processing')
    parser.add_argument('--save_every', type=int, default=10, help='size for image after processing')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--decay_every', type=int, default=50)
    parser.add_argument('--gpu_device_num', type=int, default=0)
    parser.add_argument('--trainval_every', type=int, default=10)
    
    parser.add_argument('--trained_path', type=str, default="vgg16-397923af.pth", help="pytorch official pretrained data is like vgg16-397923af.pth")
    parser.add_argument('--setting_file', type=str, default=None, help="use arguments from setting file")
    
    # flags
    parser.add_argument('-batch_batch', action="store_true", default=False, help='calc in batch in batch')
    parser.add_argument('-use_tensorboard', action="store_true", default=False, help='calc in batch in batch')
    
    args = parser.parse_args()
    
    train(args)
