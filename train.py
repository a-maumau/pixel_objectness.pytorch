import torch 
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import argparse
from tqdm import tqdm

from model import POVGG16
from data_loader import get_loader, VOC12Seg
import pair_transforms
import metric

def train(args):
        if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
                
        with torch.cuda.device(args.gpu_device_num):
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

                optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-6)
                batch_batch_count = 0

                pair_transform = pair_transforms.PairCompose([pair_transforms.PairRandomCrop(args.crop_size),
                                                              pair_transforms.PairRandomHorizontalFlip()])

                val_pair_transform = pair_transforms.PairCompose([pair_transforms.PairRandomCrop(args.crop_size)])

                input_transform = transforms.Compose([transforms.ToTensor()])
                
                # file_list_path, img_root, mask_root, pair_transform=None, input_transform=None, target_transform=None
                train_data_set = VOC12Seg(file_list_path=args.train_image_list,
                                          img_root=args.train_image_dir,
                                          mask_root=args.train_mask_dir,
                                          pair_transform=pair_transform,
                                          input_transform=input_transform,
                                          target_transform=None)

                trainval_data_set = VOC12Seg(file_list_path=args.trainval_image_list,
                                          img_root=args.trainval_image_dir,
                                          mask_root=args.trainval_mask_dir,
                                          pair_transform=pair_transform,
                                          input_transform=input_transform,
                                          target_transform=None)

                val_data_set = VOC12Seg(file_list_path=args.val_image_list,
                                          img_root=args.val_image_dir,
                                          mask_root=args.val_mask_dir,
                                          pair_transform=val_pair_transform,
                                          input_transform=input_transform,
                                          target_transform=None)

                train_loader= get_loader(train_data_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                trainval_loader= get_loader(trainval_data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                val_loader= get_loader(val_data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

                # loss
                criterion = nn.NLLLoss2d()
                
                epochs = tqdm(range(args.epochs), ncols=80)

                for epoch in epochs:
                        epoch_total_loss = 0.0
                        _train_loader = tqdm(train_loader, ncols=80)

                        if (epoch+1) % args.decay_every == 0:
                                for param_group in optimizer.param_groups:
                                        param_group['lr'] *= 0.1

                        for img, mask in _train_loader:
                                images = Variable(img).cuda()
                                masks = Variable(mask).cuda()

                                optimizer.zero_grad()
                                
                                outputs = model(images)

                                #outputs = F.upsample(outputs, scale_factor=8, mode='bilinear')
                                outputs = F.upsample(outputs, size=[args.crop_size, args.crop_size], mode='bilinear')

                                #batch_loss = F.cross_entropy(outputs, masks)
                                #batch_loss = criterion(outputs, masks)
                                batch_loss = model.loss(outputs, masks)
                                
                                epoch_total_loss += batch_loss.data[0]
                                
                                batch_loss.backward()
                                optimizer.step()
                                
                                _train_loader.set_description("batch loss: {:5.5f}".format(batch_loss.data[0]))

                        epochs.set_description("[#{}] epoch loss: {:5.5f}".format(epoch+1, epoch_total_loss))

                        if (epoch+1) % args.trainval_every == 0:
                                _trainval_loader = tqdm(train_loader, ncols=80)
                                _trainval_loader.set_description("train val")
                                trainval_total_loss = 0.0
                                pix_acc = 0.0
                                for img, mask in _trainval_loader:
                                        images = Variable(img).cuda()
                                        masks = Variable(mask).cuda()
                                        
                                        outputs = model(images)

                                        #outputs = F.upsample(outputs, scale_factor=8)
                                        outputs = F.upsample(outputs, size=[args.crop_size, args.crop_size], mode='bilinear')
                                        #batch_loss = criterion(outputs, masks)
                                        batch_loss = model.loss(outputs, masks)
                                        #pred = model.inference(images)
                                        #pix_acc += metric.pix_acc(pred, masks)
                                        
                                        trainval_total_loss += batch_loss.data[0]

                                tqdm.write("[#{}] trainval total loss: {:5.5f}, pix acc.:{:5.5f}".format(epoch+1, trainval_total_loss, pix_acc))
    
                        if (epoch+1) % args.save_every == 0:
                                state = {'epoch': epoch + 1,
                                         'optimizer_state_dict' : optimizer.state_dict()}

                                model.save(add_state=state, file_name=os.path.join(args.save_dir,'model_param_e{}.pkl'.format(epoch+1)))
                                tqdm.write("model saved.")

                _val_loader = tqdm(train_loader, ncols=80)
                _val_loader.set_description("val")
                val_total_loss = 0.0
                pix_acc = 0.0
                for img, mask in _val_loader:
                        images = Variable(img).cuda()
                        masks = Variable(mask).cuda()
                        
                        outputs = model(images)
                        
                        #outputs = F.upsample(outputs, scale_factor=8)
                        outputs = F.upsample(outputs, size=[args.crop_size, args.crop_size], mode='bilinear')
                        #batch_loss = criterion(outputs, masks)
                        batch_loss = model.loss(outputs, masks)
                        #pred = model.inference(images)
                        #pix_acc += metric.pix_acc(pred, masks)
                        
                        val_total_loss += batch_loss.data[0]
                tqdm.write("val total loss: {:5.5f}, pix acc.: {:5.5f}".format(val_total_loss, pix_acc))

if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        # settings train.txt  trainval.txt  val.txt
        parser.add_argument('--train_image_dir', type=str, default='./dataset/img', help='directory for train images')
        parser.add_argument('--train_mask_dir', type=str, default='./dataset/mask', help='directory for train mask images')
        parser.add_argument('--trainval_image_dir', type=str, default='./dataset/img', help='directory for train images')
        parser.add_argument('--trainval_mask_dir', type=str, default='./dataset/mask', help='directory for train mask images')
        parser.add_argument('--val_image_dir', type=str, default='./dataset/img', help='directory for val images')
        parser.add_argument('--val_mask_dir', type=str, default='./dataset/mask', help='directory for validation mask images')
        parser.add_argument('--train_image_list', type=str, default='./dataset/train.txt', help='directory of json file for training dataset')
        parser.add_argument('--trainval_image_list', type=str, default='./dataset/trainval.txt', help='directory of json file for validation dataset')
        parser.add_argument('--val_image_list', type=str, default='./dataset/val.txt', help='directory of json file for validation dataset')
        parser.add_argument('--crop_size', type=int, default=321, help='size for image after processing')
        parser.add_argument('--save_dir', type=str, default="./log/", help='size for image after processing')
        parser.add_argument('--save_every', type=int, default=10, help='size for image after processing')
        parser.add_argument('--epochs', type=int, default=200)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--batch_batch_size', type=int, default=64)
        parser.add_argument('--num_workers', type=int, default=4)
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
        
