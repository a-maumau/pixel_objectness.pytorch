import torch 
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.datasets as dsets
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import argparse
from tqdm import tqdm
from PIL import Image

from models.vgg_po import VGG16_PixelObjectness as Model
from data_loader import get_pred_loader

def prediction(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
            
    with torch.no_grad():
        model = Model(input_channel=3, num_class=2).eval()
        
        if torch.cuda.is_available() and not args.nogpu:
            map_device = torch.device('cuda:{}'.format(args.gpu_device_num))
        else:
            map_device = torch.device('cpu')

        model = model.to(map_device)

        if args.trained_path is not None:
            try:
                chkp = torch.load(args.trained_path)
                print(chkp.keys())
                model_dict = model.state_dict()
                
                pretrained_dict = {k: v for k, v in chkp["state_dict"].items() if k in model_dict}
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

        data_loader = get_pred_loader(args)

        for img, file_name in tqdm(data_loader, ncols=80):
            images = img.to(map_device)

            outputs, prob_map = model.inference(images)
            outputs = F.upsample(outputs, size=[args.resize_size, args.resize_size], mode='bilinear', align_corners=False)

            # save all
            for n in range(outputs.size()[0]):
                torchvision.utils.save_image(images[n].cpu().data, "{}_input.png".format(os.path.join(args.save_dir, file_name[n])), nrow=0, padding=0, normalize=False)
                torchvision.utils.save_image(outputs[n].cpu().data, "{}_predict.png".format(os.path.join(args.save_dir, file_name[n])), nrow=0, padding=0, normalize=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--image_dir', type=str, default='./dataset/test', help='directory for test images')

    # detail settings
    parser.add_argument('--resize_size', type=int, default=321, help='size for image after processing') # paper default
    parser.add_argument('--save_dir', type=str, default="./log/predicted", help='dir of saving log and model parameters and so on.')
    parser.add_argument('--batch_size', type=int, default=10, help="mini batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="worker num. of data loader")
    parser.add_argument('--gpu_device_num', type=int, default=0, help="device number of gpu")    
    parser.add_argument('--trained_path', type=str, default="./log/pretrained.pth", help="")

    parser.add_argument('-nogpu', action="store_true", default=False, help="don't use gpu")
    
    args = parser.parse_args()
    
    prediction(args)
    
