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

from model import POVGG16
from data_loader import get_test_loader, TestDataLoader
import pair_transforms
import metric

def save_image_tensor(tensor):
    im = Image.fromarray()
    im.save(filename)

def prediction(args):
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

        input_transform = transforms.Compose([transforms.Scale(args.resize_size), transforms.CenterCrop(args.resize_size), transforms.ToTensor()])

        input_data = TestDataLoader(img_dir=args.image_dir, input_transform=input_transform)

        data_loader = get_test_loader(input_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        for img, file_name in tqdm(data_loader, ncols=80):
            images = Variable(img).cuda()

            outputs = model.inference(images)
            outputs = F.upsample(outputs, size=[args.resize_size, args.resize_size], mode='bilinear')

            for n in range(outputs.size()[0]):
                torchvision.utils.save_image(images[n].cpu().data, "{}_input.png".format(os.path.join(args.save_dir, file_name[n])), nrow=0, padding=0, normalize=False)
                torchvision.utils.save_image(outputs[n].cpu().data, "{}_predict.png".format(os.path.join(args.save_dir, file_name[n])), nrow=0, padding=0, normalize=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--image_dir', type=str, default='./dataset/test', help='directory for train images')
    parser.add_argument('--resize_size', type=int, default=321, help='resize for input')
    parser.add_argument('--save_dir', type=str, default="./predicted_image/", help='save dir')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_device_num', type=int, default=0)
    
    parser.add_argument('--trained_path', type=str, default="./log/pretrained.pth", help="")
    
    # flags
    parser.add_argument('-batch_batch', action="store_true", default=False, help='calc in batch in batch')
    parser.add_argument('-use_tensorboard', action="store_true", default=False, help='calc in batch in batch')
    
    args = parser.parse_args()
    
    prediction(args)
    
