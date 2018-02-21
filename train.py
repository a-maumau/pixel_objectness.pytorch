import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import argparse
from tqdm import tqdm

from model import POVGG16
from data_loader import get_loader
import pair_transforms

def train(args):
	with torch.cuda.device(args.gpu_device_num):
		model = POVGG16()
		model._initialize_weights()
		try:
			chkp = torch.load(args.trained_path)
			model_dict = model.state_dict()

			pretrained_dict = {k: v for k, v in chkp.items() if k in model_dict}
			model_dict.update(pretrained_dict)
			model.load_state_dict(model_dict)
		except Exception as e:
			import traceback
			traceback.print_exc()
			print(e)
			print("cannot load pretrained data.")

		pair_transform = pair_transforms.PairCompose([ 
		pair_transforms.PairRandomCrop(224),
		pair_transforms.PairRandomHorizontalFlip()])

		val_pair_transform = pair_transforms.PairCompose([ 
		pair_transforms.PairRandomCrop(224)
		])



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--train_image_dir', type=str, default='./data/train',
						help='directory for train images')
	parser.add_argument('--train_mask_dir', type=str, default='./data/train',
						help='directory for train mask images')
	
	parser.add_argument('--val_image_dir', type=str, default='./data/val',
						help='directory for val images')
	parser.add_argument('--val_mask_dir', type=str, default='./data/val',
						help='directory for validation mask images')
	
	parser.add_argument('--train_json_path', type=str, default='./data/json',
						help='directory of json file for training dataset')
	parser.add_argument('--val_json_path', type=str, default='./data/json',
						help='directory of json file for validation dataset')
	
	parser.add_argument('--crop_size', type=int, default=224,
						help='size for image after processing')

	parser.add_argument('--save_dir', type=str, default="./log/",
						help='size for image after processing')

	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--batch_size', type=int, default=2)
	parser.add_argument('--batch_batch_size', type=int, default=64)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--learning_rate', type=float, default=0.01)
	parser.add_argument('--gpu_device_num', type=int, default=0)
	parser.add_argument('--trained_path', type=str, default="vgg16-397923af.pth")

	parser.add_argument('-batch_batch', action="store_true", default=False, help='calc in batch in batch')
	args = parser.parse_args()
	train(args)
