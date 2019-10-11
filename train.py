import argparse

import torch

from trainer import Trainer_PixelObjectness as Trainer
from templates import gen_policy_args
from learning_rate_policy import *

def train(args):
    if args.model == "vgg":
        from models.vgg_po import VGG16_PixelObjectness as Model
        model = Model(input_channel=3, num_class=args.class_num)
        model.load_imagenet_param(args.pretrained_path)

    # not working
    elif args.model == "resnet":
        from models.resnet_po import resnet_po as Model
        model = Model(num_classes=args.class_num, model_type="101", imagenet_pretrained_path=args.pretrained_path)
      
    else:
        raise("invalid model name.")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    lr_policy = StepBasedPolicy(**gen_policy_args(optimizer=optimizer, args=args))
    
    trainer = Trainer(args, model, optimizer, lr_policy)

    if args.test_loader:
        trainer.test_loader()
    else:
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--train_dataset', type=str, default='./train.pkl', help='path of pickled train data')
    parser.add_argument('--val_dataset', type=str, default='./val.pkl', help='path of pickled train data ')

    parser.add_argument('--save_name', type=str, default="po_log", help='name of log')
    parser.add_argument('--save_dir', type=str, default="./log/", help='dir of saving log and model parameters and so on')

    # model name
    #parser.add_argument('--model', type=str, default="vgg", choices=["vgg", "resnet"], help='')
    # I didn't finish it, so only vgg
    parser.add_argument('--model', type=str, default="vgg", choices=["vgg"], help='')

    # model setting
    parser.add_argument('--class_num', type=int, default=2, help="output map channel")

    # data augments settings
    parser.add_argument('--crop_size', type=int, default=321, help='size for image after processing')
    parser.add_argument('--resize_size', type=int, default=360, help='size for image after processing')
    #parser.add_argument('--resize_scale_min', type=float, default=0.8, help='')
    #parser.add_argument('--resize_scale_max', type=float, default=1.2, help='')

    parser.add_argument('--rotate_deg', type=int, default=10, help='rotation degree in the augmentation')

    parser.add_argument('--blur_radius', type=float, default=0.8, help='')
    parser.add_argument('--blur_scale_max', type=float, default=1.2, help='')
    parser.add_argument('--blur_scale_min', type=float, default=0.5, help='')
    parser.add_argument('--blur_prob', type=float, default=0.8, help='')

    # train, data setting
    parser.add_argument('--epochs', type=int, default=16, help="how many epochs to train. if your are using iter wise training this should be set enough number")
    parser.add_argument('--max_iter', type=int, default=12500, help="train iter max num.")
    
    parser.add_argument('--batch_size', type=int, default=10, help="mini batch size, original is 10")
    parser.add_argument('--num_workers', type=int, default=8, help="worker number of data loader")
    
    parser.add_argument('--learning_rate', type=float, default=0.001, help="initial value of learning rate")
    parser.add_argument('--min_learning_rate', type=float, default=0.001, help="initial value of learning rate")
    parser.add_argument('--lr_decay_power', type=float, default=0.9, help="count of decaying learning rate")
    parser.add_argument('--decay_value', type=float, default=0.1, help="decay learning rate with count of args:decay_every in this factor.")
    parser.add_argument('--lr_hp_k', type=float, default=1.0, help="")

    # nums
    parser.add_argument('--decay_every', type=int, default=2500, help="count of decaying learning rate")
    parser.add_argument('--save_every', type=int, default=2500, help='count of saving model')
    parser.add_argument('--trainval_every', type=int, default=2500, help="evaluate trainval data acc.")
    parser.add_argument('--log_every', type=int, default=1250, help="count of showing log")

    # gpu number
    parser.add_argument('--gpu_device_num', type=int, default=0, help="device number of gpu")
    
    # trained path
    parser.add_argument('--pretrained_path', type=str, default="vgg16-397923af.pth",
                        help="restore parameter or use pretrained model\npytorch official pretrained parameter is like resnet34-333f7ec4.pth")

    # setting of visualization
    parser.add_argument('--viz_fetch_stride', type=int, default=1, help="")
    parser.add_argument('--http_server_port', type=int, default=8080, help="")
    parser.add_argument('--msg_server_port', type=int, default=8081, help="")

    # notificate type
    parser.add_argument('--notify_type', type=str, nargs='*', default=["slack"], help="you can pick up multiple type from [slack mail twitter]")

    # for batching the training
    parser.add_argument('--train_type', type=int, default=0, help="")

    # option
    parser.add_argument('-nogpu', action="store_true", default=False, help="don't use gpu")
    parser.add_argument('-show_parameters', action="store_true", default=False, help='show model parameters')
    parser.add_argument('-quiet', action="store_true", default=False, help='only showing the log of loss and validation')
    parser.add_argument('-use_http_server', action="store_true", default=False, help='')
    parser.add_argument('-use_msg_server', action="store_true", default=False, help='')

    parser.add_argument('-name_with_flag', action="store_true", default=False, help='')

    parser.add_argument('-test_mode', action="store_true", default=False, help='')
    parser.add_argument('-test_loader', action="store_true", default=False, help='')
    # this option should be manually set, but it is bother us. so I set to True which make alway iterwise lr policy
    parser.add_argument('-force_lr_policy_iter_wise', action="store_true", default=True, help='')
    parser.add_argument('-force_lr_policy_epoch_wise', action="store_true", default=False, help='')

    args = parser.parse_args()

    # debug mode
    if args.test_mode:
        args.epochs = 1
        args.save_every = 1
        args.trainval_every = 1
        args.save_name = "_test_running"
        args.save_dir = "./_test"

    if args.test_loader:
        args.epochs = 10
        args.save_every = 1
        args.trainval_every = 1
        args.save_name = "_test_running"
        args.save_dir = "./_test"
    
    train(args)
