"""
    Just for beauty of form.
"""
import argparse

import train
import prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # train settings
    parser.add_argument('--voc_image_dir', type=str, default='./dataset/voc/img', help='directory for train images\ndefault is ./dataset/voc/img')
    parser.add_argument('--voc_mask_dir', type=str, default='./dataset/voc/mask', help='directory for train mask images\ndefault is ./dataset/voc/mask')
    parser.add_argument('--sbd_image_dir', type=str, default='./dataset/sbd/img', help='directory for train images\ndefault is ./dataset/sbd/img')
    parser.add_argument('--sbd_mask_dir', type=str, default='./dataset/sbd/mask', help='directory for train mask images\ndefault is ./dataset/sbd/mask')

    parser.add_argument('--voc_train_image_list', type=str, default='./dataset/voc/train.txt', help='directory of image list of train\ndefault is ./dataset/voc/train.txt')
    parser.add_argument('--voc_trainval_image_list', type=str, default='./dataset/voc/trainval.txt', help='directory of image list of trainval\ndefault is ./dataset/voc/trainval.txt')
    parser.add_argument('--voc_val_image_list', type=str, default='./dataset/voc/val.txt', help='directory of image list of validation\ndefault is ./dataset/voc/val.txt')
    parser.add_argument('--sbd_train_image_list', type=str, default='./dataset/sbd/train.txt', help='directory of image list of train\ndefault is ./dataset/sbd/train.txt')
    parser.add_argument('--sbd_val_image_list', type=str, default='./dataset/sbd/val.txt', help='directory of image list of val\ndefault is ./dataset/sbd/val.txt')

    # prediction setting
    parser.add_argument('--image_dir', type=str, default='./dataset/test', help='directory for test images\ndefault is ./dataset/test')

    # detail options
    parser.add_argument('--crop_size', type=int, default=321, help='size for image after processing\ndefault is 321') # paper default
    parser.add_argument('--save_dir', type=str, default="./log/", help='dir of saving log and model parameters and so on.\ndefault is ./log/')
    parser.add_argument('--save_every', type=int, default=10, help='count of saving model\ndefault is 10')
    parser.add_argument('--epochs', type=int, default=200, help="train epoch num.\ndefault is 200")
    parser.add_argument('--batch_size', type=int, default=10, help="mini batch size\ndefault is 10") # paper default
    parser.add_argument('--batch_batch_size', type=int, default=64, help="the size of 'mini batch' of 'mini batch'\ndefault is 64") # not ready yet
    parser.add_argument('--num_workers', type=int, default=8, help="worker num. of data loader\ndefault is 8")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="initial value of learning rate\ndefault is 0.001")
    parser.add_argument('--decay_value', type=float, default=0.1, help="decay learning rate with count of args:decay_every in this factor.\ndefault is 0.1")
    parser.add_argument('--decay_every', type=int, default=50, help="count of decaying learning rate\ndefault is 50")
    parser.add_argument('--gpu_device_num', type=int, default=0, help="device number of gpu\ndefault is 0")
    parser.add_argument('--trainval_every', type=int, default=10, help="evaluate trainval data acc.\ndefault is 10")
    
    parser.add_argument('--trained_path', type=str, default="vgg16-397923af.pth", help="restore parameter or use pretrained model, pytorch official pretrained data is like vgg16-397923af.pth\ndefault is vgg16-397923af.pth")
    parser.add_argument('--setting_file', type=str, default=None, help="use arguments from setting file\ndefault is None") # not ready yet
    
    # flags
    parser.add_argument('-batch_batch', action="store_true", default=False, help='calc in batch in batch')
    parser.add_argument('-use_tensorboard', action="store_true", default=False, help='use TensorBoard') # not ready yet
    parser.add_argument('-predict', action="store_true", default=False, help='run with prediction')
    
    args = parser.parse_args()
    
    if args.predict:
        prediction.prediction(args)
    else:
        train.train(args)