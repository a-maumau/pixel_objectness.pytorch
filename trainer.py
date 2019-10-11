import os
import math
import argparse
from datetime import datetime

import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from PIL import Image

import data_loader
from mau_ml_util.train_logger import TrainLogger
#from mau_ml_util.metric import SegmentationMetric
from metric_from_latest_mmu import SegmentationMetric
from templates import Template_Trainer

torch.backends.cudnn.benchmark = True

class ColorMap(object):
    def __init__(self, base_color=[[0,0,1], [0,1,1], [0,1,0], [1,1,0], [1,0,0]]):
        """
            color_points: list of [int, int, int]
                each value of component represent R,G,B.
        """

        self.base_color = base_color
        self.num_color_min1 = len(self.base_color)-1

    def __call__(self, val):
        return self.to_colormap(val)

    def to_colormap(self, val):
        """
            returns tpule of (R,G,B) value in range [0,1].
        """

        fract_between = 0

        if val <= 0:
            idx1 = idx2 = 0
        elif val >= 1:
            idx1 = idx2 = self.num_color_min1
        else:
            val = val * (self.num_color_min1)
            idx1  = math.floor(val);
            idx2  = idx1+1;
            fract_between = val - idx1
        
        r = (self.base_color[idx2][0] - self.base_color[idx1][0])*fract_between + self.base_color[idx1][0]
        g = (self.base_color[idx2][1] - self.base_color[idx1][1])*fract_between + self.base_color[idx1][1]
        b = (self.base_color[idx2][2] - self.base_color[idx1][2])*fract_between + self.base_color[idx1][2]

        return (r,g,b) 

class Trainer_PixelObjectness(Template_Trainer):
    def __init__(self, args, model, optimizer, lr_policy):
        self.args = args        
        self.lr_policy = lr_policy
        self.iter_wise = self.lr_policy.iteration_wise

        # for loggin the training
        val_head = ["iter" if self.iter_wise else "epoch", "mean_pixel_accuracy"]
        for i in range(self.args.class_num):
            val_head.append("mean_precision_class_{}".format(i))
        for i in range(self.args.class_num):
            val_head.append("mean_IoU_class_{}".format(i))
        self.tlog = self.get_train_logger({"train":["iter" if self.iter_wise else "epoch", "batch_mean_total_loss"], "val":val_head},
                                          save_dir=self.args.save_dir, save_name=self.args.save_name, arguments=self.get_argparse_arguments(self.args),
                                          use_http_server=self.args.use_http_server, use_msg_server=self.args.use_msg_server, notificate=False,
                                          visualize_fetch_stride=self.args.viz_fetch_stride, http_port=self.args.http_server_port, msg_port=self.args.msg_server_port)
        


        # paths
        self.save_dir = self.tlog.log_save_path
        self.model_param_dir = self.tlog.mkdir("model_param")

        if torch.cuda.is_available() and not self.args.nogpu:
            self.map_device = torch.device('cuda:{}'.format(self.args.gpu_device_num))
        else:
            self.map_device = torch.device('cpu')

        self.model = model
        if torch.cuda.is_available() and not args.nogpu:
            self.model = self.model.to(self.map_device)

        self.optimizer = optimizer

        self.train_loader = data_loader.get_train_loader(self.args, [(0.5, 0.5, 0.5),(0.5, 0.5, 0.5)])#[(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)])
        self.val_loader = data_loader.get_val_loader(self.args, [(0.5, 0.5, 0.5),(0.5, 0.5, 0.5)])

        self.cmap = self._gen_cmap()

        if self.args.show_parameters:
            for idx, m in enumerate(model.modules()):
                print(idx, '->', m)
            print(args)

        print("\nsaving at {}\n".format(self.save_dir))

    # PASCAL VOC color maps
    # borrowed from https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    def _gen_cmap_voc(self, class_num=255):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        cmap = np.zeros((class_num+1, 3), dtype='uint8')
        for i in range(class_num+1):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        return cmap

    def _gen_cmap(self, max_value=255):
        mapper = ColorMap()
        cmap = []

        for v in range(max_value+1):
            cmap.append(np.uint8(np.array(mapper(v/max_value))*255))

        return cmap

    def convert_to_color_map(self, img_array, color_map=None, class_num=255):
        """
            img_array: numpy.ndarray
                shape must be (width, height)
        """

        if color_map is None:
            color_map = self._gen_cmap()

        new_img = np.empty(shape=(img_array.shape[0], img_array.shape[1], 3), dtype='uint8')

        for c in range(class_num+1):
            index = np.where(img_array == c)
            new_img[index] = color_map[c]

        return new_img

    def validate(self, count):
        with torch.no_grad():
            self.model.eval()

            # logging
            pix_acc = 0.0
            precision_class = []
            jaccard_class = []

            #data_count_precision = [0 for i in range(self.args.class_num)]
            #data_count_jaccard = [0 for i in range(self.args.class_num)]
            
            metric = SegmentationMetric(self.args.class_num, map_device=self.map_device)

            if self.args.quiet:
                _trainval_loader = self.val_loader
            else:
                _trainval_loader = self.to_tqdm(self.val_loader, desc="train val")

            for b, (image, mask, original_image) in enumerate(_trainval_loader):
                batch_size = image.shape[0]

                img = self.format_tensor(image, requires_grad=False, map_device=self.map_device)
                mask = self.format_tensor(mask, requires_grad=False, map_device=self.map_device)

                outputs, prob_maps = self.model.inference(img)
                outputs = F.interpolate(outputs, size=[self.args.crop_size, self.args.crop_size], mode='bilinear', align_corners=False)
                prob_maps = F.interpolate(prob_maps, size=[self.args.crop_size, self.args.crop_size], mode='bilinear', align_corners=False)

                metric(outputs, mask)
                 
                # save only few batch for sample
                if b < 1:
                    self.tlog.setup_output("{}_{}_batch_{}_sample".format("iter" if self.iter_wise else "epoch", count, b))

                    # test color image
                    #test_img = np.ones((256,256))
                    #for i in range(256):
                    #    test_img[i] = test_img[i]*i
                    #    
                    #self.tlog.pack_output(Image.fromarray(self.convert_to_color_map(np.uint8(test_img))))
                    
                    for n in range(batch_size):
                        self.tlog.pack_output(Image.fromarray(np.uint8(original_image[n].detach().numpy())))

                        pred_img = np.uint8(outputs[n].squeeze(0).cpu().detach().numpy())
                        prob_img = prob_maps[n].squeeze(0).cpu().detach().numpy()
                        self.tlog.pack_output(Image.fromarray(pred_img*255), not_in_schema=True)
                        self.tlog.pack_output(Image.fromarray(self.convert_to_color_map(np.uint8(prob_img[1]*255), self.cmap)))

                        gt_img = np.uint8(mask[n].cpu().detach().numpy())
                        self.tlog.pack_output(Image.fromarray(gt_img*255), not_in_schema=True)

                        self.tlog.pack_output(None, " ")

            self.tlog.pack_output(None, "validation sample", ["left: input", "center: pred cmap", "right: output mask"])
            self.tlog.flush_output()

            pix_acc = metric.calc_pix_acc()
            precision = metric.calc_mean_precision()
            jaccard_index = metric.calc_mean_jaccard_index()

            # might I should return the non evaluated class with nan and filter the list
            # by filter(lambda n: n!=float("nan"), items)

            for class_id in range(self.args.class_num):
                precision_class.append(precision["class_{}".format(class_id)])
                jaccard_class.append(jaccard_index["class_{}".format(class_id)])

                #data_count_precision[class_id] += len(precision["class_{}".format(str(class_id))])
                #data_count_jaccard[class_id] += len(jaccard_index["class_{}".format(str(class_id))])

            # logging, this implementation is not caring missing value
            #mean_precision_classes = [y/x if x > 0 else 0 for y, x in zip(precision_class, data_count_precision)]
            #mean_iou_classes = [y/x if x > 0 else 0 for y, x in zip(jaccard_class, data_count_jaccard)]
            
            # clac. with out background
            log_msg_data = [count, pix_acc, np.mean(precision_class[1:]), np.mean(jaccard_class[1:])]

            self.tlog.log("val", [count, pix_acc]+precision_class+jaccard_class)
            self.tlog.log_message("[{}] mean pix acc.:{:.5f}, precision:{:.5f}, IoU:{:.5f}".format(*log_msg_data), "LOG", "validation")

            if not self.args.quiet:
               tqdm.write("[{}] mean pix acc.:{:.5f}, precision:{:.5f}, IoU:{:.5f}".format(*log_msg_data))

            self.model.train()

    def train(self):
        train_finish = False
        
        if self.args.quiet:
            epochs = range(1, self.args.epochs+1)
        else:
            epochs = self.to_tqdm(range(1, self.args.epochs+1), desc="train")

        curr_iter = 0
        epoch = 0

        total_loss = 0.0
        data_num = 0

        # for epoch wise and iter wise
        decay_arg = {"curr_iter":curr_iter, "curr_epoch":epoch}

        for epoch in epochs:
            if not self.iter_wise:
                total_loss = 0.0
                data_num = 0

            if self.args.quiet:
                _train_loader = self.train_loader
            else:
                _train_loader = self.to_tqdm(self.train_loader)

            for img, mask in _train_loader:
                # loss log will be showed in size averaged
                data_num += 1

                self.optimizer.zero_grad()

                images = self.format_tensor(img, map_device=self.map_device)
                masks = self.format_tensor(mask, map_device=self.map_device)

                output = self.model(images)
                output = F.interpolate(output, size=[self.args.crop_size, self.args.crop_size], mode='bilinear', align_corners=False)

                batch_loss = self.model.loss(output, masks)
                total_loss += batch_loss.item()
                
                batch_loss.backward()
                self.optimizer.step()

                curr_iter += 1

                if not self.args.quiet:
                    _train_loader.set_description("{: 3d}: train[{}] loss: {:.5f}".format(curr_iter if self.iter_wise else epoch, self.args.save_name, total_loss/data_num))

                if self.iter_wise:
                    self.lr_policy.decay_lr(**decay_arg)
                    
                    if curr_iter % self.args.trainval_every == 0:
                        self.validate(curr_iter)

                    if curr_iter % self.args.save_every == 0:
                        state = {'iter': curr_iter,
                                 'optimizer_state_dict' : self.optimizer.state_dict()}
                        self.model.save(add_state=state, file_name=os.path.join(self.model_param_dir,'model_param_iter{}.pth'.format(curr_iter)))
                
                        self.tlog.log_message("[iter:{}] model saved.".format(curr_iter), "LOG", "train")

                    if curr_iter % self.args.log_every == 0:
                        if not self.args.quiet:
                            tqdm.write("[#{: 3d}] {} iter mean loss: {:.5f}".format(curr_iter, self.args.log_every, total_loss/data_num))
                            
                        self.tlog.log("train", [curr_iter, float(total_loss/data_num)])
                        self.tlog.log_message("[{}] {} iter mean loss:{:.5f}".format("iter:{}".format(curr_iter), self.args.log_every, float(total_loss/data_num)), "LOG", "train")

                        total_loss = 0
                        data_num = 0

                if curr_iter == self.args.max_iter:
                    train_finish = True
                    _train_loader.close()
                    break
                
            if train_finish:
                epochs.close()
                break

            if not self.iter_wise:
                if not self.args.quiet:
                    tqdm.write("[# {: 3d}] batch mean loss: {:.5f}".format(epoch, total_loss/data_num))
                
                if epoch % self.args.log_every == 0:
                    self.tlog.log("train", [epoch, float(total_loss/data_num)])
                    self.tlog.log_message("[{}] batch mean loss:{:.5f}".format("epoch:{}".format(epoch), float(total_loss/data_num)), "LOG", "train")

                # check train validation
                if epoch % self.args.trainval_every == 0:
                    self.validate(epoch)

                self.lr_policy.decay_lr(**decay_arg)
                #if epoch % self.args.decay_every == 0:
                #    for param_group in self.optimizer.param_groups:
                #        param_group['lr'] *= self.args.decay_value
                #
                #    self.tlog.log_message("[epoch:{}] decay learning rate by {}".format(epoch, self.args.decay_value), "LOG", "train")
                
                # save model
                if epoch % self.args.save_every == 0:
                    state = {'epoch': epoch,
                             'optimizer_state_dict' : self.optimizer.state_dict()}
                    self.model.save(add_state=state, file_name=os.path.join(self.model_param_dir,'model_param_e{}.pth'.format(epoch)))
                    
                    self.tlog.log_message("[epoch:{}] model saved.".format(epoch), "LOG", "train")

        self.model.save(add_state={'optimizer_state_dict' : self.optimizer.state_dict()},
                        file_name=os.path.join(self.model_param_dir, 'model_param_fin_{}.pth'.format(datetime.now().strftime("%Y%m%d_%H-%M-%S"))))

        print("data is saved at {}".format(self.save_dir))

    def test_loader(self):
        from matplotlib import pylab as plt
        import time

        if self.args.quiet:
            epochs = range(1, self.args.epochs+1)
        else:
            epochs = self.to_tqdm(range(1, self.args.epochs+1), desc="train")

        for epoch in epochs:
            if self.args.quiet:
                _train_loader = self.train_loader
            else:
                _train_loader = self.to_tqdm(self.train_loader)

            for img, mask in _train_loader:
                batch_size = img.shape[0]

                img = img.numpy()
                mask = mask.numpy()

                for i in range(batch_size):
                    _img = np.uint8(img[i]*255).transpose(1,2,0)
                    _mask = self.convert_to_color_map(np.uint8(mask[i]), self.cmap)

                    merged_img = np.concatenate([_img, _mask], axis=1)

                    plt.imshow(merged_img)
                    plt.show()

