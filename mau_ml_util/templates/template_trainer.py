import abc

import torch

from tqdm import tqdm

from ..train_logger import TrainLogger

CPU = torch.device('cpu')

class Template_Trainer:
    __metaclass__ = abc.ABCMeta

    tqdm_ncols = 100

    @abc.abstractmethod
    def validate(self):
        """
            in here, model should set at eval mode
            model.eval()

            after evaliation, set train mode
            model.train()
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError()

    def get_train_logger(self, namespaces, save_dir="./", save_name="log", arguments=[],
                           use_http_server=False, use_msg_server=False, notificate=False,
                           visualize_fetch_stride=1, http_port=8080, msg_port=8081):
        # saving directory can get with save_dir = tlog.log_save_path
        tlog = TrainLogger(log_dir=save_dir, log_name=save_name, namespaces=namespaces,
                           arguments=arguments, notificate=notificate, suppress_err=True, visualize_fetch_stride=visualize_fetch_stride)
        if use_http_server:
            tlog.start_http_server(bind_port=http_port)

        if use_msg_server:
            tlog.start_msg_server(bind_port=msg_port)

        return tlog

    def get_argparse_arguments(self, args):
        return args._get_kwargs()

    def format_tensor(self, x, requires_grad=True, map_device=CPU):
        if not requires_grad:
            x = x.to(map_device).detach()
        else:
            x = x.to(map_device)

        return x

    def map_on_gpu(self, model, gpu_device_num=0):
        if torch.cuda.is_available():
            # for cpu, it is 'cpu', but default mapping is cpu.
            # so if you want use on cpu, just don't call this
            map_device = torch.device('cuda:{}'.format(gpu_device_num))
            model = model.to(map_device)

    def decay_learning_rate(self, optimizer, decay_value):
        for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_value

    def to_tqdm(self, loader, desc=""):
        return tqdm(loader, desc=desc, ncols=self.tqdm_ncols)
