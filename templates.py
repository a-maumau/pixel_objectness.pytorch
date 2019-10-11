import abc
import traceback

import torch
import torch.nn as nn

from tqdm import tqdm

from mau_ml_util.train_logger import TrainLogger

CPU = torch.device('cpu')

def gen_policy_args(optimizer, args):
        policy_args = {"optimizer":optimizer}

        policy_args["initial_learning_rate"] = args.learning_rate
        policy_args["decay_epoch"] = args.decay_every
        policy_args["decay_val"] = args.decay_value
        policy_args["max_iter"] = args.max_iter
        policy_args["lr_decay_power"] = args.lr_decay_power
        policy_args["max_learning_rate"] = args.learning_rate
        policy_args["min_learning_rate"] = args.min_learning_rate
        policy_args["k"] = args.lr_hp_k

        if args.force_lr_policy_iter_wise and args.force_lr_policy_epoch_wise:
            pass
        elif args.force_lr_policy_iter_wise:
            policy_args["iteration_wise"] = True
        elif args.force_lr_policy_epoch_wise:
            policy_args["iteration_wise"] = False
        else:
            pass

        return policy_args

class Template_DecayPolicy(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, optimizer, initial_learning_rate, iteration_wise):
        """
            optimizer: torch.optim.*
                Pytorch optimizer

            initial_learning_rate: float
                initial value of the learning rate.
                this will be the base value of each policy to calculate a next learning rate

            iteration_wise: bool
                Variable for using in the step-wise decaying or epoch-wise.
                So, it doesn't actually effect the codes, it's for management.
                Default value of this depends on a inherited class.
        """

        self.optimizer = optimizer
        self.initial_learning_rate = initial_learning_rate
        self.iteration_wise = iteration_wise

    @abc.abstractmethod
    def decay_lr(self, **kwargs):
        raise NotImplementedError()

    def calc_iter_to_epoch(self, epoch_data_batch, max_iter):
        e = max_iter//epoch_data_batch

        if max_iter % epoch_data_batch != 0:
            e += 1

        return e

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
    
    @staticmethod
    def gen_policy_args(self, optimizer, args):
        return gen_policy_args(optimizer, args)

    def map_on_gpu(self, model, gpu_device_num=0):
        if torch.cuda.is_available():
            # for cpu, it is 'cpu', but default mapping is cpu.
            # so if you want use on cpu, just don't call this
            map_device = torch.device('cuda:{}'.format(gpu_device_num))
            model = model.to(map_device)

    def decay_learning_rate(self, optimizer, decay_value):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_value

    def to_tqdm(self, loader, desc="", quiet=False):
        if quiet:
            return loader

        return tqdm(loader, desc=desc, ncols=self.tqdm_ncols)

class Template_Model(nn.Module):
    __metaclass__ = abc.ABCMeta

    def initialize_weights(self, *models):
        for model in models:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.ConvTranspose2d):
                    module.weight.data.normal_(0, 0.02)
                    if module.bias is not None:
                        module.bias.data.zero_()
                    module.bias.data.zero_()

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def loss(self, inputs, targets):
        raise NotImplementedError()

    def inference(self, x):
        pass

    # for loading trained parameter of this model.
    def load_trained_param(self, parameter_path, print_debug=False):
        if parameter_path is not None:
            try:
                print("loading pretrained parameter... ", end="")
                
                chkp = torch.load(os.path.abspath(parameter_path), map_location=lambda storage, location: storage)

                if print_debug:
                    print(chkp.keys())

                self.load_state_dict(chkp["state_dict"])
                
                print("done.")

            except Exception as e:
                print("\n"+e+"\n")
                traceback.print_exc()
                print("cannot load pretrained data.")

    def save(self, add_state={}, file_name="model_param.pth"):
        #assert type(add_state) is dict, "arg1:add_state must be dict"
        
        if "state_dict" in add_state:
            print("the value of key:'state_dict' will be over write with model's state_dict parameters")

        _state = add_state
        _state["state_dict"] = self.state_dict()
        
        try:
            torch.save(_state, file_name)
        except:
            torch.save(self.state_dict(), "./model_param.pth.tmp")
            print("save_error.\nsaved at ./model_param.pth.tmp only model params.")
