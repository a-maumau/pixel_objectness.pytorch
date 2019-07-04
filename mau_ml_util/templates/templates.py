import abc
import traceback

import torch
import torch.nn as nn
import torch.utils.data as data

from tqdm import tqdm

from mau_ml_util.train_logger import TrainLogger

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

    def format_tensor(self, x, requires_grad=True, nogpu=False, gpu_device_num=0):
        if torch.cuda.is_available() and not nogpu:
            if not requires_grad:
                x = x.cuda(gpu_device_num).detach()
            else:
                x = x.cuda(gpu_device_num)
        else:
            if not requires_grad:
                x = x.detach()

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

class Template_SegmentationDataLoader(data.Dataset):
    def __init__(self, img_root, mask_root, img_list_path=None,
                       pair_transform=None, input_transform=None, target_transform=None,
                       load_all_in_ram=True, img_ext=".jpg", mask_ext=".png", return_original=False):
        """
            args:
                img_root: str
                    root directory of images.

                mask_root: str
                    root directory of mask images.

                img_list_path: str
                    path to the file which is written a image name.
                    if this is "not" None, it will only use this image written in this file.
                    it is considered to be like
                        img_001
                        img_002
                        img_003
                        .
                        .
                        .
                      in the file.
                    if this is None, it will read all file in the img_root directory.
                      in this scenario, if you set load_all_in_ram=False, it might raise some
                      errors if there is a non opneable file with PIL in the directory or no pairs.
                    setting the option of img_exr, or mask_ext to use different extensions.

                pair_transform: function
                    function that compose transform to PIL.Image object for image and mask.
                    this function must take 2 PIL.Image object which is (image, mask).
                    if it is None, nothing will be done.

                input_transform: function
                    function that compose transform to PIL.Image object for image.
                    torchvision.transforms is considered as a typical function.
                    if it is None, transforms.ToTensor will only be performed.

                target_transform: function
                    function that compose transform to PIL.Image object for mask.
                    torchvision.transforms is considered as a typical function.
                    if it is None, it will convert to torch.LongTensor.

                load_all_in_ram: bool
                    if this is True. the all dataset image will be loaded on the memory.
                    if you cause no memory problem, you can set this to False,
                     and this loader will only load the file paths at the initial moment.

                img_ext: str
                    extension for image.
                mask_ext: str
                    extension for mask image.
        """

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.pair_transform = pair_transform
        self.load_all_in_ram = load_all_in_ram
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.return_original = return_original

        # all images must have pairs
        if img_list_path is None:
            name_list = []
            image_list = os.listdir(os.path.join(img_root))
            for name in image_list:
                name_list.append(name.replace(img_ext, "").replace(mask_ext, ""))

            image_list = list(set(*name_list))

        else:
            with open(os.path.join(img_list_path), "r") as file:
                image_list = file.readlines()
                image_list = [img_name.rstrip("\n") for img_name in image_list]

        self.image_names = image_list

        self.imgs = []
        self.mask_imgs = []

        for img_name in self.image_names:
            try:
                if load_all_in_ram:
                    _img = Image.open(os.path.join(img_root, img_name+self.img_ext)).convert('RGB')
                    _mask_img = Image.open(os.path.join(mask_root, img_name+self.mask_ext)).convert('P')
                else:
                    _img = os.path.join(img_root, img_name+self.img_ext)
                    _mask_img = os.path.join(mask_root, img_name+self.mask_ext)


                self.imgs.append(_img)
                self.mask_imgs.append(_mask_img)

            except Exception as e:
                print(e)
                print("pass {}".format(img_name))

            self.data_num = len(self.imgs)
    
    def __getitem__(self, index):
        if self.load_all_in_ram:
            img = self.imgs[index]
            mask = self.mask_imgs[index]
        else:
            img = Image.open(self.imgs[index]).convert('RGB')
            mask = Image.open(self.mask_imgs[index]).convert('P')

        if self.pair_transform is not None:
            _img, _mask_img = self.pair_transform(img, mask)
        else:
            _img = img
            _mask_img = mask

        if self.return_original:
            original_img = _img.copy()
                
        if self.input_transform is not None:
            _img = self.input_transform(_img)
        else:
            _img = torch.from_numpy(np.asarray(_img).transpose(2,0,1)).type(torch.FloatTensor)
                
        if self.target_transform is not None:
            _mask_img = self.target_transform(_mask_img)
        else:
            _mask_img = torch.from_numpy(np.asarray(_mask_img)).type(torch.LongTensor)

        if self.return_original:
            return _img, _mask_img, torch.from_numpy(np.asarray(original_img)).type(torch.LongTensor)

        return _img, _mask_img

    def __len__(self):
        return self.data_num

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
