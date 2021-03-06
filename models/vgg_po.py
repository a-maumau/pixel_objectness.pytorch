import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
"""
originate from
https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

pretrained data.
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
"""

"""
# base net
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
def vgg16(pretrained=False, **kwargs):
    #""VGG 16-layer model (configuration "D")
    #Args:
    #    pretrained (bool): If True, returns a model pre-trained on ImageNet
    #""
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


which will make

nn.Conv2d(3, 64, stride=1, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=2, stride=2), # diff

nn.Conv2d(64, 128, stride=1, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.Conv2d(128, 128, stride=1, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=2, stride=2), #diff

nn.Conv2d(128, 256, stride=1, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.Conv2d(256, 256, stride=1, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.Conv2d(256, 256, stride=1, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=2, stride=2), #diff

nn.Conv2d(256, 512, stride=1, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=2, stride=2, # diff

nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=2, stride=2), #diff
"""

# model of Pixel Objectness which is based on VGG Net
# this is a model which was written in the paper
class VGG16_PixelObjectness(nn.Module):
    inplace_flag = False
    def __init__(self, input_channel=3, num_class=2, init_weights=True):
        super(VGG16_PixelObjectness, self).__init__()

        self.input_channel = input_channel
        self.num_class = num_class
        
        self.loss_function = nn.NLLLoss()

        # I hate writing in for-loops.
        # Name must be "features" because in the pretrained data's dict is named "features"
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channel, 64, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            nn.Conv2d(128, 128, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 256, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            nn.Conv2d(256, 256, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            nn.Conv2d(256, 256, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 512, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            #nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1), # this part must be a dilated convolution? is equal?
            nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=2, dilation=2), # I don't know this part is correct

            nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            #nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1), # this part must be a dilated convolution? is equal?
            nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=2, dilation=2), # I don't know this part is correct

            nn.Conv2d(512, 1024, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            nn.Dropout(inplace=self.inplace_flag),
            nn.Conv2d(1024, 1024, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=self.inplace_flag),
            nn.Dropout(inplace=self.inplace_flag),
            nn.Conv2d(1024, self.num_class, stride=1, kernel_size=1, padding=1)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def loss(self, inputs, targets):
        return self.loss_function(F.log_softmax(inputs, dim=1), targets)

    def inference(self, inputs):
        # same thing
        prob_map = torch.nn.functional.softmax(self.features(inputs), dim=1)
        return torch.max(torch.nn.functional.softmax(prob_map, dim=1), dim=1)[1].unsqueeze(1).type(torch.float), prob_map

    # for loading ImageNet pretrained model parameter from pytorch official.
    def load_imagenet_param(self, parameter_path, print_debug=False):
        if parameter_path is not None:
            try:
                print("loading pretrained parameter... ", end="")
                
                chkp = torch.load(os.path.abspath(parameter_path))
                _model_dict = self.features.state_dict()

                # rename the name, name , original names is like "layer3.0.conv1.weight"
                pretrained_dict = {k: v for k, v in chkp.items() if k in _model_dict}

                if print_debug:
                    print("")
                    print(pretrained_dict.keys())

                _model_dict.update(pretrained_dict)
                self.features.load_state_dict(_model_dict)
                
                print("done.")

            except Exception as e:
                print("")
                import traceback
                traceback.print_exc()
                print(e)
                print("cannot load pretrained data.")

    # for loading trained parameter of this model.
    def load_trained_param(self, parameter_path, print_debug=False):
        if parameter_path is not None:
            try:
                print("loading pretrained parameter... ", end="")
                
                chkp = torch.load(os.path.abspath(parameter_path), map_location=lambda storage, loc: storage)

                if print_debug:
                    print(chkp.keys())

                self.load_state_dict(chkp["state_dict"])
                
                print("done.")

            except Exception as e:
                print("")
                import traceback
                traceback.print_exc()
                print(e)
                print("cannot load pretrained data.")

    def save(self, add_state={}, file_name="model_param.pth"):
        #assert type(add_state) is dict, "arg1:add_state must be dict"
        
        if "state_dict" in add_state:
            print("the value of key:'state_dict' will be over written by model's state_dict parameters")

        _state = add_state
        _state["state_dict"] = self.state_dict()
        
        try:
            torch.save(_state, file_name)
        except:
            torch.save(self.state_dict(), "./model_param.pth.tmp")
            print("save_error.\nsaved at ./model_param.pth.tmp only model params.")

    def _initialize_weights(self, all=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
