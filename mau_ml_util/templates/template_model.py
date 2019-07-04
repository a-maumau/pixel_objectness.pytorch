import abc
import traceback

import torch
import torch.nn as nn
import torch.utils.data as data

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
