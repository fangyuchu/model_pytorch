from network import vgg,resnet
from framework import config as conf
from prune.prune_module import get_module,prune_conv_layer,prune_conv_layer_resnet
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import os
from copy import deepcopy
class NetWithMask(nn.Module):
    def __init__(self,dataset_name,net_name):
        super(NetWithMask,self).__init__()
        self.dataset_name=dataset_name
        self.net_name=net_name

        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'vgg' in net_name:
            if 'imagenet' ==dataset_name:
                self.net = getattr(globals()['vgg'], net_name)(dataset_name=dataset_name,pretrained=True)
            elif 'cifar10' ==dataset_name:
                self.net = getattr(globals()['vgg'], net_name)(dataset_name=dataset_name, pretrained=False)
                c=torch.load(os.path.join(conf.root_path,'baseline/vgg16_bn_cifar10,accuracy=0.941.tar'))
                self.net.load_state_dict(c['state_dict'])
            else:
                raise Exception('Only support vgg for cifar10/ImageNet.')
        elif 'resnet' in net_name:
            if 'imagenet' == dataset_name:
                self.net = getattr(globals()['resnet'], net_name)(pretrained=True)
            else:
                raise Exception('Only support resnet for ImageNet.')
        self.net.to(device)
        self.mask=[nn.Parameter(torch.ones(3).to(device),requires_grad=False)]                                 #for rgb channels
        for name,mod in self.net.named_modules():
            if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
                self.mask+=[nn.Parameter(torch.ones(mod.out_channels).to(device),requires_grad=False)]
                _modules = get_module(model=self.net, name=name)
                _modules[name.split('.')[-1]] = Conv2dMask(mod,mask=self.mask[-1],front_mask=self.mask[-2]).to(device)             #replace conv



    def forward(self, input):
        return self.net(input)

    def mask_filters(self,layer_index,filter_index):
        '''
        prune the network in a masking way
        :param layer_index: index of conv layer that need to prune filters, start with 0
        :param filter_index: index of filters to be pruned
        :return:
        '''
        self.mask[layer_index+1][filter_index]=0                    #first element in mask is [1,1,1] for rgb

    def reset_mask(self):
        '''
        set all mask to 1
        :return:
        '''
        for mask in self.mask[1:]:
            mask[:]=1

    def prune(self):
        '''
        return a copy of the pruned network based on mask
        :return: an actual pruned network
        '''
        net=deepcopy(self.net)
        layer=0
        for mask in self.mask[1:]:
            mask = mask.detach().cpu().numpy()
            filter_index = np.argwhere(mask==0).reshape(-1).tolist()
            if 'vgg' in self.net_name:
                prune_conv_layer(net,layer,filter_index)
            elif 'resnet' in self.net_name:
                prune_conv_layer_resnet(net,layer,filter_index)
            else:
                raise Exception('Unknown net_name:'+self.net_name)
            layer+=1
        return net





class Conv2dMask(nn.Conv2d):
    def __init__(self,conv,mask,front_mask):
        super(Conv2dMask,self).__init__( conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                 padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=(conv.bias is not None))
        self.weight=conv.weight
        if self.bias is not None:
            self.bias=conv.bias
        self.mask=mask
        self.front_mask=front_mask

    def forward(self, input):
        #mask the pruned filter and channel
        masked_weight=self.weight * self.mask.detach().reshape((-1,1,1,1)) * self.front_mask.detach().reshape((1, -1, 1, 1))
        masked_bias=self.bias * self.mask.detach()
        out = F.conv2d(input, masked_weight, masked_bias, self.stride,
                       self.padding, self.dilation, self.groups)

        return out







if __name__ == "__main__":
    net=NetWithMask(dataset_name='imagenet',net_name='vgg16_bn')

    tmp=net.prune()
    net.mask_filters(layer_index=0,filter_index=[0,2,5])

    from framework import evaluate,data_loader
    dl=data_loader.create_validation_loader(batch_size=512,num_workers=2,dataset_name='cifar10')
    evaluate.evaluate_net(net,dl,save_net=False)
