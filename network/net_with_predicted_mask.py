from filter_characteristic import filter_feature_extractor
from prune.prune_module import get_module
from network.modules import conv2d_with_mask,conv2d_with_mask_shortcut
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from datetime import datetime
import math
import matplotlib.pyplot as plt
from framework import data_loader, measure_flops, evaluate, train, config as conf
from math import ceil
from network import storage
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class predicted_mask_net(nn.Module):
    def __init__(self,net,net_name,dataset_name,feature_len=15,gcn_rounds=2,only_gcn=False,only_inner_features=False,mask_update_freq=4000,mask_update_steps=400):
        '''
        Use filter feature extractor to extract features from a cnn and predict mask for it. The mask will guide the
        cnn to skip filters when forwarding. Both extractor and cnn are updated through back-propagation.
        :param net:
        :param net_name:
        :param dataset_name:
        :param feature_len: expected length of features to extract
        :param gcn_rounds:
        :param only_gcn:
        :param only_inner_features:
        :param mask_update_freq: how often does the extractor being updated. The extractor will be updated every mask_update_freq STEPs!
        :param mask_update_steps: update mask for mask_update_steps STEPs
        '''
        super(predicted_mask_net, self).__init__()
        self.net=self.transform(net)#.to(device)
        self.extractor=filter_feature_extractor.extractor(feature_len=feature_len,gcn_rounds=gcn_rounds,only_gcn=only_gcn,only_inner_features=only_inner_features)
        self.net_name=net_name
        self.dataset_name=dataset_name
        self.feature_len=feature_len
        self.gcn_rounds=gcn_rounds
        # self.data_parallel=True
        self.mask_update_freq=mask_update_freq
        self.mask_update_steps=mask_update_steps
        self.step_tracked=0

    def train(self, mode=True):
        super().train(mode)
        # if mode is False:
        #     self.update_mask()  # update the mask when being switched to eval mode
        return self

    def eval(self):
        return self.train(False)

    def copy(self):
        '''
        self-made deepcopy
        :return: a cloned network
        '''
        checkpoint = storage.get_net_information(self, self.dataset_name, self.net_name)
        copied_net = storage.restore_net(checkpoint, pretrained=True)
        copied_net.to(self.extractor.network[0].weight.device)
        return copied_net

    def transform(self, net):
        for name, mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                device = mod.weight.device
                _modules = get_module(model=net, name=name)
                _modules[name.split('.')[-1]] = conv2d_with_mask(mod).to(device)  # replace conv
        return net

    def track_running_stats(self, track=True):
        for name, mod in self.named_modules():
            if 'net' in name and (isinstance(mod, nn.BatchNorm2d) or isinstance(mod, nn.BatchNorm1d)):
                if track is False:
                    mod.reset_running_stats()   #reset all tracked value
                mod.track_running_stats = track
    def mask_need_update(self):
        if self.step_tracked % self.mask_update_freq <= self.mask_update_steps:
            return True
        else:
            return False
    def update_mask(self):
        mask = self.extractor(self, self.net_name, self.dataset_name)  # predict mask using extractor
        lo = hi = 0
        last_conv_mask = None
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
                hi += mod.out_channels
                _modules = get_module(model=self.net, name=name)
                mod.mask = mask[lo:hi].view(-1)  # update mask for each conv

                channel_num=torch.sum(mod.mask!=0)                #code for debug
                if self.training is True:
                    if self.step_tracked % self.mask_update_freq ==1 or  self.step_tracked % self.mask_update_freq==self.mask_update_steps:
                        print(channel_num)                                #print the number of channels without being pruned

                if torch.sum(mod.mask == 0) == mod.out_channels:
                    print(name+' has been removed')
                    print(mask.reshape(-1))
                    # raise Exception('all filters are pruned')
                lo = hi
                last_conv_mask = mod.mask
            else:
                if isinstance(mod, nn.BatchNorm2d) and last_conv_mask is not None:
                    # prune the corresponding mean and var according to mask
                    # todo:用gcn提取的时候需要考虑bn的影响吗？
                    mod.running_mean[last_conv_mask == 0] = 0
                    mod.running_var[last_conv_mask == 0] = 1
                last_conv_mask = None

    def forward(self, input):
        self.track_running_stats(track=True)
        if self.training is True:
            self.step_tracked += 1
            if self.mask_need_update():
                self.track_running_stats(track=False)  # don't track stats on BN when updating the mask to avoid volatility
                self.update_mask()  # mask only need to be updated when training.
            else:
                for name, mod in self.net.named_modules():
                    if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
                        mod.mask = mod.mask.detach()  # detach masks from computation graph so the extractor will not be updated
        return self.net(input)



class predicted_mask_and_shortcut_net(predicted_mask_net):
    def __init__(self,net,net_name,dataset_name):
        '''
        Use filter feature extractor to extract features from a cnn and predict mask for it. The mask will guide the
        cnn to skip filters when forwarding. Both extractor and cnn are updated through back-propagation.
        :param net:
        :param net_name:
        :param dataset_name:
        :param feature_len: expected length of features to extract
        :param gcn_rounds:
        :param only_gcn:
        :param only_inner_features:
        :param mask_update_freq: how often does the extractor being updated. The extractor will be updated every mask_update_freq STEPs!
        :param mask_update_steps: update mask for mask_update_steps STEPs
        '''
        super(predicted_mask_and_shortcut_net, self).__init__(net, net_name, dataset_name)
        self.net = self.transform(net, dataset_name)

    def transform(self, net, dataset_name):
        def hook(module, input, output):
            map_size[module] = input[0].shape[1]

        net.eval()
        map_size = {}
        handle = []
        for name, mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                device = mod.weight.device
                handle += [mod.register_forward_hook(hook)]

        if dataset_name is 'imagenet' or dataset_name is 'tiny_imagenet':
            image = torch.zeros((2, 3, 224, 224)).to(device)
        elif dataset_name is 'cifar10' or dataset_name is 'cifar100':
            image = torch.zeros((2, 3, 32, 32)).to(device)
        net(image)  # record input feature map sizes of each conv
        for h in handle:
            del h

        for name, mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                _modules = get_module(model=net, name=name)
                _modules[name.split('.')[-1]] = conv2d_with_mask_shortcut(mod, map_size[mod]).to(device)  # replace conv

        return net
