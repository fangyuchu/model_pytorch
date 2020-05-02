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
    def __init__(self,net,net_name,dataset_name,flop_expected,feature_len=15,gcn_rounds=2,only_gcn=False,only_inner_features=False,mask_update_freq=4000,mask_update_steps=400):
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
        self.extractor=filter_feature_extractor.extractor(feature_len=feature_len,gcn_rounds=gcn_rounds,only_gcn=only_gcn,only_inner_features=only_inner_features)
        self.net_name=net_name
        self.dataset_name=dataset_name
        self.feature_len=feature_len
        self.gcn_rounds=gcn_rounds
        # self.data_parallel=True
        self.mask_update_freq=mask_update_freq
        self.mask_update_steps=mask_update_steps
        self.step_tracked=0
        self.net=self.transform(net)#.to(device)
        self.flop_expected=flop_expected


    def train(self, mode=True):
        super().train(mode)
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
                    mod.reset_running_stats()  # reset all tracked value
                mod.track_running_stats = track

    def train_mask(self):
        #update mask and track_running_stats in BN according to current step
        if self.training:
            if self.step_tracked % self.mask_update_freq <= self.mask_update_steps:  # mask need to be trained
                if self.step_tracked % self.mask_update_freq == 1:
                    print('{} start updating the mask.'.format(datetime.now()))
                    self.print_mask()
                    self.track_running_stats(track=False)  # don't track stats on BN when updating the mask to avoid volatility
                self.update_mask()
            else:  # mask will not be trained. Only net will be trained
                if self.step_tracked % self.mask_update_freq == self.mask_update_steps + 1:
                    print('{} stop updating the mask.'.format(datetime.now()))
                    self.print_mask()
                    self.track_running_stats(track=True)  # track stats on BN when mask is not updated
                self.detach_mask()
        else:
            raise Exception('Masks should not be trained or updated in evaluation mode.')

    def update_mask(self):
        if self.training is False:
            raise Exception('Masks should not be updated in evaluation mode.')
        mask = self.extractor(self, self.net_name, self.dataset_name)  # predict mask using extractor

        prune_rate=self.find_prune_rate(mask)
        _, mask_index = torch.topk(torch.abs(mask), k=int(prune_rate * mask.shape[0]), dim=0, largest=False)
        mask[mask_index] = 0

        lo = hi = 0
        last_conv_mask = None
        layer=0
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
                layer+=1
                hi += mod.out_channels
                _modules = get_module(model=self.net, name=name)
                mod.mask = mask[lo:hi].view(-1)  # update mask for each conv
                lo = hi
                last_conv_mask = mod.mask
            else:
                if isinstance(mod, nn.BatchNorm2d) and last_conv_mask is not None:
                    # prune the corresponding mean and var according to mask
                    # todo:用gcn提取的时候需要考虑bn的影响吗？
                    mod.running_mean[last_conv_mask == 0] = 0
                    mod.running_var[last_conv_mask == 0] = 1
                last_conv_mask = None

    def find_prune_rate(self, mask):
        '''
        use binary search to determine the number of filters to prune for a given flops
        :param mask:
        :return:
        '''
        prune_rate_hi = 1
        prune_rate_lo = 0
        pruned_filters = mask.shape[0]  # number of filters being pruned

        while True:
            prune_rate = (prune_rate_hi + prune_rate_lo) / 2
            mask_tmp = mask.clone().detach()
            last_pruned_filters = pruned_filters  # number of filters being pruned in the last time
            pruned_filters = int(prune_rate * mask.shape[0])
            if pruned_filters == last_pruned_filters:  # the search reach convergence
                break

            _, mask_index = torch.topk(torch.abs(mask_tmp), k=pruned_filters, dim=0, largest=False)
            mask_tmp[mask_index] = 0
            lo = hi = 0
            layer = 0
            for name, mod in self.net.named_modules():
                if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
                    layer += 1
                    hi += mod.out_channels
                    _modules = get_module(model=self.net, name=name)
                    mod.mask = mask_tmp[lo:hi].view(-1)  # update mask for each con
                    lo = hi

            flops = measure_flops.measure_model(self, dataset_name=self.dataset_name, print_flop=False)
            if flops < self.flop_expected:  # need to prune less(a lower prune rate)
                prune_rate_hi = prune_rate
            elif flops > self.flop_expected:  # need to prune more(a higher prune rate)
                prune_rate_lo = prune_rate
            else:
                break
        return prune_rate

    def print_mask(self):
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
                channel_num = torch.sum(mod.mask != 0)
                print(channel_num)  # print the number of channels without being pruned

    def detach_mask(self):
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
                mod.mask = mod.mask.detach()  # detach masks from computation graph so the extractor will not be updated

    def forward(self, input):
        if self.training is True:
            self.step_tracked += 1
            self.train_mask()
            if self.step_tracked==1:
                self.print_mask()
        return self.net(input)



class predicted_mask_and_shortcut_net(predicted_mask_net):
    def __init__(self,net,net_name,dataset_name,flop_expected,feature_len=15,gcn_rounds=2,only_gcn=False,only_inner_features=False,mask_update_freq=4000,mask_update_steps=400):
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
        super(predicted_mask_and_shortcut_net, self).__init__(net, net_name, dataset_name,flop_expected, feature_len, gcn_rounds,
                                                              only_gcn, only_inner_features, mask_update_freq,
                                                              mask_update_steps)

    def transform(self, net):
        def hook(module, input, output):
            map_size[module] = input[0].shape[1]

        net.eval()
        map_size = {}
        handle = []
        for name, mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                device = mod.weight.device
                handle += [mod.register_forward_hook(hook)]

        if self.dataset_name is 'imagenet' or self.dataset_name is 'tiny_imagenet':
            image = torch.zeros((2, 3, 224, 224)).to(device)
        elif self.dataset_name is 'cifar10' or self.dataset_name is 'cifar100':
            image = torch.zeros((2, 3, 32, 32)).to(device)
        net(image)  # record input feature map sizes of each conv
        for h in handle:
            del h

        for name, mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                _modules = get_module(model=net, name=name)
                _modules[name.split('.')[-1]] = conv2d_with_mask_shortcut(mod, map_size[mod]).to(device)  # replace conv

        return net

    def print_mask(self):
        super().print_mask()
        measure_flops.measure_model(self.net,self.dataset_name,True)
