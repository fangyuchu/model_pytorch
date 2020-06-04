from filter_characteristic import filter_feature_extractor
from prune.prune_module import get_module
from network.modules import conv2d_with_mask, conv2d_with_mask_shortcut, block_with_mask_shortcut
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
import os,sys


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class predicted_mask_net(nn.Module):
    def __init__(self, net, net_name, dataset_name, flop_expected, feature_len=15, gcn_rounds=2, only_gcn=False,
                 only_inner_features=False, mask_update_freq=10, mask_update_epochs=1,batch_size=128,
                 mask_training_start_epoch=10,mask_training_stop_epoch=80):
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
        :param mask_update_freq: how often does the extractor being updated. The extractor will be updated every mask_update_freq EPOCHs!
        :param mask_update_epochs: how many epochs used to update mask in each update
        '''
        super(predicted_mask_net, self).__init__()
        self.extractor = filter_feature_extractor.extractor(feature_len=feature_len, gcn_rounds=gcn_rounds,
                                                            only_gcn=only_gcn, only_inner_features=only_inner_features)
        self.net_name = net_name
        self.dataset_name = dataset_name
        self.feature_len = feature_len
        self.gcn_rounds = gcn_rounds
        # self.data_parallel=True
        self.mask_update_freq = mask_update_freq
        self.mask_update_epochs = mask_update_epochs
        self.mask_updating=False
        self.step_tracked = 0
        self.current_epoch=1
        self.net = self.transform(net)  # .to(device)
        self.flop_expected = flop_expected
        self.loader =None
        self.batch_size=batch_size
        self.mask_training_start_epoch=mask_training_start_epoch
        self.mask_training_stop_epoch=mask_training_stop_epoch
        self.set_bn_momentum(momentum=0.1)



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
            if 'net.' in name and (isinstance(mod, nn.BatchNorm2d) or isinstance(mod, nn.BatchNorm1d)):
                # if track is False:
                mod.reset_running_stats()  # reset all tracked value
                mod.track_running_stats = track

        self.net_pruned=True
    def train_mask(self):
        # update mask and track_running_stats in BN according to current step
        if self.training:
            if self.mask_training_start_epoch <= self.current_epoch <= self.mask_training_stop_epoch:
                if (self.current_epoch-self.mask_training_start_epoch) % self.mask_update_freq < self.mask_update_epochs:  # mask need to be trained
                        self.update_mask()
                        if (self.current_epoch-self.mask_training_start_epoch) % self.mask_update_freq == 0 and self.step_tracked==1:
                            print('{} start updating the mask.'.format(datetime.now()))
                            self.mask_updating = True
                            self.print_mask()
                            self.track_running_stats(track=False)  # don't track stats on BN when updating the mask to avoid volatility
                else:
                    if (self.current_epoch-self.mask_training_start_epoch) % self.mask_update_freq == self.mask_update_epochs and self.step_tracked==1:
                        print('{} stop updating the mask.'.format(datetime.now()))
                        self.mask_updating = False
                        self.print_mask()
                        self.track_running_stats(track=True)  # track stats on BN when mask is not updated
                    self.detach_mask()
            else:  # mask will not be trained. Only net will be trained
                if self.current_epoch==self.mask_training_stop_epoch+1 and self.step_tracked==1:
                    print('{} stop updating the mask since the current epoch is {}.'.format(datetime.now(),self.current_epoch))
                    self.detach_mask()
                    self.mask_updating = False
                    self.print_mask()
                    self.track_running_stats(track=True)  # track stats on BN when mask is not updated
                pass
        else:
            raise Exception('Masks should not be trained or updated in evaluation mode.')

    def update_mask(self):
        if self.training is False:
            raise Exception('Masks should not be updated in evaluation mode.')
        mask = self.extractor(self, self.net_name, self.dataset_name)  # predict mask using extractor
        mask_clone=mask.detach().clone()

        # if 'resnet' in self.net_name:#preserve all filters in first conv for resnet
        #     for name,mod in self.net.named_modules():
        #         if isinstance(mod,nn.Conv2d):
        #             out_channels=mod.out_channels
        #             mask_clone[:out_channels]=1
        #             break

        prune_rate = self.find_prune_rate(mask_clone)
        _, mask_index = torch.topk(torch.abs(mask_clone), k=int(prune_rate * mask.shape[0]), dim=0, largest=False)
        index=torch.ones(mask.shape).to(mask.device)
        index[mask_index]=0
        mask=mask*index
        # mask[mask_index] = 0

        lo = hi = 0
        last_conv_mask = None
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
                hi += mod.out_channels
                _modules = get_module(model=self.net, name=name)
                mod.mask = mask[lo:hi].view(-1)  # update mask for each conv
                lo = hi
                last_conv_mask = mod.mask
            else:
                if isinstance(mod, nn.BatchNorm2d) and last_conv_mask is not None:
                    # prune the corresponding mean and var according to mask
                    # todo:在有shortcut的结构中，bn中的值还和shortcut有关，直接剪不一定合理
                    mod.running_mean[last_conv_mask == 0] = 0
                    mod.running_var[last_conv_mask == 0] = 1
                last_conv_mask = None

    def set_bn_momentum(self,momentum):
        '''
        set the momentum of BatchNorm2d in net
        :param momentum:
        :return:
        '''
        for name,mod in self.net.named_modules():
            if isinstance(mod,nn.BatchNorm2d):
                mod.momentum=momentum


    def find_prune_rate(self, mask):
        '''
        use binary search to determine the number of filters to prune for a given flops
        :param mask:
        :return:
        '''
        is_training=self.training
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
        self.train(mode=is_training)
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
            if math.floor(self.step_tracked * self.batch_size / getattr(conf, self.dataset_name)['train_set_size'])==1:
                # an epoch of training is finished
                self.current_epoch+=1
                self.step_tracked=0
            if self.current_epoch==1 and self.step_tracked == 1:
                self.print_mask()
        return self.net(input)


class predicted_mask_and_shortcut_net(predicted_mask_net):
    def __init__(self, net, net_name, dataset_name, flop_expected, feature_len=15, gcn_rounds=2, only_gcn=False,
                 only_inner_features=False, mask_update_freq=10, mask_update_epochs=1,batch_size=128,
                 mask_training_start_epoch=10,mask_training_stop_epoch=80):
        '''
        Use filter feature extractor to extract features from a cnn and predict mask for it. The mask will guide the
        cnn to skip filters when forwarding. Every conv has a shortcut. Both extractor and cnn are updated through back-propagation.
        :param net:
        :param net_name:
        :param dataset_name:
        :param feature_len: expected length of features to extract
        :param gcn_rounds:
        :param only_gcn:
        :param only_inner_features:
        :param mask_update_freq: how often does the extractor being updated. The extractor will be updated every mask_update_freq STEPs!
        :param mask_update_epochs: update mask for mask_update_epochs STEPs
        '''
        super(predicted_mask_and_shortcut_net, self).__init__(net, net_name, dataset_name, flop_expected, feature_len,
                                                              gcn_rounds,
                                                              only_gcn, only_inner_features, mask_update_freq,
                                                              mask_update_epochs, batch_size,
                                                              mask_training_start_epoch,
                                                              mask_training_stop_epoch)

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

        if self.dataset_name == 'imagenet' or self.dataset_name == 'tiny_imagenet':
            image = torch.zeros((2, 3, 224, 224)).to(device)
        elif self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100':
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
        measure_flops.measure_model(self.net, self.dataset_name, True)


class predicted_mask_shortcut_with_weight_net(predicted_mask_net):
    def __init__(self, net, net_name, dataset_name, flop_expected, feature_len=15, gcn_rounds=2, only_gcn=False,
                 only_inner_features=False, mask_update_freq=10, mask_update_epochs=1,batch_size=128,
                 mask_training_start_epoch=10,mask_training_stop_epoch=80):
        '''
        Use filter feature extractor to extract features from a cnn and predict mask for it. The mask will guide the
        cnn to skip filters when forwarding. Every conv-bn-relu form a block with a shortcut from its input to relu. The output of bn will multiply by a weight.
         Both extractor and cnn are updated through back-propagation.
        :param net:
        :param net_name:
        :param dataset_name:
        :param feature_len: expected length of features to extract
        :param gcn_rounds:
        :param only_gcn:
        :param only_inner_features:
        :param mask_update_freq: how often does the extractor being updated. The extractor will be updated every mask_update_freq STEPs!
        :param mask_update_epochs: update mask for mask_update_epochs STEPs
        '''
        super(predicted_mask_shortcut_with_weight_net, self).__init__(net, net_name, dataset_name, flop_expected,
                                                                      feature_len, gcn_rounds,
                                                                      only_gcn, only_inner_features, mask_update_freq,
                                                                      mask_update_epochs, batch_size,
                                                                      mask_training_start_epoch,
                                                                      mask_training_stop_epoch)



        # self.check_step=0
        # global mask_list,shortcut_mask_list
        # mask_list=[]
        # shortcut_mask_list=[]


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

        if self.dataset_name == 'imagenet' or self.dataset_name == 'tiny_imagenet':
            image = torch.zeros((2, 3, 224, 224)).to(device)
        elif self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100':
            image = torch.zeros((2, 3, 32, 32)).to(device)
        net(image)  # record input feature map sizes of each conv
        for h in handle:
            del h
        conv_mod =  None
        for name, mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                assert conv_mod is None, 'Previous conv is not handled.'
                _modules = get_module(model=net, name=name)
                _modules[name.split('.')[-1]] = block_with_mask_shortcut(mod, map_size[mod]).to(device)  # replace conv with a block
                conv_mod = _modules[name.split('.')[-1]]
            elif isinstance(mod, nn.BatchNorm2d) and 'downsample' not in name:
                assert conv_mod is not None
                _modules = get_module(model=net, name=name)
                _modules[name.split('.')[-1]] = nn.Sequential()  # replace bn with an empty sequential(equal to delete it)
                conv_mod = None

        return net





    def update_mask(self):
        super().update_mask()
        for mod in self.net.modules():
            if isinstance(mod,block_with_mask_shortcut):
                mod.shortcut_mask=torch.mean(mod.mask.abs()).view(-1)


    def print_mask(self):
        channel_num_list=[]
        for name, mod in self.net.named_modules():
            if isinstance(mod, block_with_mask_shortcut):
                print('shortcut_mask:%f'%float(mod.shortcut_mask),end='\t\t')
                channel_num = torch.sum(mod.mask != 0)
                channel_num_list+=[int(channel_num)]
                print('channel_num:', int(channel_num),end='\t')  # print the number of channels without being pruned
                print(name)
        print(channel_num_list)

    def detach_mask(self):
        for name, mod in self.net.named_modules():
            if isinstance(mod, block_with_mask_shortcut):
                mod.shortcut_mask = mod.shortcut_mask.detach()  # detach shortcut_mask from computation graph
                mod.mask = mod.mask.detach()  # detach mask from computation graph


    def forward(self, input):
        # self.check_same_mask()
        return super().forward(input)



