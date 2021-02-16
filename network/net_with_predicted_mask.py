from filter_characteristic import filter_feature_extractor
from prune.prune_module import get_module
from network.modules import conv2d_with_mask, block_with_mask_shortcut, block_with_mask_weighted_shortcut, \
    conv2d_with_mask_and_variable_shortcut, named_conv_list
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from datetime import datetime
import math
import matplotlib.pyplot as plt
from framework import data_loader, measure_flops, evaluate, train, config as conf
# import torchsnooper
from network import storage, resnet_cifar, resnet
import os, sys
import numpy as np
from collections import OrderedDict
from sklearn import manifold


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class predicted_mask_net(nn.Module):
    def __init__(self, net, net_name, dataset_name, flop_expected, feature_len=15, gcn_layer_num=2, only_gcn=False,
                 only_inner_features=False, mask_update_freq=10, mask_update_epochs=1, batch_size=128,
                 mask_training_start_epoch=10, mask_training_stop_epoch=80):
        '''
        Use filter feature extractor to extract features from a cnn and predict mask for it. The mask will guide the
        cnn to skip filters when forwarding. Both extractor and cnn are updated through back-propagation.
        :param net:
        :param net_name:
        :param dataset_name:
        :param feature_len: expected length of features to extract
        :param gcn_layer_num:
        :param only_gcn:
        :param only_inner_features:
        :param mask_update_freq: how often does the extractor being updated. The extractor will be updated every mask_update_freq EPOCHs!
        :param mask_update_epochs: how many epochs used to update mask in each update
        '''
        super(predicted_mask_net, self).__init__()
        # self.extractor = filter_feature_extractor.extractor(feature_len=feature_len, gcn_rounds=gcn_rounds,
        #                                                     only_gcn=only_gcn, only_inner_features=only_inner_features)
        self.extractor=filter_feature_extractor.extractor(net, feature_len=feature_len, layer_num=gcn_layer_num)
        self.net_name = net_name
        self.dataset_name = dataset_name
        self.feature_len = feature_len
        self.gcn_layer_num = gcn_layer_num
        # self.data_parallel=True
        self.mask_update_freq = mask_update_freq
        self.mask_update_epochs = mask_update_epochs
        self.mask_updating = False
        self.step_tracked = 1
        self.current_epoch = 1
        self.net = self.transform(net)  # .to(device)
        self.flop_expected = flop_expected
        self.batch_size = batch_size
        self.mask_training_start_epoch = mask_training_start_epoch  # mask be trained during [start_epoch,stop_epoch)
        self.mask_training_stop_epoch = mask_training_stop_epoch
        self.set_bn_momentum(momentum=0.1)
        train_set_size = getattr(conf, dataset_name)['train_set_size']
        self.num_train = train_set_size
        # self.num_train = train_set_size - int(train_set_size * 0.1)
        self.copied_time = 0

    # def train(self, mode=True):
    #     super().train(mode)
    #     return self

    # def eval(self):
    #     return self.train(False)

    def copy(self):
        '''
        self-made deepcopy
        :return: a cloned network
        '''
        if self.copied_time == 0:
            print('Warning: method copy is not exactly right. Some parameters are not copied!')
            print('Being called by: ', sys._getframe(1).f_code.co_name)
            print(sys._getframe(1).f_lineno)  # 调用该函数的行号
            self.copied_time += 1
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
                mod.track_running_stats = track
                if track is False:
                    # self.register_parameter('running_mean', None)
                    # self.register_parameter('running_var', None)
                    # self.register_parameter('num_batches_tracked', None)
                    mod.running_mean=mod.running_var=mod.num_batches_tracked=None
                else:
                    mod.register_buffer('running_mean', torch.zeros(mod.num_features))
                    mod.register_buffer('running_var', torch.ones(mod.num_features))
                    mod.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
                    mod.cuda()
                    # mod.reset_running_stats()  # reset all tracked value


    def train_mask(self):
        # update mask and track_running_stats in BN according to current step
        if self.training:
            if self.mask_training_start_epoch <= self.current_epoch < self.mask_training_stop_epoch:
                if (self.current_epoch - self.mask_training_start_epoch) % self.mask_update_freq < self.mask_update_epochs:  # mask need to be trained
                    self.update_mask()
                    if (self.current_epoch - self.mask_training_start_epoch) % self.mask_update_freq == 0 and self.step_tracked == 1:
                        self.mask_updating = True
                        self.print_mask()
                        print('{} start updating the mask.'.format(datetime.now()))
                        self.track_running_stats(
                            track=False)  # don't track stats on BN when updating the mask to avoid volatility
                else:
                    if (
                            self.current_epoch - self.mask_training_start_epoch) % self.mask_update_freq == self.mask_update_epochs and self.step_tracked == 1:
                        self.mask_net()
                        self.mask_updating = False
                        self.print_mask()
                        print('{} stop updating the mask.'.format(datetime.now()))
                        self.track_running_stats(track=True)  # track stats on BN when mask is not updated
                    self.detach_mask()
            else:  # mask will not be trained. Only net will be trained
                if self.current_epoch == self.mask_training_stop_epoch and self.step_tracked == 1:
                    self.print_mask()
                    print('{} stop updating the mask since the current epoch is {}.'.format(datetime.now(),
                                                                                            self.current_epoch))
                    self.detach_mask()
                    self.mask_updating = False
                    self.track_running_stats(track=True)  # track stats on BN when mask is not updated
                pass
        else:
            raise Exception('Masks should not be trained or updated in evaluation mode.')

    def update_mask(self):
        if self.training is False:
            raise Exception('Masks should not be updated in evaluation mode.')
        mask = self.extractor(self, self.net_name, self.dataset_name)  # predict mask using extractor
        lo = hi = 0
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
                hi += mod.out_channels
                _modules = get_module(model=self.net, name=name)
                mod.set_mask(mask[lo:hi].view(-1))  # update mask for each conv
                lo = hi

    def set_bn_momentum(self, momentum):
        '''
        set the momentum of BatchNorm2d in net
        :param momentum:
        :return:
        '''
        for name, mod in self.net.named_modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.momentum = momentum

    def find_prune_num(self, mask):
        '''
        use binary search to determine the number of filters to prune for a given flops
        :param mask:
        :return:
        '''
        is_training = self.training
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
                    # mod.mask = mask_tmp[lo:hi].view(-1)  # update mask for each con
                    mod.set_mask(mask_tmp[lo:hi].view(-1))
                    lo = hi

            flops = measure_flops.measure_model(self, dataset_name=self.dataset_name, print_flop=False)
            if flops < self.flop_expected:  # need to prune less(a lower prune rate)
                prune_rate_hi = prune_rate
            elif flops > self.flop_expected:  # need to prune more(a higher prune rate)
                prune_rate_lo = prune_rate
            else:
                break
        self.train(mode=is_training)
        return int(prune_rate * len(mask))

    def print_mask(self):
        channel_num_list = []
        layer = -1
        for name, mod in self.net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                layer += 1
                if isinstance(mod, conv2d_with_mask):
                    channel_num = torch.sum(mod.mask != 0)
                else:
                    channel_num = mod.out_channels
                channel_num_list += [int(channel_num)]
                print('channel_num:  ', int(channel_num), end='\t')  # print the number of channels without being pruned
                print('layer', layer, end='\t')
                print(name)
        print(channel_num_list)

    def measure_self_flops(self):
        return measure_flops.measure_model(self.net, dataset_name=self.dataset_name, print_flop=False)

    def detach_mask(self):
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
                mod.mask = mod.mask.detach()  # detach masks from computation graph so the extractor will not be updated

    def forward(self, input):
        if self.training is True:
            self.train_mask()
            if math.floor(self.step_tracked * self.batch_size / self.num_train) == 1:
                # an epoch of training is finished
                self.current_epoch += 1
                self.step_tracked = 0
            self.step_tracked += 1
        return self.net(input)

    def set_structure(self, structure):
        layer = -1
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask):
                layer += 1
                mod.mask[structure[layer]:] = 0

    def mask_net(self):
        '''
        set top k mask to 0
        :return:
        '''
        if self.training is False:
            raise Exception('Masks should not be updated in evaluation mode.')
        mask = self.extractor(self, self.net_name, self.dataset_name)  # predict mask using extractor
        mask_clone = mask.detach().clone()

        # if 'resnet' in self.net_name:#preserve all filters in first conv for resnet
        #     for name,mod in self.net.named_modules():
        #         if isinstance(mod,nn.Conv2d):
        #             out_channels=mod.out_channels
        #             mask_clone[:out_channels]=1
        #             break

        prune_num = self.find_prune_num(mask_clone)
        _, mask_index = torch.topk(torch.abs(mask_clone), k=prune_num, dim=0, largest=False)
        index = torch.ones(mask.shape).to(mask.device)
        index[mask_index] = 0
        mask = mask * index

        lo = hi = 0
        last_conv_mask = None
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
                hi += mod.out_channels
                # _modules = get_module(model=self.net, name=name)
                mod.set_mask(mask[lo:hi].view(-1))  # update mask for each conv
                lo = hi
                last_conv_mask = mod.mask
            else:
                if isinstance(mod, nn.BatchNorm2d) and last_conv_mask is not None:
                    # prune the corresponding mean and var according to mask
                    # todo:在有shortcut的结构中，bn中的值还和shortcut有关，直接剪不一定合理
                    mod.running_mean[last_conv_mask == 0] = 0
                    mod.running_var[last_conv_mask == 0] = 1
                last_conv_mask = None

    def t_sne(self):
        hidden_states=self.extractor.gat(self)
        mask=self.extractor.network(hidden_states)
        prune_num = self.find_prune_num(mask)
        _, mask_index = torch.topk(torch.abs(mask), k=prune_num, dim=0, largest=False)
        mask = mask.detach().cpu().numpy()
        index = np.ones(mask.shape).reshape(-1)
        index[mask_index.cpu().view(-1)] = 0
        mask = mask.reshape(-1) * index
        mask[mask!=0]=1

        tsne = manifold.TSNE(n_components=2, perplexity=40, learning_rate=600, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(hidden_states.detach().cpu().numpy())
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], 'o', color=plt.cm.Set1(mask[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()


class predicted_mask_and_variable_shortcut_net(predicted_mask_net):
    def __init__(self, net, net_name, dataset_name, flop_expected, add_shortcut_ratio, feature_len=15, gcn_layer_num=2,
                 only_gcn=False,
                 only_inner_features=False, mask_update_freq=10, mask_update_epochs=1, batch_size=128,
                 mask_training_start_epoch=10, mask_training_stop_epoch=80,
                 ):
        '''
        Use filter feature extractor to extract features from a cnn and predict mask for it. The mask will guide the
        cnn to skip filters when forwarding. Conv will have a shortcut if being pruned massively. Both extractor and cnn are updated through back-propagation.
        :param net:
        :param net_name:
        :param dataset_name:
        :param feature_len: expected length of features to extract
        :param gcn_layer_num:
        :param only_gcn:
        :param only_inner_features:
        :param mask_update_freq: how often does the extractor being updated. The extractor will be updated every mask_update_freq STEPs!
        :param mask_update_epochs: update mask for mask_update_epochs STEPs
        :param add_shortcut_ratio: a shortcut will be added if the ratio of masked conv is greater than r*100%
        '''

        self.__add_shortcut_ratio = add_shortcut_ratio
        super(predicted_mask_and_variable_shortcut_net, self).__init__(net=net,
                                                                       net_name=net_name,
                                                                       dataset_name=dataset_name,
                                                                       flop_expected=flop_expected,
                                                                       feature_len=feature_len,
                                                                       gcn_layer_num=gcn_layer_num,
                                                                       only_gcn=only_gcn,
                                                                       only_inner_features=only_inner_features,
                                                                       mask_update_freq=mask_update_freq,
                                                                       mask_update_epochs=mask_update_epochs,
                                                                       batch_size=batch_size,
                                                                       mask_training_start_epoch=mask_training_start_epoch,
                                                                       mask_training_stop_epoch=mask_training_stop_epoch)
        self.pruned = False

    def transform(self, net):
        def hook(module, input, output):
            map_size[module] = input[0].shape[2]

        net.eval()
        map_size = {}
        handle = []
        for name, mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                device = mod.weight.device
                handle += [mod.register_forward_hook(hook)]

        if self.dataset_name == 'imagenet' or self.dataset_name == 'tiny_imagenet':
            image = torch.zeros((1, 3, 224, 224)).to(device)
        elif self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100':
            image = torch.zeros((1, 3, 32, 32)).to(device)
        net(image)  # record input feature map sizes of each conv
        for h in handle:
            del h
        for name, mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                # assert conv_mod is None, 'Previous conv is not handled.'
                _modules = get_module(model=net, name=name)
                _modules[name.split('.')[-1]] = \
                    conv2d_with_mask_and_variable_shortcut(mod, map_size[mod],self.__add_shortcut_ratio).to(device)  # replace conv with a block
        return net

    def get_shortcut_ratio(self):
        return self.__add_shortcut_ratio

    def set_shortcut_ratio(self, add_shortcut_ratio):
        print('If you do not know what you are doing ,stop doing that!')
        self.__add_shortcut_ratio = add_shortcut_ratio
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                mod.add_shortcut_ratio = add_shortcut_ratio
                mod.add_shortcut_num = math.ceil(mod.out_channels * (1 - add_shortcut_ratio))
                if len(mod.downsample) > 0:
                    mod.downsample = nn.Sequential(OrderedDict([
                        ('downsampleConv', nn.Conv2d(in_channels=mod.downsample[0].in_channels,
                                                     out_channels=mod.add_shortcut_num,
                                                     stride=mod.downsample[0].stride,
                                                     kernel_size=1,
                                                     bias=False)),
                        ('downsampleBN', nn.BatchNorm2d(mod.add_shortcut_num))
                    ]))
                    mod.downsample.cuda()

    def find_prune_num(self, mask, delta=0.005, start_prune_num=100):
        '''
        determine the number of filters to prune for a given flops
        f(prune_rate) represents the flops of network w.r.t prune_rate. It's not monotonically decreasing.
        So this code try to find the target prune_rate in the left
        :param mask:
        :param delta: tolerance  of flops between net's flops and expected flops
        :param start_prune_num: initial number of filters to prune
        :return:
        '''
        # temporarily used for resnet50
        # 1:0.8:16167;2:0.8:17615 ;3:0.8:19670; 4:0.9:20884 0.75:16530 0.8：19124; 5:0.8:13633,0.85:14755; 6:0.9:14166,0.85:10560,0.8:9318/9200,0.75:/8170,0.7:/7200,0.5:/3150
        if isinstance(self.net, resnet.ResNet):
            num = 3150
            print('prune:{} filters'.format(num))
            return num

        is_training = self.training
        mask = np.abs(mask.clone().view(-1).detach().cpu().numpy())
        out_channel_list = []
        conv_list = []
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                mod.compute_flops(mod.in_channels, mod.out_channels)
                out_channel_list += [mod.out_channels]
                conv_list += [mod]
        total_flop = self.measure_self_flops(num_filter_unmasked=out_channel_list)
        if abs(total_flop - self.flop_expected) / total_flop <= delta:
            return 0
        #todo:这里需要找到
        prune_num = start_prune_num - 1
        top_k_idx = mask.argsort()[0:prune_num]  # find the top k smallest mask
        mask[top_k_idx] = 999  # set mask to 999 to simulate that this mask is set to 0
        while prune_num < len(mask):
            prune_num += 1
            mask_index = np.argmin(mask)
            mask[mask_index] = 999  # simulate that this mask is set to 0
            i = 0
            filter_num_list = []
            for mod in conv_list:
                filter_num_list += [int((mask[i:i + len(mod.mask)] != 999).sum())]  # num of filters in each layer
                i += len(mod.mask)
            in_channel_list = self.compute_net_structure(filter_num_list=filter_num_list)  # get net structure
            flop_delta = 0  # flops reduced from pruning
            for i,mod in enumerate(conv_list):
                flop_delta += mod.compute_flops(mod.in_channels, mod.out_channels) \
                              - mod.compute_flops(in_channel_list[i], filter_num_list[i])  # flops reduced by conv
                # todo this might not be completely right since num_feature_map can be larger than out_channels（because of the shortcut）
                flop_delta += 2 * mod.w_out * mod.w_out * ( mod.out_channels - filter_num_list[i])  # flops reduced by BatchNormalization

                if filter_num_list[i] <= mod.add_shortcut_num:
                    flop_delta -= mod.compute_downsample_flops(in_channel_list[i])  # flops increased by downsample
            flops = total_flop - flop_delta
            if abs(flops - self.flop_expected) / total_flop <= delta:
                self.train(mode=is_training)
                for name, mod in self.net.named_modules():
                    if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                        mod.flops = None
                return prune_num
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                mod.flops = None
        raise Exception('Can\'t find appropriate prune_rate. Consider decreasing prune_num')

    def print_mask(self):
        channel_num_list = []
        layer = -1
        for name, mod in self.net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                layer += 1
                if self.pruned is True:
                    print('in_channels:',mod.in_channels,end='\t')
                if isinstance(mod, conv2d_with_mask):
                    channel_num = torch.sum(mod.mask != 0)
                else:
                    channel_num = mod.out_channels
                channel_num_list += [int(channel_num)]
                print('channel_num:  ', int(channel_num), end='\t')  # print the number of channels without being pruned
                print('layer', layer, end='\t')
                print(name)
        print(channel_num_list)


    def reshape_data_to_net_structure_resnet(self, data):
        '''
        reshape the data to net-structure-like shape
        :param data: array-like data
        :return: data reshaped
        '''
        new_shaped_data = []
        first_conv = False
        for name, mod in self.net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name and first_conv is False:
                first_conv = True
                new_shaped_data.append([data[0]])
                i = 1
            elif isinstance(mod, resnet_cifar.BasicBlock) or isinstance(mod, resnet.Bottleneck):
                _, block_conv_list = named_conv_list(mod)
                new_shaped_data.append(data[i:i + len(block_conv_list)])
                i = i + len(block_conv_list)
        return new_shaped_data

    def compute_net_structure(self, filter_num_list):
        if 'resnet' in self.net_name:
            return self.compute_net_structure_resnet(filter_num_list)
        elif 'vgg' in self.net_name or 'mobilenet_v1' in self.net_name:
            in_channel_list = [3]
            first_conv = False
            i = -1
            for name, mod in self.net.named_modules():
                if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                    i += 1
                    if first_conv is False:
                        first_conv = True
                        last_conv = mod
                        continue
                    last_conv_filter_num = filter_num_list[i - 1]
                    last_conv_in_channels = in_channel_list[i - 1]
                    if last_conv_filter_num > last_conv.add_shortcut_num:  # last conv doesn't have a shortcut
                        in_channel_list += [last_conv_filter_num]
                    else:  # last conv has a shortcut
                        if last_conv.w_in != last_conv.w_out:  # a conv shortcut
                            in_channel_list += [last_conv.add_shortcut_num]
                        else:  # a sequential shortcut
                            in_channel_list += [max(last_conv_in_channels, last_conv_filter_num)]
                    last_conv = mod
            return in_channel_list
        else:
            raise Exception('Unsupported net.')

    def compute_net_structure_resnet(self, filter_num_list):
        '''
        compute the structure given filter num of each conv
        :param filter_num_list:
        :return: the input channel for each conv
        '''
        filter_num_list = self.reshape_data_to_net_structure_resnet(filter_num_list)
        conv_list = []  # conv module (saved in net structure)
        block_list = []  # block module(including first conv)
        first_conv_in_block = []  # whether the conv is the first in the block( saved in net structure)
        block_in_channels_list = []  # in_channels of each conv (saved in net structure)
        first_conv = False
        for name, mod in self.net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name and first_conv is False:
                first_conv = True
                block_in_channels_list.append([3])
                first_conv_in_block.append([False])
                conv_list.append([mod])
                block_list.append(mod)
            elif isinstance(mod, resnet_cifar.BasicBlock) or isinstance(mod, resnet.Bottleneck):
                _, block_conv_list = named_conv_list(mod)
                block_in_channels_list.append([-1 for i in range(len(block_conv_list))])
                first_conv_in_block.append([False for i in range(len(block_conv_list))])
                first_conv_in_block[-1][0] = True
                conv_list.append(block_conv_list)
                block_list.append(mod)

        for i in range(1, len(block_in_channels_list)):
            for j in range(len(block_in_channels_list[i])):
                if first_conv_in_block[i][j] is True:  # conv in the beginning of the block
                    last_conv = conv_list[i - 1][-1]
                    last_conv_filter_num = filter_num_list[i - 1][-1]
                    last_conv_in_channels = block_in_channels_list[i - 1][-1]
                    if last_conv.w_in != last_conv.w_out:  # last conv has a conv shortcut
                        last_conv_out_channels = max(last_conv.add_shortcut_num, last_conv_filter_num)
                    elif last_conv_filter_num > last_conv.add_shortcut_num:  # last conv doesn't have a shortcut
                        last_conv_out_channels = last_conv_filter_num
                    else:  # last conv has a sequential shortcut
                        last_conv_out_channels = max(last_conv_in_channels, last_conv_filter_num)
                    if hasattr(block_list[i - 1], 'downsample') \
                            and block_list[i - 1].downsample is not None \
                            and isinstance(block_list[i - 1], resnet.Bottleneck):  # previous block has a conv shortcut
                        block_in_channels_list[i][j] = last_conv_out_channels
                    else:  # for others
                        block_in_channels_list[i][j] = max(block_in_channels_list[i - 1][0], last_conv_out_channels)
                else:  # conv is not after a block shortcut
                    last_conv = conv_list[i][j - 1]
                    last_conv_filter_num = filter_num_list[i][j - 1]
                    last_conv_in_channels = block_in_channels_list[i][j - 1]
                    if last_conv_filter_num > last_conv.add_shortcut_num:  # last conv doesn't have a shortcut
                        block_in_channels_list[i][j] = last_conv_filter_num
                    else:  # last conv has a shortcut
                        if last_conv.w_in != last_conv.w_out:  # a conv shortcut
                            block_in_channels_list[i][j] = last_conv.add_shortcut_num
                        else:  # a sequential shortcut
                            block_in_channels_list[i][j] = max(last_conv_in_channels, last_conv_filter_num)
        in_channels_list = [3]
        for i in range(1, len(block_in_channels_list)):
            in_channels_list += block_in_channels_list[i]
        return in_channels_list

    def measure_self_flops(self, num_filter_unmasked=None):
        self.detach_mask()
        is_training = self.training
        self.eval()
        downsample_flop_overcomputed = 0  # flop in downsample may be over counted if the net has not been pruned
        bn_reduction = 0
        if self.pruned is False:
            if num_filter_unmasked is None:  # num_filter_unmasked is not given
                num_filter_unmasked = []
                for name, mod in self.net.named_modules():
                    if isinstance(mod, conv2d_with_mask):
                        num_filter_unmasked += [int(torch.sum(mod.mask != 0))]

            num_in_channels = self.compute_net_structure(num_filter_unmasked)
            i = -1
            for name, mod in self.net.named_modules():
                if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                    i += 1
                    mod.compute_flops(num_in_channels[i], num_filter_unmasked[i])
                    bn_reduction += mod.w_out * mod.w_out * 2 * (
                            mod.out_channels - num_filter_unmasked[i])  # flops reduced by BatchNormalization
                    if num_filter_unmasked[i] <= mod.add_shortcut_num:
                        downsample_flop_overcomputed += mod.compute_downsample_flops(
                            mod.in_channels - num_in_channels[i])
        else:  # net has been pruned, measure flops directly
            for name, mod in self.net.named_modules():
                if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                    mod.compute_flops(mod.in_channels, mod.out_channels)

        flops = measure_flops.measure_model(self.net, dataset_name=self.dataset_name, print_flop=False)
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                mod.flops = None
        self.train(mode=is_training)  # restore the mode
        return flops - downsample_flop_overcomputed - bn_reduction

    def print_in_out_channels(self):
        if self.pruned is not True:
            raise Exception('Net has not been pruned.')
        in_list = []
        out_list = []
        for name, mod in self.net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                print(name, '\t in:', mod.in_channels, '\t out:', mod.out_channels)
                in_list += [mod.in_channels]
                out_list += [mod.out_channels]
        print('in:', in_list)
        print('out:', out_list)

    def prune_net(self):
        if self.pruned is True:
            raise Exception('net has already been pruned')
        self.pruned = True
        # prune the conv
        if 'resnet' in self.net_name:
            last_conv_index, num_feature_maps, num_feature_maps_after_prune = self.prune_net_resnet()
        elif 'vgg' in self.net_name:
            last_conv_index, num_feature_maps, num_feature_maps_after_prune = self.prune_net_vgg()
        else:
            raise Exception('What is this net?')

        # prune the first linear layer
        with torch.no_grad():
            old_linear_layer = None
            for linear_name, module in self.net.named_modules():
                if isinstance(module, torch.nn.Linear):
                    old_linear_layer = module
                    break
            if old_linear_layer is None:
                return  # net with no fc layer
            params_per_input_channel = int(old_linear_layer.in_features / num_feature_maps)
            new_linear_layer = torch.nn.Linear(num_feature_maps_after_prune * params_per_input_channel,
                                               old_linear_layer.out_features)
            old_weights = old_linear_layer.weight
            new_weights = new_linear_layer.weight
            node_index = []
            for f in last_conv_index:
                node_index.extend([i for i in range(f * params_per_input_channel, (f + 1) * params_per_input_channel)])
            new_weights[:, :len(node_index)] = old_weights[:, [i for i in range(old_weights.shape[1]) if
                                                               i in node_index]]  # 复制剩余的filters的weight
            new_linear_layer.bias = old_linear_layer.bias
            new_linear_layer.to(list(self.parameters())[0].device)
            _modules = get_module(model=self.net, name=linear_name)
            _modules[linear_name.split('.')[-1]] = new_linear_layer
        if self.dataset_name == 'imagenet':
            data = torch.zeros((2, 3, 224, 224))
            if torch.cuda.is_available():
                data = data.cuda()
            self.net(data)
        self.track_running_stats(True)

    def prune_net_vgg(self):
        self.detach_mask()
        layer = -1
        num_layer_pruned = 0
        with torch.no_grad():
            for name, mod in self.net.named_modules():
                if isinstance(mod, conv2d_with_mask_and_variable_shortcut) and 'downsample' not in name:
                    layer += 1
                    self.prune_conv_layer(layer - num_layer_pruned)
                    index = torch.where(mod.mask != 0)[0]
                    last_conv = mod
                    if torch.sum(mod.mask != 0) == 0:
                        num_layer_pruned += 1
            last_conv_index = index
            num_feature_maps = last_conv.out_channels
            _, conv_list = named_conv_list(self.net)
            last_conv_after_prune = conv_list[-1]
            if len(last_conv_index) > last_conv.add_shortcut_num:  # no shortcut
                num_feature_maps_after_prune = len(last_conv_index)
            else:
                if last_conv.w_in != last_conv.w_out:  # conv shortcut
                    num_feature_maps_after_prune = last_conv.add_shortcut_num
                else:  # sequential shortcut
                    num_feature_maps_after_prune = max(last_conv_after_prune.in_channels, len(last_conv_index))

        return last_conv_index, num_feature_maps, num_feature_maps_after_prune

    def prune_net_resnet(self):
        self.detach_mask()
        first_conv = None
        block_list = []
        for name, mod in self.net.named_modules():
            if isinstance(mod, resnet_cifar.BasicBlock) or isinstance(mod, resnet.Bottleneck):
                block_list += [mod]
            elif isinstance(mod, nn.Conv2d) and first_conv is None:  # first conv in resnet
                first_conv = mod
                block_list += [mod]
        block_out_channels = [0 for i in
                              range(len(block_list))]  # num of out_channels for each block(including first conv)
        block_mask_index = []  # mask index of the last conv in each block

        block_out_channels[0] = max(3, int(torch.sum(first_conv.mask != 0)))
        block_mask_index += [torch.where(block_list[0].mask != 0)[0].tolist()]

        # get the mask index of last conv for each block
        for i in range(1, len(block_list)):
            name_list, conv_list = named_conv_list(block_list[i])
            last_conv = conv_list[-1]
            block_mask_index += [torch.where(last_conv.mask != 0)[0].tolist()]

        # get the number of out_channels for each block
        filter_num_list = []
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                filter_num_list += [int(torch.sum(mod.mask != 0))]
                last_conv = mod
        in_channel_list = self.compute_net_structure_resnet(filter_num_list)
        block_in_channel_list = self.reshape_data_to_net_structure_resnet(in_channel_list)
        block_out_channels = []
        for in_channel in block_in_channel_list:
            if len(in_channel) == 1:  # first conv in resnet
                continue
            else:
                block_out_channels += [in_channel[0]]  # out_channels of a block equals in_channels of the next block

        if len(block_mask_index[-1]) <= last_conv.add_shortcut_num:  # last conv has sequential shortcut
            last_block_out_channnels = max(len(block_mask_index[-1]), block_in_channel_list[-1][-1],block_in_channel_list[-1][0])
        else: #last conv is only a conv
            last_block_out_channnels = max(len(block_mask_index[-1]),block_in_channel_list[-1][0])
        block_out_channels += [last_block_out_channnels]

        num_conv_processing = 0
        if torch.sum(first_conv.mask != 0) == 0:
            num_layer_pruned = 1
        else:
            num_layer_pruned = 0
        self.prune_conv_layer(0)  # prune first conv
        with torch.no_grad():
            for i in range(1, len(block_list)):
                for name, mod in list(block_list[i].named_modules()):
                    if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                        num_conv_processing += 1
                    if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                        if 'conv_a' in name or 'conv1' in name:  # prune the input_channel of first conv in each blocks
                            device = mod.weight.device
                            new_conv = conv2d_with_mask_and_variable_shortcut(
                                torch.nn.Conv2d(in_channels=block_out_channels[i - 1],
                                                out_channels=mod.out_channels,
                                                kernel_size=mod.kernel_size,
                                                stride=mod.stride,
                                                padding=mod.padding,
                                                dilation=mod.dilation,
                                                groups=mod.groups,
                                                bias=(mod.bias is not None)),
                                w_in=mod.w_in,
                                add_shortcut_ratio=mod.add_shortcut_ratio,
                                specified_add_shortcut_num=mod.add_shortcut_num
                            )
                            new_conv.weight[:, :len(block_mask_index[i - 1]), :, :] \
                                = mod.weight[:, block_mask_index[i - 1], :, :]
                            if len(new_conv.downsample) > 0:
                                # copy corresponding weight of conv in downsample
                                new_conv.downsample[0].weight[:, :len(block_mask_index[i - 1]), :, :] \
                                    = mod.downsample[0].weight[:, block_mask_index[i - 1], :, :]
                            if mod.bias is not None:
                                new_conv.bias = mod.bias
                            new_conv.mask = mod.mask
                            new_conv.cuda()
                            # 替换
                            _modules = get_module(model=block_list[i], name=name)
                            _modules[name.split('.')[-1]] = new_conv
                        self.prune_conv_layer(num_conv_processing - num_layer_pruned)  # prune the conv
                        if torch.sum(mod.mask != 0) == 0:
                            num_layer_pruned += 1
                    if isinstance(mod, nn.BatchNorm2d) and 'downsample' not in name:
                        last_bn = mod
            last_conv_index = block_mask_index[-1]
            num_feature_maps_after_prune = block_out_channels[-1]
            num_feature_maps = last_bn.num_features
        return last_conv_index, num_feature_maps, num_feature_maps_after_prune

    def prune_conv_layer(self, layer_index):
        """
        pruning conv_a results in pruning responding in_channels in conv_b
        pruning conv_b will not affect the next conv or linear
        :param layer_index: 要删的卷基层的索引,从0开始
        :return:
        """
        conv_to_prune = None  # 获取要删filter的那层conv
        batch_norm = None  # 如果有的话：获取要删的conv后的batch normalization层
        next_conv = None  # 如果有的话：获取要删的那层后一层的conv，用于删除对应通道
        i = -1
        for name, mod in self.net.named_modules():
            if isinstance(mod, nn.Conv2d):
                if 'downsample' not in name:  # a real conv layer
                    i += 1
                if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                    if i == layer_index:
                        conv_name, conv_to_prune = name, mod
                        continue
                    if conv_to_prune is not None:
                        if 'downsample' in name:
                            raise Exception('Pruning bypass conv in block is not implemented yet.')
                        next_conv_name, next_conv = name, mod
                        break
            if isinstance(mod, nn.BatchNorm2d) and conv_to_prune is not None and 'downsample' not in name:
                if next_conv is not None:
                    raise Exception('Where is this bn from?')
                batch_norm_name, batch_norm = name, mod
                if 'bn_b' in name or 'bn_1' in name or ('layer' not in name and 'bn1' in name) or 'bn3' in name:
                    # do not prune the in_channels of first conv in each block(because it will be pruned in prune_net())
                    #including the second and third layer in resnet 56 and resnet 56 and the first layer in miblienet_v1
                    next_conv = 0
                    break
        try:
            filter_index = torch.where(conv_to_prune.mask == 0)[0].detach().cpu().numpy()
        except AttributeError:
            self.print_mask()
            print(layer_index)
            raise AttributeError
        if len(filter_index) == 0:  # no filter need to be pruned
            return
        index_to_copy = [i for i in range(conv_to_prune.weight.shape[0]) if i not in filter_index]
        with torch.no_grad():
            if torch.sum(conv_to_prune.mask != 0) > conv_to_prune.add_shortcut_num:  # prune the conv since it doesn't have a shortcut
                new_conv = nn.Conv2d(in_channels=conv_to_prune.in_channels,
                                     out_channels=conv_to_prune.out_channels - len(
                                         filter_index),
                                     kernel_size=conv_to_prune.kernel_size,
                                     stride=conv_to_prune.stride,
                                     padding=conv_to_prune.padding,
                                     dilation=conv_to_prune.dilation,
                                     groups=conv_to_prune.groups,
                                     bias=(conv_to_prune.bias is not None))
            else:
                if len(index_to_copy) == 0:  # all filters will be pruned
                    new_conv = conv_to_prune.downsample
                else:  # a conv with shortcut
                    new_conv = conv2d_with_mask_and_variable_shortcut(nn.Conv2d(in_channels=conv_to_prune.in_channels,
                                                                                out_channels=conv_to_prune.out_channels - len(
                                                                                    filter_index),
                                                                                kernel_size=conv_to_prune.kernel_size,
                                                                                stride=conv_to_prune.stride,
                                                                                padding=conv_to_prune.padding,
                                                                                dilation=conv_to_prune.dilation,
                                                                                groups=conv_to_prune.groups,
                                                                                bias=(conv_to_prune.bias is not None)),
                                                                      w_in=conv_to_prune.w_in,
                                                                      add_shortcut_ratio=conv_to_prune.add_shortcut_ratio,
                                                                      specified_add_shortcut_num=conv_to_prune.add_shortcut_num)
            if len(index_to_copy) != 0:
                # copy weights
                new_conv.weight[:] = conv_to_prune.weight[index_to_copy] * conv_to_prune.mask[index_to_copy].view(-1, 1,1,1)  # 复制剩余的filters的weight

                if conv_to_prune.bias is not None:
                    new_conv.bias[:] = conv_to_prune.bias[index_to_copy] * conv_to_prune.mask[
                        index_to_copy]  # 复制剩余的filters的bias
            new_conv.cuda()
            # replace
            _modules = get_module(model=self.net, name=conv_name)
            _modules[conv_name.split('.')[-1]] = new_conv

            if batch_norm is not None:
                if isinstance(new_conv, nn.Conv2d):
                    if isinstance(new_conv, conv2d_with_mask_and_variable_shortcut):  # new_conv has a shortcut
                        out_channels = new_conv.get_out_channels_after_prune()
                    else:  # new_conv is a norm conv
                        out_channels = new_conv.out_channels
                    new_batch_norm = torch.nn.BatchNorm2d(out_channels)
                    new_batch_norm.num_batches_tracked = batch_norm.num_batches_tracked
                    new_batch_norm.weight[:len(index_to_copy)] = batch_norm.weight[index_to_copy]
                    new_batch_norm.bias[:len(index_to_copy)] = batch_norm.bias[index_to_copy]
                    new_batch_norm.running_mean[:len(index_to_copy)] = batch_norm.running_mean[index_to_copy]
                    new_batch_norm.running_var[:len(index_to_copy)] = batch_norm.running_var[index_to_copy]
                    new_batch_norm.cuda()
                else:  # new_conv is shortcut
                    new_batch_norm = nn.Sequential()
                # 替换
                _modules = get_module(model=self.net, name=batch_norm_name)
                _modules[batch_norm_name.split('.')[-1]] = new_batch_norm

            if next_conv is not None:  # prune corresponding channel in the next conv(first conv in each block will not enter here)
                if next_conv == 0:
                    return
                if isinstance(new_conv, nn.Conv2d) and not isinstance(new_conv,
                                                                      conv2d_with_mask_and_variable_shortcut):  # new_conv is a normal conv
                    in_channels = new_conv.out_channels
                elif isinstance(new_conv,
                                conv2d_with_mask_and_variable_shortcut) and conv_to_prune.w_out != conv_to_prune.w_in:  # new_conv is a conv with shortcut
                    in_channels = new_conv.get_out_channels_after_prune()
                elif conv_to_prune.w_out == conv_to_prune.w_in:  # new_conv is a sequential shortcut
                    in_channels = max(conv_to_prune.in_channels, len(index_to_copy))
                else:  # new_conv is a conv shortcut
                    in_channels = conv_to_prune.add_shortcut_num

                new_next_conv = conv2d_with_mask_and_variable_shortcut(
                    torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=next_conv.out_channels,
                                    kernel_size=next_conv.kernel_size,
                                    stride=next_conv.stride,
                                    padding=next_conv.padding,
                                    dilation=next_conv.dilation,
                                    groups=next_conv.groups,
                                    bias=(next_conv.bias is not None)),
                    next_conv.w_in,
                    next_conv.add_shortcut_ratio
                )
                new_next_conv.weight[:, :len(index_to_copy), :, :] = next_conv.weight[:, index_to_copy, :, :]
                if next_conv.bias is not None:
                    new_next_conv.bias = next_conv.bias
                # new_next_conv.mask=next_conv.mask
                new_next_conv.set_mask(next_conv.mask)
                new_next_conv.cuda()
                # 替换
                _modules = get_module(model=self.net, name=next_conv_name)
                _modules[next_conv_name.split('.')[-1]] = new_next_conv

    def forward(self, input):
        if self.training and self.current_epoch == self.mask_training_stop_epoch and self.step_tracked == 1:  # training of mask is finished
            print('Prune the network.')
            # todo: this only works when training_stop_epoch can't be divided by train_freq, or the net will be masked again in forward() after pruned
            self.mask_net()
            self.print_mask()
            self.prune_net()  # prune filters
        if self.pruned is not True:
            out = super().forward(input)
        else:
            out = self.net(input)
        return out

    def load_state_dict(self, state_dict, strict=True):
        try:
            msg = super().load_state_dict(state_dict, strict)
        except RuntimeError as e:
            if 'size mismatch' in e.args[0]:
                print('{} Loading parameters from a pruned network'.format(datetime.now()))
                self.prune_net()
                msg = super().load_state_dict(state_dict, strict)
                self.detach_mask()  # the mask will not be trained under a pruned cnn
                self.mask_updating = False
                self.current_epoch = self.mask_training_stop_epoch + 1
            else:
                raise e
        return msg


# class predicted_mask_and_shortcut_net(predicted_mask_net):
#     def __init__(self, net, net_name, dataset_name, flop_expected, feature_len=15, gcn_rounds=2, only_gcn=False,
#                  only_inner_features=False, mask_update_freq=10, mask_update_epochs=1,batch_size=128,
#                  mask_training_start_epoch=10,mask_training_stop_epoch=80,
#                  add_shortcut_ratio=0):
#         '''
#         Use filter feature extractor to extract features from a cnn and predict mask for it. The mask will guide the
#         cnn to skip filters when forwarding. Every conv has a shortcut. Both extractor and cnn are updated through back-propagation.
#         :param net:
#         :param net_name:
#         :param dataset_name:
#         :param feature_len: expected length of features to extract
#         :param gcn_rounds:
#         :param only_gcn:
#         :param only_inner_features:
#         :param mask_update_freq: how often does the extractor being updated. The extractor will be updated every mask_update_freq STEPs!
#         :param mask_update_epochs: update mask for mask_update_epochs STEPs
#         :param add_shortcut_ratio: a shortcut will be added if the ratio of masked conv is greater than r*100%
#         '''
#
#         # self.add_shortcut_ratio = add_shortcut_ratio
#         super(predicted_mask_and_shortcut_net, self).__init__(net, net_name, dataset_name, flop_expected, feature_len,
#                                                               gcn_rounds,
#                                                               only_gcn, only_inner_features, mask_update_freq,
#                                                               mask_update_epochs, batch_size,
#                                                               mask_training_start_epoch,
#                                                               mask_training_stop_epoch)
#
#
#     def transform(self, net):
#         def hook(module, input, output):
#             map_size[module] = input[0].shape[1]
#
#         net.eval()
#         map_size = {}
#         handle = []
#         for name, mod in net.named_modules():
#             if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
#                 device = mod.weight.device
#                 handle += [mod.register_forward_hook(hook)]
#
#         if self.dataset_name == 'imagenet' or self.dataset_name == 'tiny_imagenet':
#             image = torch.zeros((2, 3, 224, 224)).to(device)
#         elif self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100':
#             image = torch.zeros((1, 3, 32, 32)).to(device)
#         net(image)  # record input feature map sizes of each conv
#         for h in handle:
#             del h
#         conv_mod = None
#         for name, mod in net.named_modules():
#             if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
#                 assert conv_mod is None, 'Previous conv is not handled.'
#                 _modules = get_module(model=net, name=name)
#                 _modules[name.split('.')[-1]] = block_with_mask_shortcut(mod, map_size[mod]).to(
#                     device)  # replace conv with a block
#                 conv_mod = _modules[name.split('.')[-1]]
#             elif isinstance(mod, nn.BatchNorm2d) and 'downsample' not in name:
#                 assert conv_mod is not None
#                 _modules = get_module(model=net, name=name)
#                 _modules[
#                     name.split('.')[-1]] = nn.Sequential()  # replace bn with an empty sequential(equal to delete it)
#                 conv_mod = None
#
#         return net
#
#     def print_mask(self):
#         super().print_mask()
#         # measure_flops.measure_model(self.net, self.dataset_name, True)
#
#     # def prune_conv(self):
#     #     if 'resnet' in self.net_name and 'cifar10'==self.dataset_name:
#     #         layer_index=-1
#     #         for name,mod in list(self.net.named_modules()):
#     #             if isinstance(mod,block_with_mask_shortcut):
#     #                 layer_index+=1
#     #                 if 'conv_a' in name:
#     #                     if torch.sum(mod.mask==0)/float(len(mod.mask))< mod.add_shortcut_ratio: #prune the conv since it doesn't have a shortcut
#     #                         filter_index=torch.where(mod.mask==0)[0].detach().cpu().numpy()
#     #                         print(layer_index)
#     #                         print(len(filter_index))
#     #                         prune_conv_layer_resnet(self.net,layer_index,filter_index)
#     #
#     #
#     #     print()
#
#
#     def prune_zero_block(self):
#         if self.mask_updating is True or self.current_epoch <= self.mask_training_stop_epoch:
#             raise Exception('You should not prune zero block before mask updated.')
#         print('Prune unnecessary blocks')
#         for name, mod in self.net.named_modules():
#             if isinstance(mod, block_with_mask_shortcut):
#                 if torch.sum(mod.mask.abs()) == 0:
#                     _modules = get_module(model=self.net, name=name)
#                     _modules[name.split('.')[-1]] = mod.downsample  # replace the whole block with downsample
#                     print('prune:',name)
#
#     def forward(self, input):
#         if self.current_epoch == self.mask_training_stop_epoch + 1 and self.step_tracked == 1:  # training of mask is finished
#             self.prune_zero_block()  # prune block with zero mask
#
#         out = super().forward(input)
#
#         return out


# class predicted_mask_shortcut_with_weight_net(predicted_mask_net):
#     def __init__(self, net, net_name, dataset_name, flop_expected, feature_len=15, gcn_rounds=2, only_gcn=False,
#                  only_inner_features=False, mask_update_freq=10, mask_update_epochs=1, batch_size=128,
#                  mask_training_start_epoch=10, mask_training_stop_epoch=80):
#         '''
#         Use filter feature extractor to extract features from a cnn and predict mask for it. The mask will guide the
#         cnn to skip filters when forwarding. Every conv-bn-relu form a block with a shortcut from its input to relu. The output of bn will multiply by a weight.
#          Both extractor and cnn are updated through back-propagation.
#         :param net:
#         :param net_name:
#         :param dataset_name:
#         :param feature_len: expected length of features to extract
#         :param gcn_rounds:
#         :param only_gcn:
#         :param only_inner_features:
#         :param mask_update_freq: how often does the extractor being updated. The extractor will be updated every mask_update_freq STEPs!
#         :param mask_update_epochs: update mask for mask_update_epochs STEPs
#         '''
#         raise Exception('补全！！！')
#         super(predicted_mask_shortcut_with_weight_net, self).__init__(net, net_name, dataset_name, flop_expected,
#                                                                       feature_len, gcn_rounds,
#                                                                       only_gcn, only_inner_features, mask_update_freq,
#                                                                       mask_update_epochs, batch_size,
#                                                                       mask_training_start_epoch,
#                                                                       mask_training_stop_epoch)
#
#         # self.check_step=0
#         # global mask_list,shortcut_mask_list
#         # mask_list=[]
#         # shortcut_mask_list=[]
#
#     def transform(self, net):
#         def hook(module, input, output):
#             map_size[module] = input[0].shape[1]
#
#         net.eval()
#         map_size = {}
#         handle = []
#         for name, mod in net.named_modules():
#             if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
#                 device = mod.weight.device
#                 handle += [mod.register_forward_hook(hook)]
#
#         if self.dataset_name == 'imagenet' or self.dataset_name == 'tiny_imagenet':
#             image = torch.zeros((2, 3, 224, 224)).to(device)
#         elif self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100':
#             image = torch.zeros((1, 3, 32, 32)).to(device)
#         net(image)  # record input feature map sizes of each conv
#         for h in handle:
#             del h
#         conv_mod = None
#         for name, mod in net.named_modules():
#             if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
#                 assert conv_mod is None, 'Previous conv is not handled.'
#                 _modules = get_module(model=net, name=name)
#                 _modules[name.split('.')[-1]] = block_with_mask_weighted_shortcut(mod, map_size[mod]).to(
#                     device)  # replace conv with a block
#                 conv_mod = _modules[name.split('.')[-1]]
#             elif isinstance(mod, nn.BatchNorm2d) and 'downsample' not in name:
#                 assert conv_mod is not None
#                 _modules = get_module(model=net, name=name)
#                 _modules[
#                     name.split('.')[-1]] = nn.Sequential()  # replace bn with an empty sequential(equal to delete it)
#                 conv_mod = None
#
#         return net
#
#     def update_mask(self):
#         super().update_mask()
#         for mod in self.net.modules():
#             if isinstance(mod, block_with_mask_weighted_shortcut):
#                 mod.shortcut_mask = torch.mean(mod.mask.abs()).view(-1)
#
#     def print_mask(self):
#         channel_num_list = []
#         for name, mod in self.net.named_modules():
#             if isinstance(mod, block_with_mask_weighted_shortcut):
#                 print('shortcut_mask:%f' % float(mod.shortcut_mask), end='\t\t')
#                 channel_num = torch.sum(mod.mask != 0)
#                 channel_num_list += [int(channel_num)]
#                 print('channel_num:', int(channel_num), end='\t')  # print the number of channels without being pruned
#                 print(name)
#         print(channel_num_list)
#
#     def detach_mask(self):
#         for name, mod in self.net.named_modules():
#             if isinstance(mod, block_with_mask_weighted_shortcut):
#                 mod.shortcut_mask = mod.shortcut_mask.detach()  # detach shortcut_mask from computation graph
#                 mod.mask = mod.mask.detach()  # detach mask from computation graph
