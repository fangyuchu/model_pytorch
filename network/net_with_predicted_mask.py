from filter_characteristic import filter_feature_extractor
from prune.prune_module import get_module
from network.modules import conv2d_with_mask, block_with_mask_shortcut, block_with_mask_weighted_shortcut,conv2d_with_mask_and_variable_shortcut
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from datetime import datetime
import math
import matplotlib.pyplot as plt
from framework import data_loader, measure_flops, evaluate, train, config as conf
import torchsnooper
from network import storage
import os,sys
import numpy as np

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
        self.step_tracked = 1
        self.current_epoch=1
        self.net = self.transform(net)  # .to(device)
        self.flop_expected = flop_expected
        self.loader =None
        self.batch_size=batch_size
        self.mask_training_start_epoch=mask_training_start_epoch    #mask be trained during [start_epoch,stop_epoch)
        self.mask_training_stop_epoch=mask_training_stop_epoch
        self.set_bn_momentum(momentum=0.1)
        train_set_size = getattr(conf, dataset_name)['train_set_size']
        self.num_train = train_set_size - int(train_set_size * 0.1)
        self.copied_time=0



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
        if self.copied_time==0:
            print('Warning: method copy is not exactly right. Some parameters are not copied!')
            print('Being called by: ',sys._getframe(1).f_code.co_name)
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
                # if track is False:
                mod.reset_running_stats()  # reset all tracked value
                mod.track_running_stats = track

    def train_mask(self):
        # update mask and track_running_stats in BN according to current step
        if self.training:
            if self.mask_training_start_epoch <= self.current_epoch < self.mask_training_stop_epoch:
                if (self.current_epoch-self.mask_training_start_epoch) % self.mask_update_freq < self.mask_update_epochs:  # mask need to be trained
                        self.update_mask()
                        if (self.current_epoch-self.mask_training_start_epoch) % self.mask_update_freq == 0 and self.step_tracked==1:
                            self.mask_updating = True
                            self.print_mask()
                            print('{} start updating the mask.'.format(datetime.now()))
                            self.track_running_stats(track=False)  # don't track stats on BN when updating the mask to avoid volatility
                else:
                    if (self.current_epoch-self.mask_training_start_epoch) % self.mask_update_freq == self.mask_update_epochs and self.step_tracked==1:
                        self.mask_updating = False
                        self.print_mask()
                        print('{} stop updating the mask.'.format(datetime.now()))
                        self.track_running_stats(track=True)  # track stats on BN when mask is not updated
                    self.detach_mask()
            else:  # mask will not be trained. Only net will be trained
                if self.current_epoch==self.mask_training_stop_epoch and self.step_tracked==1:
                    self.print_mask()
                    print('{} stop updating the mask since the current epoch is {}.'.format(datetime.now(),self.current_epoch))
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
        mask_clone=mask.detach().clone()

        # if 'resnet' in self.net_name:#preserve all filters in first conv for resnet
        #     for name,mod in self.net.named_modules():
        #         if isinstance(mod,nn.Conv2d):
        #             out_channels=mod.out_channels
        #             mask_clone[:out_channels]=1
        #             break

        prune_num = self.find_prune_num(mask_clone)
        _, mask_index = torch.topk(torch.abs(mask_clone), k=prune_num , dim=0, largest=False)
        index=torch.ones(mask.shape).to(mask.device)
        index[mask_index]=0
        mask=mask*index

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


    def find_prune_num(self, mask):
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
        return int(prune_rate*len(mask))

    def print_mask(self):
        channel_num_list = []
        for name, mod in self.net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                if isinstance(mod, conv2d_with_mask):
                    channel_num = torch.sum(mod.mask != 0)
                else:
                    channel_num = mod.out_channels
                channel_num_list += [int(channel_num)]
                print('channel_num:', int(channel_num), end='\t')  # print the number of channels without being pruned
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
            if math.floor(self.step_tracked * self.batch_size / self.num_train)==1:
                # an epoch of training is finished
                self.current_epoch+=1
                self.step_tracked=0
            self.step_tracked += 1
        return self.net(input)

    def set_structure(self, structure):
        layer=-1
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask):
                layer+=1
                mod.mask[structure[layer]:]=0


class predicted_mask_and_variable_shortcut_net(predicted_mask_net):
    def __init__(self, net, net_name, dataset_name, flop_expected,add_shortcut_ratio, feature_len=15, gcn_rounds=2, only_gcn=False,
                 only_inner_features=False, mask_update_freq=10, mask_update_epochs=1,batch_size=128,
                 mask_training_start_epoch=10,mask_training_stop_epoch=80,
                 ):
        '''
        Use filter feature extractor to extract features from a cnn and predict mask for it. The mask will guide the
        cnn to skip filters when forwarding. Conv will have a shortcut if being pruned massively. Both extractor and cnn are updated through back-propagation.
        :param net:
        :param net_name:
        :param dataset_name:
        :param feature_len: expected length of features to extract
        :param gcn_rounds:
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
                                                                       gcn_rounds=gcn_rounds,
                                                                       only_gcn=only_gcn,
                                                                       only_inner_features=only_inner_features,
                                                                       mask_update_freq=mask_update_freq,
                                                                       mask_update_epochs=mask_update_epochs,
                                                                       batch_size=batch_size,
                                                                       mask_training_start_epoch=mask_training_start_epoch,
                                                                       mask_training_stop_epoch=mask_training_stop_epoch)
        self.change_shortcut_ratio(add_shortcut_ratio)


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
                _modules[name.split('.')[-1]] = conv2d_with_mask_and_variable_shortcut(mod, map_size[mod], self.__add_shortcut_ratio).to(device)  # replace conv with a block
        return net

    def change_shortcut_ratio(self,ratio):
        self.__add_shortcut_ratio=ratio
        for name,mod in self.net.named_modules():
            if isinstance(mod,conv2d_with_mask_and_variable_shortcut):
                mod.add_shortcut_ratio=ratio
    def get_shortcut_ratio(self):
        return self.__add_shortcut_ratio

    def find_prune_num(self, mask, delta=5e5):
        '''

        determine the number of filters to prune for a given flops
        f(prune_rate) represents the flops of network w.r.t prune_rate.
        The flops first decreases along prune_rate and increases afterwards(due to added shortcut ) and then decrease to zero.
        So this code try to find the target prune_rate in the left
        :param mask:
        :param delta: tolerance of flops between net's flops and expected flops
        :return:
        '''
        # total_flops=
        is_training = self.training
        conv_layer_index = []
        in_c = []
        num_conv = []
        conv_list = []
        conv_flop = []
        conv_inf = []
        bn_flop_reduction=[]
        layer = -1
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                layer += 1
                conv_layer_index += [layer for i in range(mod.out_channels)]
                conv_list += [mod]
                in_c += [mod.in_channels]
                num_conv += [mod.out_channels]
                conv_inf += [{'shortcut': False, 'static_oc': False}]  # static_oc stands for static out_channels
                if 'conv_a' not in name:
                    conv_inf[-1]['static_oc'] = True
                conv_flop += [mod.compute_flops(mod.in_channels, mod.out_channels)]
                bn_flop_reduction+=[0]
        total_flops = measure_flops.measure_model(self.net, self.dataset_name, False)
        total_maskconv_flop = sum(conv_flop)
        total_nonmaskconv_flop = total_flops - total_maskconv_flop  # total flops of non_conv layer
        mask = np.abs(mask.clone().detach().cpu().numpy())

        prune_num = 0
        while prune_num < len(mask):
            prune_num += 1
            mask_index = np.argmin(mask)
            mask[mask_index] = 999  # simulate that this mask is set to 0
            layer = conv_layer_index[mask_index]
            conv = conv_list[layer]
            next_conv = conv_list[layer + 1] if layer < len(conv_list) - 1 else None
            num_conv[layer] -= 1

            # judge if the input number of feature maps also decreases one in the next layer
            conv_ratio_filter_masked = 1 - num_conv[layer] / conv.out_channels
            if conv_ratio_filter_masked < conv.add_shortcut_ratio:  # no shortcut needed
                if conv_inf[layer]['static_oc'] is False:  # current conv can be pruned
                    in_c[layer + 1] -= 1  # in_channel of next layer decrease one
                    bn_flop_reduction[layer]+=conv.w_out*conv.w_out*2# bn after conv can skip one feature map of calculation
                else:  # current conv is only masked
                    pass  # in_channel of next layer remains same
            else:  # shortcut is needed
                if conv_inf[layer]['shortcut'] is False:  # a new shortcut will be added
                    conv_inf[layer]['shortcut'] = True
                    conv_inf[layer]['static_oc'] = True
                    total_nonmaskconv_flop += conv.compute_downsample_flops()  # calculate flops introduced by downsample
                    bn_flop_reduction[layer]=0
                    if next_conv is not None:
                        in_c[layer + 1] = next_conv.in_channels  # in_channels in next layer returns to the original
                else:  # shortcut already existed
                    pass
            # update flops of the conv and next_conv
            conv_flop[layer]=conv.compute_flops(in_c[layer], num_conv[layer])
            if next_conv is not None:
                conv_flop[layer+1]=next_conv.compute_flops(in_c[layer + 1], num_conv[layer + 1])

            # recalculate total mask conv flop and total net flops
            total_flops = total_nonmaskconv_flop + sum(conv_flop)-sum(bn_flop_reduction)

            if abs(total_flops - self.flop_expected) <= delta:
                self.train(mode=is_training)
                for name, mod in self.net.named_modules():
                    if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                        mod.flops = None

                return prune_num

        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                mod.flops = None
        raise Exception('Can\'t find appropriate prune_rate')

    def measure_self_flops(self):
        in_c=[]
        out_c=[]
        for name, mod in self.net.named_modules():
            if isinstance(mod,conv2d_with_mask_and_variable_shortcut):
                in_c+=[mod.in_channels]
                out_c+=[mod.out_channels-int(torch.sum(mod.mask == 0))]
        bn_reduction_sum=0
        if 'resnet' in self.net_name and 'cifar10' == self.dataset_name:
            layer = -1
            for name, mod in self.net.named_modules():
                if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                    layer += 1
                    num_filter_masked = float(torch.sum(mod.mask == 0))
                    if num_filter_masked / float(len(mod.mask)) < mod.add_shortcut_ratio:
                        if 'conv_a' in name:  # this conv will be pruned
                            in_c[layer+1] -= int(num_filter_masked) #decrease the in_channel of next layer
                            bn_reduction_sum+=mod.w_out*mod.w_out*2*num_filter_masked
                    mod.compute_flops(in_c[layer], out_c[layer])

        flops = measure_flops.measure_model(self.net, self.dataset_name,print_flop=False)  # measure the flops on pruned net
        flops-=bn_reduction_sum
        for name, mod in self.net.named_modules():
            mod.flops = None
        return flops

    # @torchsnooper.snoop()
    def prune_net(self):
        if 'resnet' in self.net_name and 'cifar10'==self.dataset_name:
            layer_index=-1
            for name,mod in list(self.net.named_modules()):
                if isinstance(mod,conv2d_with_mask_and_variable_shortcut):
                    layer_index+=1
                    if torch.sum(mod.mask == 0) / float(len(mod.mask)) < mod.add_shortcut_ratio:  # prune the conv since it doesn't have a shortcut
                        mod.downsample = nn.Module()  # since mod's shortcut will not be used, set dwonsample to base Module to indicate it as useless
                        if 'conv_a' in name:
                            filter_index=torch.where(mod.mask==0)[0].detach().cpu().numpy()
                            self.prune_conv_layer_resnet(layer_index,filter_index)
            for name, mod in list(self.net.named_modules()):
                if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                    if torch.sum(mod.mask == 0) / float(len(mod.mask)) < mod.add_shortcut_ratio:   # this conv doesn't need shortcut
                        mod.downsample=None

        self.to(self.extractor.network[0].weight.device)
        self.train()

    # @torchsnooper.snoop()
    def prune_conv_layer_resnet(self,layer_index, filter_index, need_weight=True):
        """

        :param layer_index: 要删的卷基层的索引,从0开始
        :param filter_index: 要删layer_index层中的哪个filter
        :return:
        """
        if len(filter_index) == 0:  # no filter need to be pruned
            return
        conv_to_prune = None  # 获取要删filter的那层conv
        batch_norm = None  # 如果有的话：获取要删的conv后的batch normalization层
        next_conv = None  # 如果有的话：获取要删的那层后一层的conv，用于删除对应通道
        i = -1
        for name, mod in self.net.named_modules():
            if isinstance(mod, nn.Conv2d):
                if 'downsample' not in name:  # a real conv layer
                    i += 1
                if i == layer_index:
                    conv_name, conv_to_prune = name, mod
                    continue
                if conv_to_prune is not None:
                    if 'downsample' in name:
                        raise Exception('Pruning bypass conv in block is not implemented yet.')
                    next_conv_name, next_conv = name, mod
                    break
            if isinstance(mod, nn.BatchNorm2d) and conv_to_prune is not None:
                if next_conv is not None:
                    raise Exception('Where is this bn from?')
                batch_norm_name, batch_norm = name, mod
        device=conv_to_prune.weight.device
        index_to_copy = [i for i in range(conv_to_prune.weight.shape[0]) if i not in filter_index]
        with torch.no_grad():
            new_conv = nn.Conv2d(in_channels=conv_to_prune.in_channels,
                                 out_channels=conv_to_prune.out_channels - len(
                                     filter_index),
                                 kernel_size=conv_to_prune.kernel_size,
                                 stride=conv_to_prune.stride,
                                 padding=conv_to_prune.padding,
                                 dilation=conv_to_prune.dilation,
                                 groups=conv_to_prune.groups,
                                 bias=(conv_to_prune.bias is not None))

            # replace the conv_to_prune
            if need_weight:
                new_conv.weight[:] = conv_to_prune.weight[index_to_copy]  # 复制剩余的filters的weight
                if conv_to_prune.bias is not None:
                    new_conv.bias[:] = conv_to_prune.bias[index_to_copy]  # 复制剩余的filters的bias
            new_conv.to(device)
            # 替换
            _modules = get_module(model=self.net, name=conv_name)
            _modules[conv_name.split('.')[-1]] = new_conv

            if batch_norm is not None:
                if need_weight:
                    new_batch_norm = torch.nn.BatchNorm2d(new_conv.out_channels)
                    new_batch_norm.num_batches_tracked = batch_norm.num_batches_tracked
                    new_batch_norm.weight[:] = batch_norm.weight[index_to_copy]
                    new_batch_norm.bias[:] = batch_norm.bias[index_to_copy]
                    new_batch_norm.running_mean[:] = batch_norm.running_mean[index_to_copy]
                    new_batch_norm.running_var[:] = batch_norm.running_var[index_to_copy]
                new_batch_norm.to(device)
                # 替换
                _modules = get_module(model=self.net, name=batch_norm_name)
                _modules[batch_norm_name.split('.')[-1]] = new_batch_norm

            if next_conv is not None:  # next_conv中需要把对应的通道也删了
                new_next_conv = conv2d_with_mask_and_variable_shortcut(
                    torch.nn.Conv2d(in_channels=next_conv.in_channels - len(filter_index),
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
                if need_weight:
                    new_next_conv.weight[:] = next_conv.weight[:, index_to_copy, :, :]
                    if next_conv.bias is not None:
                        new_next_conv.bias = next_conv.bias
                    new_next_conv.mask=next_conv.mask
                new_next_conv.to(device)
                # 替换
                _modules = get_module(model=self.net, name=next_conv_name)
                _modules[next_conv_name.split('.')[-1]] = new_next_conv

            else:
                # Prunning the last conv layer. This affects the first linear layer of the classifier.
                old_linear_layer = None
                for _, module in self.net.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        old_linear_layer = module
                        break

                if old_linear_layer is None:
                    raise BaseException("No linear layer found in classifier")
                params_per_input_channel = int(old_linear_layer.in_features / conv_to_prune.out_channels)

                new_linear_layer = \
                    torch.nn.Linear(old_linear_layer.in_features - len(filter_index) * params_per_input_channel,
                                    old_linear_layer.out_features)

                old_weights = old_linear_layer.weight.data.cpu().numpy()
                new_weights = new_linear_layer.weight.data.cpu().numpy()

                node_index = []
                for f in filter_index:
                    node_index.extend([i for i in range(f * params_per_input_channel, (f + 1) * params_per_input_channel)])

                new_weights[:] = old_weights[:,
                                 [i for i in range(old_weights.shape[1]) if i not in node_index]]  # 复制剩余的filters的weight
                new_linear_layer.bias.data = old_linear_layer.bias.data
                if torch.cuda.is_available():
                    new_linear_layer.cuda()
                self.net.fc = new_linear_layer

    def forward(self, input):
        if self.training and self.current_epoch == self.mask_training_stop_epoch  and self.step_tracked == 1:  # training of mask is finished
            print('Prune the network.')
            self.prune_net()  # prune filters
        out = super().forward(input)
        return out

    def load_state_dict(self, state_dict, strict=True):
        try:
            msg=super().load_state_dict(state_dict,strict)
        except RuntimeError as e:
            if 'size mismatch' in e.args[0]:
                self.prune_net()
                msg = super().load_state_dict(state_dict, strict)
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
        raise Exception('补全！！！')
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
            image = torch.zeros((1, 3, 32, 32)).to(device)
        net(image)  # record input feature map sizes of each conv
        for h in handle:
            del h
        conv_mod =  None
        for name, mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                assert conv_mod is None, 'Previous conv is not handled.'
                _modules = get_module(model=net, name=name)
                _modules[name.split('.')[-1]] = block_with_mask_weighted_shortcut(mod, map_size[mod]).to(device)  # replace conv with a block
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
            if isinstance(mod, block_with_mask_weighted_shortcut):
                mod.shortcut_mask=torch.mean(mod.mask.abs()).view(-1)


    def print_mask(self):
        channel_num_list=[]
        for name, mod in self.net.named_modules():
            if isinstance(mod, block_with_mask_weighted_shortcut):
                print('shortcut_mask:%f'%float(mod.shortcut_mask),end='\t\t')
                channel_num = torch.sum(mod.mask != 0)
                channel_num_list+=[int(channel_num)]
                print('channel_num:', int(channel_num),end='\t')  # print the number of channels without being pruned
                print(name)
        print(channel_num_list)

    def detach_mask(self):
        for name, mod in self.net.named_modules():
            if isinstance(mod, block_with_mask_weighted_shortcut):
                mod.shortcut_mask = mod.shortcut_mask.detach()  # detach shortcut_mask from computation graph
                mod.mask = mod.mask.detach()  # detach mask from computation graph



