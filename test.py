import torch
from torch import nn
from framework import data_loader, evaluate,measure_flops,train
from network import vgg_channel_weight, vgg,storage,resnet,net_with_predicted_mask,resnet_cifar,modules,resnet_cifar,modules
from framework import config as conf
import os,sys
from filter_characteristic import filter_feature_extractor,predict_dead_filter
import numpy as np
from torch import optim
import math
from prune import prune_module
import matplotlib.pyplot as plt
import logger
import copy
from PIL import Image
#ssh -L 16006:127.0.0.1:6006 -p 20029 victorfang@210.28.133.13
# import torchsnooper
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

checkpoint = torch.load('/home/disk_new/model_saved_4gpu/model_saved/gat_resnet50_predicted_mask_and_variable_shortcut_net_newinner_newtrain_85_7/checkpoint/flop=625447007,accuracy=0.70741.pth')
state=checkpoint['state_dict']
param_num=0
for k in state:
    param_num+= state[k].numel()

optimizer_net = optim.SGD
optimizer_extractor = optim.SGD
learning_rate = {'default': 0.1, 'extractor': 0.0001}
weight_decay = {'default': 1e-4, 'extractor': 1e-4}
momentum = {'default': 0.9, 'extractor': 0.9}
batch_size = 256
# 网络参数
add_shortcut_ratio = 0.9  # 不是这儿！！！
mask_update_freq = 1000
mask_update_epochs = 900
mask_training_start_epoch = 1
mask_training_stop_epoch = 3

exp_name = 'gat_resnet50_predicted_mask_and_variable_shortcut_net_mask_newinner_bn_revised_oldreg_9'
description = exp_name + '  ' + '专门训练mask,没有warmup，训练20epoch'

total_flop = 4133641192
prune_ratio = 0.5
flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
gradient_clip_value = None
learning_rate_decay_epoch = [mask_training_stop_epoch + 1 * i for i in [30, 60]]
num_epochs = 90 * 1 + mask_training_stop_epoch

net = resnet.resnet50(pretrained=False)
net=resnet_cifar.resnet56(num_classes=10).cuda()
measure_flops.measure_model(net.cuda(),dataset_name='cifar10')
num=0
for k in net.state_dict():
    num+= net.state_dict()[k].numel()

# batch_size=128
num=0
filter_num=0
#net.extractor.gat.w
for name,mod in net.named_modules():
    if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
        filter_num+=mod.out_channels
        weight_num=int(mod.in_channels/mod.groups*mod.kernel_size[0]*mod.kernel_size[1])
        num+=mod.out_channels* weight_num*15 #for linear
        num+=2*15*mod.out_channels #for bn1d
    print()
#net.extractor.gatlayer
# num += 2*(filter_num*filter_num*15+filter_num*15*15) #for linear
num_filter_ahead=0
for name,mod in net.named_modules():
    if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
        num+=2*15*mod.out_channels*num_filter_ahead #for aggregation
        num_filter_ahead=mod.out_channels
num+=2* 15*filter_num #for activation and softmax
#net.extractor.network
num +=filter_num*15*1 #for linear
num+= filter_num+filter_num*2 #for activation and bn1d

net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                       net_name='resnet50',
                                                                       dataset_name='imagenet',
                                                                       mask_update_epochs=mask_update_epochs,
                                                                       mask_update_freq=mask_update_freq,
                                                                       flop_expected=flop_expected,
                                                                       mask_training_start_epoch=mask_training_start_epoch,
                                                                       mask_training_stop_epoch=mask_training_stop_epoch,
                                                                       batch_size=batch_size,
                                                                       add_shortcut_ratio=add_shortcut_ratio,
                                                                       gcn_layer_num=2
                                                                       )

net = net.cuda()

num=0
state= net.extractor.state_dict()
for k in state:
    num+= state[k].numel()
num+=torch.sum(net.extractor.gat.adj)
print()
# net=mobilenet.mobilenet_v1(num_class=1000).cuda()
# measure_flops.measure_model(net)
# extractor=filter_feature_extractor.extractor(net,feature_len=9).cuda()
# extractor.forward(net)
# # net=resnet_cifar.resnet56(num_classes=100)
# a=np.ndarray([0,1,0])
# b=torch.tensor([0,0,0])
# b[a>0]=1
# print()
# net=mobilenet.mobilenet_v1(num_class=1000).cuda()
# measure_flops.measure_model(net,)
#
#
# optimizer_net = optim.SGD
# optimizer_extractor = optim.SGD
# learning_rate = {'default': 0.1, 'extractor': 0.001}
# weight_decay = {'default': 5e-4, 'extractor': 5e-4}
# momentum = {'default': 0.9, 'extractor': 0.9}
# batch_size = 128
# # 网络参数
# add_shortcut_ratio = 0.9  # 不是这儿！！！
# mask_update_freq = 1000
# mask_update_epochs = 900
# mask_training_start_epoch = 1
# mask_training_stop_epoch = 80
#
# net=mobilenet.mobilenet_v1(num_class=1000)
# # dl=data_loader.create_train_loader(batch_size=10,num_workers=1)
# # for (i,data) in enumerate(data_loader):
# net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
#                                                                        net_name='mobilenet_v1',
#                                                                        dataset_name='imagenet',
#                                                                        mask_update_epochs=mask_update_epochs,
#                                                                        mask_update_freq=mask_update_freq,
#                                                                        flop_expected=578826728*0.1,
#                                                                        gcn_rounds=2,
#                                                                        mask_training_start_epoch=mask_training_start_epoch,
#                                                                        mask_training_stop_epoch=mask_training_stop_epoch,
#                                                                        batch_size=batch_size,
#                                                                        add_shortcut_ratio=add_shortcut_ratio,
#                                                                        feature_len=9,
#                                                                        )
# net.cuda()
# measure_flops.measure_model(net)
# net.mask_net()
# net.print_mask()
# net.prune_net()
print()
# total_flop=126550666#125485706
# prune_ratio=0.93
# flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
# gradient_clip_value=None
# learning_rate_decay_epoch = [mask_training_stop_epoch+1*i for i in [80,120]]
# num_epochs=160*1+mask_training_stop_epoch
#
# #
# net=resnet_cifar.resnet56(num_classes=10).cuda()
# net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
#                                                                        net_name='resnet56',
#                                                                        dataset_name='cifar10',
#                                                                        mask_update_epochs=mask_update_epochs,
#                                                                        mask_update_freq=mask_update_freq,
#                                                                        flop_expected=flop_expected,
#                                                                        gcn_rounds=2,
#                                                                        mask_training_start_epoch=mask_training_start_epoch,
#                                                                        mask_training_stop_epoch=mask_training_stop_epoch,
#                                                                        batch_size=batch_size,
#                                                                        add_shortcut_ratio=add_shortcut_ratio
#                                                                        )
# net=net.cuda()
# i = 4
#
# checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet56',str(i) + '.tar'),map_location='cpu')
# net.load_state_dict(checkpoint['state_dict'])
# net.mask_net()
# net.print_mask()
# net.prune_net()
# net.current_epoch = net.mask_training_stop_epoch + 1
# net.eval()
# dl=data_loader.create_test_loader(batch_size=1,num_workers=0,dataset_name='cifar10',shuffle=True)




# optimizer_net = optim.SGD
# learning_rate = {'default': 0.1, 'extractor': 0.0001}
# weight_decay = {'default': 1e-4, 'extractor': 1e-4}
# momentum = {'default': 0.9, 'extractor': 0.9}
# batch_size = 256
# # 网络参数
# add_shortcut_ratio = 0.9  # 不是这儿！！！
# mask_update_freq = 1000
# mask_update_epochs = 900
# mask_training_start_epoch = 1
# mask_training_stop_epoch = 3
#
# exp_name = 'resnet50_predicted_mask_and_variable_shortcut_net_mask_newinner_5'
# description = exp_name + '  ' + '专门训练mask,没有warmup，训练20epoch'
#
# total_flop = 4133641192
# prune_ratio = 0.5
# flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
# gradient_clip_value = None
# learning_rate_decay_epoch = [mask_training_stop_epoch + 1 * i for i in [30, 60]]
# num_epochs = 90 * 1 + mask_training_stop_epoch
#
# net = resnet.resnet50(pretrained=False)
# net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
#                                                                        net_name='resnet50',
#                                                                        dataset_name='imagenet',
#                                                                        mask_update_epochs=mask_update_epochs,
#                                                                        mask_update_freq=mask_update_freq,
#                                                                        flop_expected=flop_expected,
#                                                                        gcn_rounds=2,
#                                                                        mask_training_start_epoch=mask_training_start_epoch,
#                                                                        mask_training_stop_epoch=mask_training_stop_epoch,
#                                                                        batch_size=batch_size,
#                                                                        add_shortcut_ratio=add_shortcut_ratio
#                                                                        )
#
# net = net.cuda()
#
# i = 6
#
# checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet50', str(i) + '.tar'),
#                         map_location='cpu')
# net.load_state_dict(checkpoint['state_dict'])
#
#
# dl=data_loader.create_test_loader(batch_size=1,num_workers=0,dataset_name='imagenet')



train.show_feature_map(net,dl,0,16)


