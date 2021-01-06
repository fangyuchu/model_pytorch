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
from sklearn import manifold
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# net=resnet_cifar.resnet56(num_classes=100)

exp_name = 'gat_resnet56_predicted_mask_and_variable_shortcut_net_mask_newinner_20epoch_std_3'
description = exp_name + '  ' + '专门训练mask,没有warmup，训练20epoch'
optimizer_net = optim.SGD
optimizer_extractor = optim.SGD
learning_rate = {'default': 0.1, 'extractor': 0.001}
weight_decay = {'default': 5e-4, 'extractor': 5e-4}
momentum = {'default': 0.9, 'extractor': 0.9}
batch_size = 128
# 网络参数
add_shortcut_ratio = 0.9  # 不是这儿！！！
mask_update_freq = 1000
mask_update_epochs = 900
mask_training_start_epoch = 1
mask_training_stop_epoch = 20
total_flop = 126550666  # 125485706
prune_ratio = 0.7
flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
gradient_clip_value = 5
learning_rate_decay_epoch = [mask_training_stop_epoch + 1 * i for i in [80, 120]]
num_epochs = 160 * 1 + mask_training_stop_epoch
#
net = resnet_cifar.resnet56(num_classes=10).cuda()
net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                       net_name='resnet56',
                                                                       dataset_name='cifar10',
                                                                       mask_update_epochs=mask_update_epochs,
                                                                       mask_update_freq=mask_update_freq,
                                                                       flop_expected=flop_expected,
                                                                       gcn_layer_num=2,
                                                                       mask_training_start_epoch=mask_training_start_epoch,
                                                                       mask_training_stop_epoch=mask_training_stop_epoch,
                                                                       batch_size=batch_size,
                                                                       add_shortcut_ratio=add_shortcut_ratio
                                                                       )
net = net.cuda()

i = 1
exp_name = 'gat_resnet56_predicted_mask_and_variable_shortcut_net_doubleschedule' + str(
    int(prune_ratio * 100)) + '_' + str(i)
description = exp_name + '  ' + ''

checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)


checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet56', str(i) + '.pth'), map_location='cpu')
net.load_state_dict(checkpoint['state_dict'])


net.t_sne()


print()

