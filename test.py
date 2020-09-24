import torch
from torch import nn
from framework import data_loader, evaluate,measure_flops,train
from network import vgg_channel_weight, vgg,storage,resnet,net_with_predicted_mask,resnet_cifar,modules,resnet_cifar,modules
from framework import config as conf
import os,sys
from filter_characteristic import filter_feature_extractor,predict_dead_filter
import numpy as np
from torch import optim
from prune import prune_module
import matplotlib.pyplot as plt
import logger
import copy
#ssh -L 16006:127.0.0.1:6006 -p 20029 victorfang@210.28.133.13
# import torchsnooper
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# net=vgg.vgg16_bn(dataset_name='imagenet').cuda()
net=resnet.resnet50().cuda()
measure_flops.measure_model(net,'imagenet')

optimizer_net = optim.SGD
optimizer_extractor = optim.SGD
learning_rate = {'default': 0.1, 'extractor': 0.001}
weight_decay = {'default': 1e-4, 'extractor': 5e-4}
momentum = {'default': 0.9, 'extractor': 0.9}
batch_size = 128
# 网络参数
add_shortcut_ratio = 0.9  # 不是这儿！！！
mask_update_freq = 1000
mask_update_epochs = 900
mask_training_start_epoch = 1
mask_training_stop_epoch = 80

exp_name = 'resnet56_predicted_mask_and_variable_shortcut_net_mask_newinner_80epoch_7'
description = exp_name + '  ' + '专门训练mask,没有warmup，训练20epoch'

total_flop = 126550666
prune_ratio = 0.9
flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
gradient_clip_value = None
learning_rate_decay_epoch = [mask_training_stop_epoch + 1 * i for i in [80, 120]]
num_epochs = 160 * 1 + mask_training_stop_epoch
#
net = resnet_cifar.resnet56(num_classes=10).to(device)
net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                       net_name='resnet56',
                                                                       dataset_name='cifar10',
                                                                       mask_update_epochs=mask_update_epochs,
                                                                       mask_update_freq=mask_update_freq,
                                                                       flop_expected=flop_expected,
                                                                       gcn_rounds=2,
                                                                       mask_training_start_epoch=mask_training_start_epoch,
                                                                       mask_training_stop_epoch=mask_training_stop_epoch,
                                                                       batch_size=batch_size,
                                                                       add_shortcut_ratio=add_shortcut_ratio
                                                                       )
net = net.to(device)

i = 4
checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)


checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet56', str(i) + '.tar'), map_location='cpu')
net.load_state_dict(checkpoint['state_dict'])
net.mask_net()
net.print_mask()
net.prune_net()
net.print_in_out_channels()
net.current_epoch = net.mask_training_stop_epoch + 1
# measure_flops.measure_model(net, dataset_name='cifar10', print_flop=True)
measure_flops.measure_model(net.net,dataset_name='cifar10',print_flop=True)
print()
