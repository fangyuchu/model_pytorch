import train
import vgg
import torch.nn as nn
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import config as conf
import torch.optim as optim
import logger
import sys
import resnet_copied
import data_loader
import measure_flops
import prune_and_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoint=torch.load('./baseline/resnet56_cifar100_0.71580.tar')
# checkpoint=torch.load('/home/zengyao/fang/model_pytorch/model_saved/resnet56_cifar100_regressor3/checkpoint/flop=90655076,accuracy=0.71000.tar')
# net=checkpoint['net']
# net.load_state_dict(checkpoint['state_dict'])
# net.to(device)
# measure_flops.measure_model(net,'cifar100')
# prune_and_train.prune_inactive_neural_with_regressor_resnet(net=net,
#                                                             net_name='resnet56_cifar100_regressor3',
#                                                             prune_rate=0.05,
#                                                             load_regressor=True,
#                                                             dataset_name='cifar100',
#                                                             filter_preserve_ratio=0.15,
#                                                             max_filters_pruned_for_one_time=0.2,
#                                                             target_accuracy=0.708,
#                                                             tar_acc_gradual_decent=True,
#                                                             flop_expected=4e7,
#                                                             batch_size=128,
#                                                             num_epoch=250,
#                                                             checkpoint_step=3000,
#                                                             use_random_data=False,
#                                                             round_for_train=3,
#                                                             round=5,
#                                                             # optimizer=optim.Adam,
#                                                             # learning_rate=1e-3,
#                                                             # weight_decay=0
#
#                                                             optimizer=optim.SGD,
#                                                             learning_rate=0.1,
#                                                             learning_rate_decay=True,
#                                                             learning_rate_decay_epoch=[15,35,50,80, 120,160],
#                                                             learning_rate_decay_factor=0.1,
#                                                             weight_decay=1e-4,
#                                                             momentum=0.9,
#                                                             )

# checkpoint = torch.load('./baseline/vgg16bn_cifar100_0.72940.tar')
# checkpoint=torch.load('/home/zzj/fang/model_pytorch/model_saved/vgg16_cifar100_regressor/checkpoint/flop=216253868,accuracy=0.71900.tar')
#
# net = checkpoint['net'].to(device)
#
# net.load_state_dict(checkpoint['state_dict'])
# print(checkpoint['highest_accuracy'])
#
# measure_flops.measure_model(net, 'cifar100', print_flop=True)
# a=[0.05 for i in range(13)]
# a[12]=0.3
# prune_and_train.prune_inactive_neural_with_regressor(net=net,
#                                                      net_name='vgg16_cifar100_regressor',
#                                                      prune_rate=0.15,
#                                                      load_regressor=True,
#                                                      dataset_name='cifar100',
#                                                      filter_preserve_ratio=0.15,
#                                                      max_filters_pruned_for_one_time=a,
#                                                      target_accuracy=0.7177,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=1e8,
#                                                      batch_size=128,
#                                                      num_epoch=250,
#                                                      checkpoint_step=8000,
#                                                      use_random_data=False,
#                                                      round_for_train=2,
#                                                      round=3,
#                                                      # optimizer=optim.Adam,
#                                                      # learning_rate=1e-3,
#                                                      # weight_decay=0
#
#                                                      optimizer=optim.SGD,
#                                                      learning_rate=0.01,  # 标准baseline
#                                                      # learning_rate=0.001,
#                                                      learning_rate_decay=True,
#                                                      learning_rate_decay_epoch=[30, 50, 120, 160, 200],  # 标准baseline
#                                                      # learning_rate_decay_epoch=[10,20,60,90,160],
#                                                      learning_rate_decay_factor=0.2,
#                                                      weight_decay=5e-4,
#                                                      momentum=0.9,
#                                                             )