import torch
import train
import config as conf
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import math
import prune_and_train
import measure_flops
import evaluate
import numpy as np
import data_loader
from sklearn import svm
import vgg
import predict_dead_filter
from predict_dead_filter import fc
import prune
import generate_random_data
import resnet
import create_net
import matplotlib.pyplot as plt
import resnet_copied
from torch import optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('./baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
net=checkpoint['net']
#
# # checkpoint = torch.load('./baseline/resnet56_cifar10,accuracy=0.93280.tar')
# checkpoint=torch.load('./baseline/resnet56_cifar10,accuracy=0.94230.tar')
# net = resnet_copied.resnet56().to(device)

# # checkpoint=torch.load('/home/disk_new/model_saved/resnet56_cifar10_DeadNeural_realdata_good_baseline_过得去/代表/sample_num=13300000,accuracy=0.93610，flop=65931914.tar')
# # net=checkpoint['net']
#
net.load_state_dict(checkpoint['state_dict'])
print(checkpoint['highest_accuracy'])
#
prune_and_train.prune_inactive_neural_with_regressor(net=net,
                                     net_name='tmp',
                                     prune_rate=0.2,
                                     load_regressor=True,
                                     dataset_name='cifar10',
                                     filter_preserve_ratio=0.15,
                                     max_filters_pruned_for_one_time=[0.11,0.11,0.11,0.11,0.11,0.11,0.08,0.11,0.11,0.11,0.2,0.2,0.2],
                                     target_accuracy=0.933,
                                     tar_acc_gradual_decent=True,
                                     flop_expected=4e7,
                                     batch_size=1600,
                                     num_epoch=450,
                                     checkpoint_step=3000,
                                     use_random_data=False,
                                     round_for_train=2,
                                     # optimizer=optim.Adam,
                                     # learning_rate=1e-3,
                                     # weight_decay=0
                                     optimizer=optim.SGD,
                                     learning_rate=0.01,
                                     learning_rate_decay=True,
                                     learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
                                     learning_rate_decay_factor=0.5,
                                     )