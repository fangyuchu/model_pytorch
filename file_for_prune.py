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
#
checkpoint = torch.load('./baseline/vgg16bn_cifar100_0.72630_t+v.tar')
# checkpoint=torch.load('/home/zzj/fang/model_pytorch/model_saved/vgg16bn_cifar10_realdata_regressor5_大幅度/checkpoint/flop=49582154,accuracy=0.93330.tar')

net = checkpoint['net'].to(device)

net.load_state_dict(checkpoint['state_dict'])
print(checkpoint['highest_accuracy'])

measure_flops.measure_model(net, 'tiny_imagenet', print_flop=True)

prune_and_train.prune_inactive_neural_with_regressor(net=net,
                                     net_name='vgg16bn_tinyimagenet_prune',
                                     prune_rate=0.15,
                                     load_regressor=False,
                                     dataset_name='tiny_imagenet',
                                     filter_preserve_ratio=0.15,
                                     max_filters_pruned_for_one_time=0.2,
                                     # [0.11,0.11,0.11,0.11,0.11,0.11,0.08,0.11,0.11,0.11,0.2,0.2,0.2],
                                     target_accuracy=0.717,
                                     tar_acc_gradual_decent=True,
                                     flop_expected=1.7e8,
                                     batch_size=1600,
                                     num_epoch=450,
                                     checkpoint_step=3000,
                                     use_random_data=False,
                                     round_for_train=2,
                                     # optimizer=optim.Adam,
                                     # learning_rate=1e-3,
                                     # weight_decay=0
                                     optimizer=optim.SGD,
                                     learning_rate=0.0001,
                                     weight_decay=5e-4,
                                     learning_rate_decay=True,
                                     learning_rate_decay_epoch=[20, 100, 150, 250, 300, 350, 400],
                                     learning_rate_decay_factor=0.5,
                                     max_training_iteration=2
                                     )