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

checkpoint = torch.load('/home/victorfang/PycharmProjects/model_pytorch/model_saved/vgg16bn_imagenet_prune_train/checkpoint/flop=12804890484,accuracy=0.90876.tar')
# checkpoint=torch.load('/home/victorfang/PycharmProjects/model_pytorch/model_saved/vgg16bn_tinyimagenet_prune_train/checkpoint/flop=11394264872,accuracy=0.71270.tar')
net=checkpoint['net']
# net=resnet_copied.resnet56()

#
# # checkpoint = torch.load('./baseline/resnet56_cifar10,accuracy=0.93280.tar')
# checkpoint=torch.load('./baseline/resnet56_cifar10,accuracy=0.94230.tar')
# net = resnet_copied.resnet56().to(device)

# # checkpoint=torch.load('/home/disk_new/model_saved/resnet56_cifar10_DeadNeural_realdata_good_baseline_过得去/代表/sample_num=13300000,accuracy=0.93610，flop=65931914.tar')
# # net=checkpoint['net']
#
net.load_state_dict(checkpoint['state_dict'])
print(checkpoint['highest_accuracy'])
train.train(net=net,
            net_name='vgg16bn_imagenet_prune_train2',
            dataset_name='imagenet',
            test_net=True,
            num_epochs=20,
            checkpoint_step=4000,
            target_accuracy=0.9140571220008835,
            batch_size=24,
            top_acc=5,


            optimizer=optim.SGD,
            learning_rate=0.0001,
            # weight_decay=0.0006,
            momentum=0.9,
            learning_rate_decay=True,
            learning_rate_decay_epoch=[5, 10, 15],
            learning_rate_decay_factor=0.1,
            )