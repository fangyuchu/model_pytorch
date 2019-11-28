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
from torch.autograd import Variable
import vgg_channel_weight

#
# c=torch.load('/home/disk_new/model_saved/vgg16_bn_weighted_channel/checkpoint/flop=18923530,accuracy=0.93600.tar')
#
# net=c['net']
# net.load_state_dict(c['state_dict'])
# for mod in net.features:
#     if isinstance(mod,nn.Conv2d):
#         print()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





checkpoint=torch.load('/home/disk_new/model_saved/reform_vgg16_bn/checkpoint/flop=530442,accuracy=0.94000.tar')
net=vgg_channel_weight.vgg16_bn(pretrained=False,dataset='cifar10').to(device)
net.load_state_dict(checkpoint['state_dict'])


net.train_channel_weight(if_train=False)
net.prune_channel_weight(percent=[0 for i in range(13)])
net.to(device)

evaluate.evaluate_net(net,data_loader=data_loader.create_validation_loader(batch_size=512,num_workers=8,dataset_name='cifar10'),save_net=False)
print()


checkpoint=torch.load('/home/victorfang/PycharmProjects/model_pytorch/baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
import vgg
net=checkpoint['net']

net.load_state_dict(checkpoint['state_dict'])

vgg_channel_weight.reform_net(net)
net.to(device)
# evaluate.evaluate_net(net,data_loader=data_loader.create_validation_loader(batch_size=512,num_workers=8,dataset_name='cifar10'),save_net=False)



train.train(net=net,
            net_name='reform_vgg16_bn',
            dataset_name='cifar10',
            learning_rate=0.01,
            num_epochs=250,
            batch_size=256,
            checkpoint_step=4000,
            load_net=True,
            test_net=False,
            num_workers=4,
            learning_rate_decay=True,
            learning_rate_decay_factor=0.1,
            learning_rate_decay_epoch=[50,100,150,200],
            # criterion=vgg_channel_weight.CrossEntropyLoss_weighted_channel(net=net,penalty=1e-5))
            criterion=vgg_channel_weight.CrossEntropyLoss_weighted_channel(net=net, penalty=1e-1,piecewise=4)
            # criterion=nn.CrossEntropyLoss()

 )

print()