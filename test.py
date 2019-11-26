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
# out=torch.Tensor(3,3)
# a=Variable(torch.Tensor([1,2,4]),requires_grad=True)
# b=Variable(torch.Tensor([3,4,7]),requires_grad=True)
# e=Variable(torch.Tensor([3,4,7]),requires_grad=True)
# sdf=Variable(torch.Tensor([2,5,8]),requires_grad=True)
# out[0]=a
# out[1]=b
# out[2]=e
# #
# d=torch.sum(out)
# d.backward()

#
# c=torch.load('/home/disk_new/model_saved/vgg16_bn_weighted_channel/checkpoint/flop=18923530,accuracy=0.93600.tar')
#
# net=c['net']
# net.load_state_dict(c['state_dict'])
# for mod in net.features:
#     if isinstance(mod,nn.Conv2d):
#         print()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# mod=vgg_channel_weight.conv2d_weighted_channel(in_channels=3,out_channels=64,kernel_size=3,padding=1).to(device)
# print(isinstance(mod, nn.Conv2d))
# dl=data_loader.create_validation_loader(batch_size=5,num_workers=4,dataset_name='cifar10')
# for step, data in enumerate(dl, 0):
#     images, labels = data
#     images, labels = images.to(device), labels.to(device)
#     c=mod(images)
#     d=torch.sum(c)
#     d.backward()
#     print()
checkpoint=torch.load('/home/disk_new/model_saved/vgg16_bn_weighted_channel_l1penalty/checkpoint/flop=18923530,accuracy=0.93390_new_version.tar')

net=vgg_channel_weight.vgg16_bn(pretrained=False,dataset='cifar10').to(device)
net.load_state_dict(checkpoint['state_dict'])
evaluate.evaluate_net(net,data_loader=data_loader.create_validation_loader(batch_size=512,num_workers=8,dataset_name='cifar10'),save_net=False)

net.train_channel_weight(if_train=False)
net.prune_channel_weight(percent=[0.3 for i in range(13)])
net.to(device)



evaluate.evaluate_net(net,data_loader=data_loader.create_validation_loader(batch_size=512,num_workers=8,dataset_name='cifar10'),save_net=False)
print()
# train.train(net=net,
#             net_name='vgg16_bn_weighted_channel_l1penalty',
#             dataset_name='cifar10',
#             learning_rate=0.1,
#             num_epochs=250,
#             batch_size=256,
#             checkpoint_step=4000,
#             load_net=True,
#             test_net=True,
#             num_workers=4,
#             learning_rate_decay=True,
#             learning_rate_decay_factor=0.1,
#             learning_rate_decay_epoch=[50,100,150,200],
#             criterion=vgg_channel_weight.CrossEntropyLoss_weighted_channel(net=net,penalty=1e-5))






print()