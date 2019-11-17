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


c=torch.load('/home/disk_new/model_saved/vgg16_bn_weighted_channel/checkpoint/flop=18923530,accuracy=0.70950.tar')


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


net=vgg_channel_weight.vgg16_bn(pretrained=False).to(device)
net_normal=vgg.vgg16_bn(pretrained=False).to(device)


net.classifier=nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )


net_normal.classifier=nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

print("net_new have {} paramerters in total".format(sum(x.numel() for x in net.parameters())))
print("net_normal have {} paramerters in total".format(sum(x.numel() for x in net_normal.parameters())))




for m in net.modules():
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

net.to(device)

train.train(net=net,
            net_name='vgg16_bn_weighted_channel',
            dataset_name='cifar10',
            learning_rate=0.1,
            num_epochs=250,
            batch_size=256,
            checkpoint_step=4000,
            load_net=True,
            test_net=True,
            num_workers=4,
            learning_rate_decay=True,
            learning_rate_decay_factor=0.1,
            learning_rate_decay_epoch=[50,100,150,200])






print()