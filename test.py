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


checkpoint=torch.load('/home/victorfang/Desktop/vgg16_bn_imagenet_deadReLU.tar')
neural_list=checkpoint['neural_list']
relu_list=checkpoint['relu_list']


# net=create_net.vgg_cifar10()
# val_loader=data_loader.create_validation_loader(batch_size=1000,num_workers=6,dataset_name='cifar10')
# train_loader=data_loader.create_train_loader(batch_size=1600,num_workers=6,dataset_name='cifar10')
#
# relu_list,neural_list=evaluate.check_ReLU_alive(net=net,neural_dead_times=50000,data_loader=train_loader)
ndt_list=[i for i in range(35000,51000,1000)]
dead_rate=list()
for ndt in ndt_list:
    print(ndt)
    dead_rate.append(evaluate.cal_dead_neural_rate(neural_dead_times=ndt,neural_list_temp=neural_list))

plt.figure()
plt.title('df')
plt.plot(ndt_list,dead_rate)
plt.xlabel('filter activation ratio')
plt.ylabel('number of filters')
plt.legend()
plt.show()


