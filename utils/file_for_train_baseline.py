from framework import train
import os
import torch
import torch.optim as optim
import logger
import sys
from network import resnet_cifar,storage,vgg
from torch import nn
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net=vgg.vgg16_bn(pretrained=False,dataset_name='cifar100').to(device)
# train.train(net,
#             net_name='vgg16_bn',
#             exp_name='vgg16bn_cifar100_baseline2',
#             dataset_name='cifar100',
#             load_net=True,
#             test_net=True,
#             batch_size=512,
#             evaluate_step=8000,
#             # root_path='../data/model_saved/',
#             num_workers=2,
#             num_epochs=200,
#             learning_rate=0.1,
#             learning_rate_decay_epoch=[60,120,160],
#             learning_rate_decay_factor=0.2,
#
#             optimizer=optim.SGD,
#             # learning_rate=0.1,    #标准baseline
#             learning_rate_decay=True,
#             weight_decay=5e-3,
#             momentum=0.9,
#             )
# torch.optim.lr_scheduler.MultiStepLR

# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# net=resnet_cifar.resnet56(num_classes=100)
# net=net.to(device)
# train.train(net,
#             exp_name='resnet56_cifar100_baseline_2' ,
#             dataset_name='cifar100',
#             net_name='resnet56',
#             load_net=True,
#             test_net=True,
#             batch_size=512,
#             evaluate_step=8000,
#             # root_path='../data/model_saved/',
#             num_workers=4,
#             num_epochs=200,
#             optimizer=optim.SGD,
#             learning_rate=0.1,
#             learning_rate_decay=True,
#             learning_rate_decay_epoch=[80, 120],  #, 150, 250, 300, 350, 400],
#             learning_rate_decay_factor=0.1,
#             weight_decay=5e-4,
#             momentum=0.9,
#             )
i=0
while i<5:
    net=resnet_cifar.resnet56(num_classes=10)
    net=net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    train.train(net,
                exp_name='resnet56_cifar10_baseline+'+str(i) ,
                dataset_name='cifar10',
                net_name='resnet56',
                load_net=True,
                test_net=True,
                batch_size=512,
                evaluate_step=8000,
                # root_path='../data/model_saved/',
                num_workers=4,
                num_epochs=250,
                optimizer=optim.SGD,
                learning_rate=0.1,
                learning_rate_decay=True,
                learning_rate_decay_epoch=[100, 200],  #, 150, 250, 300, 350, 400],
                learning_rate_decay_factor=0.1,
                weight_decay=5e-4,
                momentum=0.9,
                )
    i+=1