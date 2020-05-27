from framework import train
import os
import torch
import torch.optim as optim
import logger
import sys
from network import resnet_cifar,storage,vgg
from torch import nn
from network import resnet_cifar
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net=resnet_cifar.resnet56().to(device)
net=resnet_cifar.resnet56(num_classes=10).to(device)

train.train(net=net,
            net_name='resnet56',
            exp_name='resnet56_baseline_sfp_heSchedule_1',
            dataset_name='cifar10',
            learning_rate=0.1,
            num_epochs=160,
            batch_size=128,
            momentum=0.9,
            num_workers=2,
            learning_rate_decay=True,
            learning_rate_decay_factor=0.1,
            # learning_rate_decay_epoch=[150,225],
            learning_rate_decay_epoch=[80, 120],

            weight_decay=1e-4,
            load_net=True
            )

# net=vgg.vgg16_bn(dataset_name='cifar10').to(device)
# train.train(net=net,
#             net_name='vgg16_bn',
#             exp_name='test',
#             dataset_name='cifar10',
#             learning_rate=0.1,
#             num_epochs=200,
#             batch_size=128,
#             momentum=0.9,
#             num_workers=2,
#             learning_rate_decay=True,
#             learning_rate_decay_factor=0.2,
#             learning_rate_decay_epoch=[60,120,160],
#             weight_decay=1e-4,
#             load_net=True
#             )




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
# net=resnet_cifar.resnet32(num_classes=100)
# net=net.to(device)
#
#
#
# i=0
# while i<5:
#     train.train(net,
#                 exp_name='resnet32_cifar100_baseline_'+str(i+3),
#                 dataset_name='cifar100',
#                 net_name='resnet32',
#                 load_net=True,
#                 test_net=True,
#                 batch_size=256,
#                 evaluate_step=8000,
#                 # root_path='../data/model_saved/',
#                 num_workers=4,
#                 num_epochs=300,
#                 optimizer=optim.SGD,
#                 learning_rate=0.1,
#                 learning_rate_decay=True,
#                 learning_rate_decay_epoch=[80, 90, 120],  # , 150, 250, 300, 350, 400],
#                 learning_rate_decay_factor=0.1,
#                 weight_decay=5e-4,
#                 momentum=0.9,
#                 )
#     i+=1