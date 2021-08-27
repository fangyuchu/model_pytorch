import os,sys
sys.path.append('../')
from framework import train
import os
import torch
import torch.optim as optim
import logger
import sys
from network import resnet_cifar,storage,vgg
from torch import nn
from network import resnet_cifar,resnet,mobilenet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = mobilenet.mobilenet_v1(num_class=1000).cuda()
learning_rate = 0.05
momentum = 0.9
weight_decay = 4e-5
num_epochs = 150
train.train(net=net,
            net_name='mobilenet_v1',
            exp_name='mobilenet_v1_baseline',
            description='',
            dataset_name='imagenet',
            optimizer=optim.SGD,
            criterion=train.CrossEntropyLabelSmooth(num_classes=1000, epsilon=0.1).cuda(),
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            scheduler_name='CosineAnnealingLR',
            num_epochs=num_epochs,
            batch_size=256,
            evaluate_step=2000,
            resume=True,
            test_net=True,
            num_workers=4,
            learning_rate_decay=True,
            top_acc=1,
            data_parallel=True,
            paint_loss=False,
            save_at_each_step=False,
            use_tensorboard=True
            )

# net=resnet_cifar.resnet56().to(device)
# for i in range(3,6):
# net=resnet_cifar.resnet56(num_classes=10).to(device)
# train.train(net=net,
#             net_name='resnet56',
#             exp_name='resnet56_baseline',
#             dataset_name='cifar10',
#             learning_rate=0.1,
#             batch_size=128,
#             momentum=0.9,
#             num_workers=4,
#             learning_rate_decay=True,
#             learning_rate_decay_factor=0.1,
#             # num_epochs=300,
#             # learning_rate_decay_epoch=[150,225],
#             num_epochs=160,
#             learning_rate_decay_epoch=[80, 120],
#             weight_decay=5e-4,
#             resume=True
#             )

# net=vgg.vgg16_bn(dataset_name='cifar10').to(device)
# train.train(net=net,
#             net_name='vgg16_bn',
#             exp_name='vgg16bn_baseline',
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
#             resume=True
#             )




# net=vgg.vgg16_bn(pretrained=False,dataset_name='cifar100').to(device)
# train.train(net,
#             net_name='vgg16_bn',
#             exp_name='vgg16bn_cifar100_baseline',
#             dataset_name='cifar100',
#             resume=False,
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
# print('resnet56 on cifar100')
# net=resnet_cifar.resnet56(num_classes=100).cuda()
# train.train(net,
#             exp_name='resnet56_cifar100_baseline2',
#             dataset_name='cifar100',
#             net_name='resnet56',
#             resume=True,
#             test_net=True,
#             batch_size=128,
#             evaluate_step=8000,
#             # root_path='../data/model_saved/',
#             num_workers=4,
#             num_epochs=300,
#             optimizer=optim.SGD,
#             learning_rate=0.1,
#             learning_rate_decay=True,
#             learning_rate_decay_epoch=[150,225],  # , 150, 250, 300, 350, 400],
#             learning_rate_decay_factor=0.1,
#             weight_decay=5e-4,
#             momentum=0.9,
#             )
#     i+=1

# net=resnet.resnet18(num_classes=10).cuda()
# train.train(net=net,
#             net_name='resnet18',
#             exp_name='resnet18_baseline2',
#             dataset_name='cifar10',
#             learning_rate=0.1,
#             batch_size=128,
#             momentum=0.9,
#             num_workers=4,
#             learning_rate_decay=True,
#             learning_rate_decay_factor=0.1,
#             # num_epochs=300,
#             # learning_rate_decay_epoch=[150,225],
#             num_epochs=160,
#             learning_rate_decay_epoch=[80, 120],
#             weight_decay=5e-4,
#             resume=True
#             )