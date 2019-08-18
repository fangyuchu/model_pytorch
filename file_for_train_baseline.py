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
# if not os.path.exists('./model_saved/vgg16_on_cifar100'):
#     os.makedirs('./model_saved/vgg16_on_cifar100', exist_ok=True)
# sys.stdout = logger.Logger( './model_saved/vgg16_on_cifar100/log.txt', sys.stdout)
# sys.stderr = logger.Logger( './model_saved/vgg16_on_cifar100/log.txt', sys.stderr)  # redirect std err, if necessary
#
#
#
# net=vgg.vgg16(pretrained=True)
# net.classifier=nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 100),
#         )
# for m in net.modules():
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, 0, 0.01)
#         nn.init.constant_(m.bias, 0)
#
# net=net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#
#
#
# train.train(net,
#             'vgg16_on_cifar100',
#             'cifar100',
#             load_net=True,
#             batch_size=1600,
#             checkpoint_step=8000,
#             root_path='./model_saved/',
#             num_workers=8,
#             weight_decay=0,
#             num_epochs=450,
#             optimizer=optim.SGD,
#             learning_rate=0.01,
#             learning_rate_decay=True,
#             learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
#             learning_rate_decay_factor=0.5,
#             )


if not os.path.exists('./model_saved/resnet56_on_cifar100'):
    os.makedirs('./model_saved/resnet56_on_cifar100', exist_ok=True)
sys.stdout = logger.Logger( './model_saved/resnet56_on_cifar100/log.txt', sys.stdout)
sys.stderr = logger.Logger( './model_saved/resnet56_on_cifar100/log.txt', sys.stderr)  # redirect std err, if necessary
net=resnet_copied.ResNet(resnet_copied.BasicBlock, [9, 9, 9],num_classes=100)
net=net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
train.train(net,
            'resnet56_on_cifar100',
            'cifar100',
            load_net=True,
            batch_size=1600,
            checkpoint_step=8000,
            root_path='./model_saved/',
            num_workers=8,
            weight_decay=0,
            num_epochs=450,
            optimizer=optim.SGD,
            learning_rate=0.1,
            learning_rate_decay=True,
            learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
            learning_rate_decay_factor=0.5,
            )
