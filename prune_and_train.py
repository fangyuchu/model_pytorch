import config as conf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import resnet
import vgg
import os
import re
from datetime import datetime
from prune import select_and_prune_filter
import train
import evaluate
import data_loader


#todo:太老了，有时间改吧


def prune_dead_neural(net,dataset_path,dataset_name='imagenet'):
    # prepare the data
    if dataset_name is 'imagenet':
        mean = conf.imagenet['mean']
        std = conf.imagenet['std']
        train_set_path = conf.imagenet['train_set_path']
        train_set_size = conf.imagenet['train_set_size']
        validation_set_path = conf.imagenet['validation_set_path']
    elif dataset_name is 'cifar10':
        train_set_size = conf.cifar10['train_set_size']
        mean = conf.cifar10['mean']
        std = conf.cifar10['std']
        train_set_path = conf.cifar10['train_set_path']
        validation_set_path = conf.cifar10['validation_set_path']


    val_loader=data_loader.create_validation_loader(dataset_path=dataset_path,)
    evaluate.check_ReLU_alive(net,)


if __name__ == "__main__":
    net=train.create_net('vgg16_bn',True)

    num_conv = 0  # num of conv layers in the net
    for mod in net.features:
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv += 1

    for i in range(1, 7):
        net = select_and_prune_filter(net, layer_index=i, percent_of_pruning=0.1,
                                      ord=2)  # prune the model



    iteration=1
    while(True):
        print('{} start iteration:{}'.format(datetime.now(),iteration))
        for i in range(7, num_conv + 1):
            net = select_and_prune_filter(net, layer_index=i, percent_of_pruning=0.1,
                                          ord=2)  # prune the model
            print('{} layer {} pruned'.format(datetime.now(),i))
            train.train(net=net,
                        net_name='vgg16_bn,gradual_pruned',
                        num_epochs=1,
                        target_accuracy=0.7,
                        learning_rate=1e-4,
                        load_net=True,
                        checkpoint_step=1000
                        )
        break
        iteration+=1
    # prune_and_train(model_name='vgg16_bn',
    #                 pretrained=True,
    #                 checkpoint_step=5000,
    #                 percent_of_pruning=0.9,
    #                 num_epochs=20,
    #                 learning_rate=0.005)