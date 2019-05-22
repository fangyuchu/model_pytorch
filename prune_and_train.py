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
import numpy as np



def prune_dead_neural(net,
                    net_name,
                      dead_times,
                    dataset_name='imagenet',
                    train_loader=None,
                    validation_loader=None,
                    learning_rate=conf.learning_rate,
                    num_epochs=conf.num_epochs,
                    batch_size=conf.batch_size,
                    checkpoint_step=conf.checkpoint_step,
                    load_net=True,
                    root_path=conf.root_path,
                    checkpoint_path=None,
                    default_image_size=224,
                    momentum=conf.momentum,
                    num_workers=conf.num_workers,
                    learning_rate_decay=False,
                    learning_rate_decay_factor=conf.learning_rate_decay_factor,
                    weight_decay=conf.weight_decay,
                    target_accuracy=1,):
    #todo: not finished
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

    validation_loader = data_loader.create_validation_loader(dataset_path=validation_set_path,
                                                             default_image_size=default_image_size,
                                                             mean=mean,
                                                             std=std,
                                                             batch_size=batch_size,
                                                             num_workers=num_workers,
                                                             dataset_name=dataset_name)
    relu_list,neural_list=evaluate.check_ReLU_alive(net,validation_loader,dead_times)

    for i in range(len(relu_list)):
        for relu_key in list(neural_list.keys()):
            if relu_list[i] is relu_key:
                dead_relu_list=neural_list[relu_key]
                dead_relu_list[dead_relu_list<dead_times]=0
                dead_relu_list[dead_relu_list>=dead_times]=1
                dead_relu_list=np.sum(dead_relu_list,axis=(1,2))




if __name__ == "__main__":
    net=train.create_net('vgg16_bn',True)

    num_conv = 0  # num of conv layers in the net
    for mod in net.features:
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv += 1

    # for i in range(1, 7):
    #     net = select_and_prune_filter(net, layer_index=i, percent_of_pruning=0.1,
    #                                   ord=2)  # prune the model

    file_new='/home/victorfang/Desktop/pytorch_model/vgg16_bn,gradual_pruned/checkpoint/sample_num=64064.tar'
    if os.path.isfile(file_new):
        checkpoint = torch.load(file_new)
        net=checkpoint['net']
        net.load_state_dict(checkpoint['state_dict'])


    iteration=1
    while(True):
        print('{} start iteration:{}'.format(datetime.now(),iteration))
        for i in range(8, num_conv + 1):
            net = select_and_prune_filter(net, layer_index=i, percent_of_pruning=0.1,
                                          ord=2)  # prune the model
            print('{} layer {} pruned'.format(datetime.now(),i))

            validation_loader = data_loader.create_validation_loader(dataset_path=conf.imagenet['validation_set_path'],
                                                                     default_image_size=224,
                                                                     mean=conf.imagenet['mean'],
                                                                     std=conf.imagenet['std'],
                                                                     batch_size=conf.batch_size,
                                                                     num_workers=conf.num_workers,
                                                                     dataset_name='imagenet')
            net_name='vgg16_bn,gradual_pruned'
            checkpoint_path = conf.root_path + net_name + '/checkpoint'
            accuracy = evaluate.evaluate_net(net, validation_loader,
                                             save_net=True,
                                             checkpoint_path=checkpoint_path,
                                             sample_num=0,
                                             target_accuracy=0.7)
            if accuracy < 0.7:

                train.train(net=net,
                            net_name=net_name,
                            num_epochs=1,
                            target_accuracy=0.7,
                            learning_rate=2e-5,
                            load_net=False,
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