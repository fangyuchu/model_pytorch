import config as conf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import resnet
import vgg
import generate_random_data
import os
import re
from datetime import datetime
from prune import select_and_prune_filter
import train
import evaluate
import data_loader
import numpy as np
import prune
import measure_flops
import logger
import sys
import create_net
import random
import copy



def prune_dead_neural(net,
                      net_name,
                      neural_dead_times,
                      filter_dead_ratio,
                      target_accuracy,
                      tar_acc_gradual_decent=False,
                      flop_expected=None,
                      filter_dead_ratio_decay=0.95,
                      neural_dead_times_decay=0.95,
                      dataset_name='imagenet',
                      use_random_data=False,
                      validation_loader=None,
                      batch_size=conf.batch_size,
                      num_workers=conf.num_workers,
                      optimizer=optim.Adam,
                      learning_rate=0.01,
                      checkpoint_step=1000,
                      num_epoch=350,
                      filter_preserve_ratio=0.3,
                      max_filters_pruned_for_one_time=0.5,
                      learning_rate_decay=False,
                      learning_rate_decay_factor=conf.learning_rate_decay_factor,
                      weight_decay=conf.weight_decay,
                      learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
                     ):
    '''

    :param net:
    :param net_name:
    :param neural_dead_times(ndt):int, threshold for judging a dead neural node
    :param filter_dead_ratio(fdr):float, threshold for judging a dead filter
    :param target_accuracy: float,
    :param tar_acc_gradual_decent:bool, if true, the target accuracy will decent from original acc. to target acc. during every round of pruning
    :param flop_expected: int: expected flop after net pruned. will only work when tar_acc_gradual_decent is true
    :param filter_dead_ratio_decay:float, decay rate for fdr in each round of pruning
    :param neural_dead_times_decay:float, decay rate for ndt in each round of pruning
    :param dataset_name:
    :param use_random_data:bool, if true, generated data which fits normal distribution will be used to calculate dead filters.
    :param validation_loader:
    :param batch_size:
    :param num_workers:
    :param optimizer:
    :param learning_rate:
    :param checkpoint_step:
    :param num_epoch:
    :param filter_preserve_ratio:
    :param max_filters_pruned_for_one_time:
    :param learning_rate_decay:
    :param learning_rate_decay_factor:
    :param weight_decay:
    :param learning_rate_decay_epoch:
    :return:
    '''
    #save the output to log
    print('save log in:' + conf.root_path + net_name + '/log.txt')
    if not os.path.exists(conf.root_path + net_name ):
        os.makedirs(conf.root_path + net_name , exist_ok=True)
    sys.stdout = logger.Logger(conf.root_path+net_name+'/log.txt', sys.stdout)
    sys.stderr = logger.Logger(conf.root_path+net_name+'/log.txt', sys.stderr)  # redirect std err, if necessary

    print('net:{}\n'
          'net_name:{}\n'
          'use_random_data:{}\n'
          'neural_dead_times:{}\n'
          'filter_dead_ratio:{}\n'
          'target_accuracy:{}\n'
          'tar_acc_gradual_decent:{}\n'
          'flop_expected:{}\n'
          'filter_dead_ratio_decay:{}\n'
          'neural_dead_times_decay:{}\n'
          'dataset_name:{}\n'
          'validation_loader:{}\n'
          'batch_size:{}\n'
          'num_workers:{}\n'
          'optimizer:{}\n'
          'learning_rate:{}\n'
          'checkpoint_step:{}\n'
          'num_epoch:{}\n'
          'filter_preserve_ratio:{}\n'
          'max_filters_pruned_for_one_time:{}\n'
          'learning_rate_decay:{}\n'
          'learning_rate_decay_factor:{}\n'
          'weight_decay:{}\n'
          'learning_rate_decay_epoch:{}'
          .format(net,net_name,use_random_data,neural_dead_times,filter_dead_ratio,target_accuracy,tar_acc_gradual_decent,
                  flop_expected,filter_dead_ratio_decay,
                  neural_dead_times_decay,dataset_name,validation_loader,batch_size,num_workers,optimizer,learning_rate,checkpoint_step,
                  num_epoch,filter_preserve_ratio,max_filters_pruned_for_one_time,learning_rate_decay,learning_rate_decay_factor,
                  weight_decay,learning_rate_decay_epoch))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ', end='')
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print(device)

    # prepare the data
    if dataset_name is 'imagenet':
        mean = conf.imagenet['mean']
        std = conf.imagenet['std']
        train_set_path = conf.imagenet['train_set_path']
        train_set_size = conf.imagenet['train_set_size']
        validation_set_path = conf.imagenet['validation_set_path']
        validation_set_size = conf.imagenet['validation_set_size']
        default_image_size=conf.imagenet['default_image_size']
    elif dataset_name is 'cifar10':
        train_set_size = conf.cifar10['train_set_size']
        mean = conf.cifar10['mean']
        std = conf.cifar10['std']
        train_set_path = conf.cifar10['train_set_path']
        validation_set_path = conf.cifar10['validation_set_path']
        validation_set_size = conf.cifar10['validation_set_size']
        default_image_size=conf.cifar10['default_image_size']


    if validation_loader is None :
        validation_loader = data_loader.create_validation_loader(dataset_path=validation_set_path,
                                                                 default_image_size=default_image_size,
                                                                 mean=mean,
                                                                 std=std,
                                                                 batch_size=batch_size,
                                                                 num_workers=num_workers,
                                                                 dataset_name=dataset_name,
                                                                 )

    flop_original_net=measure_flops.measure_model(net,'cifar10')
    original_accuracy=evaluate.evaluate_net(net=net,
                                            data_loader=validation_loader,
                                            save_net=False)
    flop_drop_expected = flop_original_net - flop_expected
    acc_drop_tolerance = original_accuracy - target_accuracy


    num_conv = 0  # num of conv layers in the net
    filter_num_lower_bound=list()
    filter_num=list()
    for mod in net.features:
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv += 1
            filter_num_lower_bound.append(int(mod.out_channels*filter_preserve_ratio))
            filter_num.append(mod.out_channels)

    round=0
    while True:
        round+=1
        print('{} start round {} of filter pruning.'.format(datetime.now(),round))
        print('{} current filter_dead_ratio:{},neural_dead_times:{}'.format(datetime.now(),filter_dead_ratio,neural_dead_times))
        if use_random_data is True:
            random_data=generate_random_data.random_normal(batch_size)
            print('{} generate random data.'.format(datetime.now()))
            relu_list, neural_list = evaluate.check_ReLU_alive(net=net, neural_dead_times=neural_dead_times, data=random_data)
            del random_data
        else:
            relu_list,neural_list=evaluate.check_ReLU_alive(net=net,data_loader=validation_loader,neural_dead_times=neural_dead_times)

        if not os.path.exists(conf.root_path+net_name+'/dead_neural'):
            os.makedirs(conf.root_path+net_name+'/dead_neural', exist_ok=True)

        torch.save({'neural_dead_times':neural_dead_times,'filter_dead_ratio':filter_dead_ratio,
                    'net':net,'relu_list':relu_list,
                    'neural_list':neural_list,'state_dict':net.state_dict()},
                   conf.root_path+net_name+'/dead_neural/round %d.tar'%round,)
        net_compressed=False
        for i in range(num_conv):
            for relu_key in list(neural_list.keys()):
                if relu_list[i] is relu_key:                                    #find the neural_list_statistics in layer i+1
                    dead_relu_list=neural_list[relu_key]
                    neural_num=dead_relu_list.shape[1]*dead_relu_list.shape[2]  #neural num for one filter


                    # judge dead filter by neural_dead_times and dead_filter_ratio
                    dead_relu_list[dead_relu_list<neural_dead_times]=0
                    dead_relu_list[dead_relu_list>=neural_dead_times]=1
                    dead_relu_list=np.sum(dead_relu_list,axis=(1,2))            #count the number of dead neural for one filter
                    dead_filter_index=np.where(dead_relu_list>=neural_num*filter_dead_ratio)[0].tolist()
                    #ensure the number of filters pruned will not be too large for one time
                    if filter_num[i]*max_filters_pruned_for_one_time<len(dead_filter_index):
                        dead_filter_index = dead_filter_index[:int(filter_num[i] *max_filters_pruned_for_one_time)]
                    #ensure the lower bound of filter number
                    if filter_num[i]-len(dead_filter_index)<filter_num_lower_bound[i]:
                        dead_filter_index=dead_filter_index[:filter_num[i]-filter_num_lower_bound[i]]
                    filter_num[i]=filter_num[i]-len(dead_filter_index)
                    if len(dead_filter_index)>0:
                        net_compressed=True
                    print('layer {}: remain {} filters, prune {} filters.'.format(i, filter_num[i],
                                                                                  len(dead_filter_index)))

                    net=prune.prune_conv_layer(model=net,layer_index=i+1,filter_index=dead_filter_index)    #prune the dead filter

        if net_compressed is False:
            os.remove(conf.root_path+net_name+'/dead_neural/round %d.tar'%round)
            round-=1
            filter_dead_ratio *= filter_dead_ratio_decay
            neural_dead_times *= neural_dead_times_decay
            print('{} round {} did not prune any filters. Restart.'.format(datetime.now(),round+1))
            continue

        flop_pruned_net=measure_flops.measure_model(net,'cifar10')

        #todo:现在根据剪了多少浮点量来降低准确率，也可考虑根据剪的轮数来降低准确率
        if tar_acc_gradual_decent is True:                                      #decent the target_accuracy
            flop_reduced=flop_original_net-flop_pruned_net
            target_accuracy=original_accuracy-acc_drop_tolerance*(flop_reduced/flop_drop_expected)
            print('{} current target accuracy:{}'.format(datetime.now(),target_accuracy))


        success=False
        while not success:
            old_net=copy.deepcopy(net)
            success=train.train(net=net,
                        net_name=net_name,
                        num_epochs=num_epoch,
                        target_accuracy=target_accuracy,
                        learning_rate=learning_rate,
                        load_net=False,
                        checkpoint_step=checkpoint_step,
                        dataset_name=dataset_name,
                        optimizer=optimizer,
                        batch_size=batch_size,
                        learning_rate_decay=learning_rate_decay,
                        learning_rate_decay_factor=learning_rate_decay_factor,
                        weight_decay=weight_decay,
                        learning_rate_decay_epoch=learning_rate_decay_epoch,
                        test_net=True,
                        )
            if not success:
                net=old_net
        filter_dead_ratio*=filter_dead_ratio_decay
        neural_dead_times*=neural_dead_times_decay


def prune_filters_randomly(net,
                           net_name,
                           target_accuracy,
                           round_of_prune,
                           final_filter_num=[23,59,65,65,149,132,99,181,125,116,181,254,223],
                           dataset_name='cifar10',
                           batch_size=conf.batch_size,
                           num_workers=conf.num_workers,
                           optimizer=optim.Adam,
                           learning_rate=0.01,
                           checkpoint_step=1000,
                           num_epoch=350,
                           learning_rate_decay=False,
                           learning_rate_decay_factor=conf.learning_rate_decay_factor,
                           weight_decay=conf.weight_decay,
                           learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
                           ):
    '''

    :param net:
    :param net_name:
    :param target_accuracy:
    :param round_of_prune:
    :param final_filter_num:list[int], the number of filters in each layer after pruning
    :param dataset_name:
    :param batch_size:
    :param num_workers:
    :param optimizer:
    :param learning_rate:
    :param checkpoint_step:
    :param num_epoch:
    :param learning_rate_decay:
    :param learning_rate_decay_factor:
    :param weight_decay:
    :param learning_rate_decay_epoch:
    :return:
    '''
    # save the output to log
    print('save log in:' + conf.root_path + net_name + '/log.txt')
    if not os.path.exists(conf.root_path + net_name):
        os.makedirs(conf.root_path + net_name, exist_ok=True)
    sys.stdout = logger.Logger(conf.root_path + net_name + '/log.txt', sys.stdout)
    sys.stderr = logger.Logger(conf.root_path + net_name + '/log.txt', sys.stderr)  # redirect std err, if necessary

    print('net:{}\n'
          'net_name:{}\n'
          'target_accuracy:{}\n'
          'round of prune:{}\n'
          'final_filter_num:{}\n'
          'dataset_name:{}\n'
          'batch_size:{}\n'
          'num_workers:{}\n'
          'optimizer:{}\n'
          'learning_rate:{}\n'
          'checkpoint_step:{}\n'
          'num_epoch:{}\n'
          'learning_rate_decay:{}\n'
          'learning_rate_decay_factor:{}\n'
          'weight_decay:{}\n'
          'learning_rate_decay_epoch:{}'
          .format(net, net_name,   target_accuracy,
                  round_of_prune,final_filter_num,
                  dataset_name, batch_size, num_workers, optimizer,
                  learning_rate, checkpoint_step,
                  num_epoch, learning_rate_decay,
                  learning_rate_decay_factor,
                  weight_decay, learning_rate_decay_epoch))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ', end='')
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print(device)

    num_conv = 0  # num of conv layers in the net
    filter_num=list()
    for mod in net.features:
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv += 1
            filter_num.append(mod.out_channels)

    round=0
    while round<round_of_prune:
        round+=1
        print('{} start round {} of random filter pruning.'.format(datetime.now(),round))
        for i in range(num_conv):
            num_to_prune=int((filter_num[i]-final_filter_num[i])/(round_of_prune-round+1))           #number of filters to prune in this round
            filter_num[i]=filter_num[i]-num_to_prune                                            #update the number of filters in layer i
            pruning_filter_index=random.sample(range(0,filter_num[i]),num_to_prune)
            print('layer {}: remain {} filters, prune {} filters.'.format(i, filter_num[i],
                                                                          num_to_prune))
            net = prune.prune_conv_layer(model=net, layer_index=i + 1,
                                         filter_index=pruning_filter_index)  # prune the dead filter

        measure_flops.measure_model(net, 'cifar10')

        success=False
        while not success:
            old_net = copy.deepcopy(net)
            success = train.train(net=net,
                                  net_name=net_name,
                                  num_epochs=num_epoch,
                                  target_accuracy=target_accuracy,
                                  learning_rate=learning_rate,
                                  load_net=False,
                                  checkpoint_step=checkpoint_step,
                                  dataset_name=dataset_name,
                                  optimizer=optimizer,
                                  batch_size=batch_size,
                                  learning_rate_decay=learning_rate_decay,
                                  learning_rate_decay_factor=learning_rate_decay_factor,
                                  weight_decay=weight_decay,
                                  learning_rate_decay_epoch=learning_rate_decay_epoch,
                                  test_net=True,
                                  )
            if not success:
                net = old_net


def prune_layer_gradually():
    net = train.create_net('vgg16_bn', True)

    num_conv = 0  # num of conv layers in the net
    for mod in net.features:
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv += 1

    # for i in range(1, 7):
    #     net = select_and_prune_filter(net, layer_index=i, percent_of_pruning=0.1,
    #                                   ord=2)  # prune the model

    file_new = '/home/victorfang/Desktop/pytorch_model/vgg16_bn,gradual_pruned/checkpoint/sample_num=64064.tar'
    if os.path.isfile(file_new):
        checkpoint = torch.load(file_new)
        net = checkpoint['net']
        net.load_state_dict(checkpoint['state_dict'])

    iteration = 1
    while (True):
        print('{} start iteration:{}'.format(datetime.now(), iteration))
        for i in range(10, num_conv + 1):
            net = select_and_prune_filter(net, layer_index=i, percent_of_pruning=0.1,
                                          ord=2)  # prune the model
            print('{} layer {} pruned'.format(datetime.now(), i))

            validation_loader = data_loader.create_validation_loader(dataset_path=conf.imagenet['validation_set_path'],
                                                                     default_image_size=224,
                                                                     mean=conf.imagenet['mean'],
                                                                     std=conf.imagenet['std'],
                                                                     batch_size=conf.batch_size,
                                                                     num_workers=conf.num_workers,
                                                                     dataset_name='imagenet')
            net_name = 'vgg16_bn,gradual_pruned'
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
        iteration += 1


if __name__ == "__main__":
    net=create_net.vgg_cifar10('vgg16_bn',pretrained=True)
    prune_filters_randomly(net,net_name='vgg16_bn_debug_test',target_accuracy=0.931,round_of_prune=6,dataset_name='cifar10')

    # prune_and_train(model_name='vgg16_bn',
    #                 pretrained=True,
    #                 checkpoint_step=5000,
    #                 percent_of_pruning=0.9,
    #                 num_epochs=20,
    #                 learning_rate=0.005)