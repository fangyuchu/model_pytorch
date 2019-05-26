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

# checkpoint=torch.load('/home/victorfang/Desktop/pytorch_model/vgg16_bn_cifar10_dead_neural_pruned/checkpoint/sample_num=9950000,accuracy=0.930.tar')
# net=checkpoint['net']
# net.load_state_dict(checkpoint['state_dict'])
# accuracy=checkpoint['highest_accuracy']
# print(accuracy)
# measure_flops.measure_model(model=net,dataset_name='cifar10')





neural_dead_times=40000
filter_dead_ratio=0.9

checkpoint = torch.load('/home/victorfang/Desktop/vgg19_imagenet_deadReLU.tar')
# # checkpoint = torch.load('/home/victorfang/Desktop/pytorch_model/test5/checkpoint/sample_num=7800000,accuracy=0.911.tar')
#
# net = checkpoint['net']
# net.load_state_dict(checkpoint['state_dict'])
# print(checkpoint['highest_accuracy'])
# measure_flops.measure_model(model=net,dataset_name='cifar10')
# relu_list,neural_list=evaluate.check_ReLU_alive(net=net,
#                           data_loader=data_loader.create_validation_loader(batch_size=1024,num_workers=6,dataset_name='cifar10'),
#                           neural_dead_times=neural_dead_times)

net=checkpoint['net']
relu_list=checkpoint['relu_list']
neural_list=checkpoint['neural_list']

evaluate.cal_dead_neural_rate(neural_dead_times=neural_dead_times,neural_list_temp=neural_list)
num_conv = 0  # num of conv layers in the net
filter_num_lower_bound = list()
filter_num = list()
for mod in net.features:
    if isinstance(mod, torch.nn.modules.conv.Conv2d):
        num_conv += 1
        filter_num_lower_bound.append(int(mod.out_channels * 0.1))
        filter_num.append(mod.out_channels)

total_filters=np.sum(filter_num)

for i in range(num_conv):
    for relu_key in list(neural_list.keys()):
        if relu_list[i] is relu_key:                                    #find the neural_list_statistics in layer i+1
            dead_relu_list=neural_list[relu_key]
            neural_num=dead_relu_list.shape[1]*dead_relu_list.shape[2]  #neural num for one filter


            # judge dead filter by neural_dead_times and dead_filter_ratio
            dead_relu_list[dead_relu_list<neural_dead_times]=0
            dead_relu_list[dead_relu_list>=neural_dead_times]=1
            dead_relu_list=np.sum(dead_relu_list,axis=(1,2))            #count the number of dead neural for one filter
            dead_filter_index=np.where(dead_relu_list>neural_num*filter_dead_ratio)[0].tolist()
            #ensure the lower bound of filter number
            if filter_num[i]-len(dead_filter_index)<filter_num_lower_bound[i]:
                dead_filter_index=dead_filter_index[:filter_num[i]-filter_num_lower_bound[i]]
            filter_num[i]=filter_num[i]-len(dead_filter_index)

            print('layer {}: remain {} filters, prune {} filters.'.format(i, filter_num[i],
                                                                          len(dead_filter_index)))
total_filters_after_pruned=np.sum(filter_num)


print('before:{},after:{},prune{}'.format(total_filters,total_filters_after_pruned,1-total_filters_after_pruned/total_filters))