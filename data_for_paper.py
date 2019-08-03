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
import copy
import resnet_copied


def dead_neural_rate():
    # checkpoint=torch.load('/home/victorfang/Desktop/vgg16_bn_imagenet_deadReLU.tar')
    # neural_list=checkpoint['neural_list']
    # relu_list=checkpoint['relu_list']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint=torch.load('./baseline/resnet56_cifar10,accuracy=0.94230.tar')
    net = resnet_copied.resnet56().to(device)
    net.load_state_dict(checkpoint['state_dict'])

    # net=create_net.vgg_cifar10()
    val_loader=data_loader.create_validation_loader(batch_size=1000,num_workers=6,dataset_name='cifar10')
    # train_loader=data_loader.create_train_loader(batch_size=1600,num_workers=6,dataset_name='cifar10')
    #
    relu_list,neural_list=evaluate.check_ReLU_alive(net=net,neural_dead_times=50000,data_loader=val_loader)
    # ndt_list=[i for i in range(35000,51000,1000)]
    ndt_list=[i for i in range(6000,11000,1000)]
    dead_rate=list()
    for ndt in ndt_list:
        print(ndt)
        dead_rate.append(evaluate.cal_dead_neural_rate(neural_dead_times=ndt,neural_list_temp=neural_list))

    plt.figure()
    plt.title('df')
    plt.plot(ndt_list,dead_rate)
    plt.xlabel('neural dead times')
    plt.ylabel('neuron dead rate%')
    plt.legend()
    plt.show()

def plot_dead_filter_num_with_different_dft():
    # print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # net=create_net.vgg_cifar10()
    checkpoint = torch.load('./baseline/resnet56_cifar10,accuracy=0.94230.tar')
    net = resnet_copied.resnet56().to(device)
    net.load_state_dict(checkpoint['state_dict'])

    val_loader=data_loader.create_validation_loader(batch_size=500,num_workers=6,dataset_name='cifar10')
    relu_list,neural_list=evaluate.check_ReLU_alive(net=net,neural_dead_times=8000,data_loader=val_loader)


    # net=vgg.vgg16_bn(pretrained=False)
    # checkpoint=torch.load('/home/victorfang/Desktop/vgg16_bn_imagenet_deadReLU.tar')
    # neural_list=checkpoint['neural_list']
    # relu_list=checkpoint['relu_list']

    neural_dead_times=8000
    fdt_list=[0.001*i for i in range(1,1001)]
    dead_filter_num=list()
    for fdt in fdt_list:
        dead_filter_num.append(dead_filter_statistics(net=net,neural_list=neural_list,neural_dead_times=neural_dead_times,filter_dead_ratio=fdt,relu_list=relu_list))
    plt.figure()
    plt.title('df')
    plt.plot(fdt_list,dead_filter_num)
    plt.xlabel('filter activation ratio')
    plt.ylabel('number of filters')
    plt.legend()
    plt.show()

def dead_filter_statistics(net,relu_list,neural_list,neural_dead_times,filter_dead_ratio):



    dead_filter_num=list()                                                                      #num of dead filters in each layer
    filter_num=list()                                                                           #num of filters in each layer
    num_conv = 0  # num of conv layers in the net
    for mod in net.modules():
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv += 1

    for i in range(num_conv):
        for relu_key in list(neural_list.keys()):
            if relu_list[i] is relu_key:  # find the neural_list_statistics in layer i+1
                dead_relu_list = copy.deepcopy(neural_list[relu_key])
                neural_num = dead_relu_list.shape[1] * dead_relu_list.shape[2]  # neural num for one filter

                # judge dead filter by neural_dead_times and dead_filter_ratio
                dead_relu_list[dead_relu_list < neural_dead_times] = 0
                dead_relu_list[dead_relu_list >= neural_dead_times] = 1
                dead_relu_list = np.sum(dead_relu_list, axis=(1, 2))  # count the number of dead neural for one filter
                dead_filter_index = np.where(dead_relu_list >= neural_num * filter_dead_ratio)[0].tolist()
                dead_filter_num.append(len(dead_filter_index))
                filter_num.append(len(neural_list[relu_key]))

    dead_filter_num_sum=np.sum(dead_filter_num)
    return dead_filter_num_sum
    # plt.figure()
    # plt.title('statistics of dead filter\nneural_dead_time={},filter_dead_ratio={}'.format(neural_dead_times,filter_dead_ratio))
    # plt.bar(range(len(filter_num)),filter_num,label='filter')
    # plt.bar(range(len(dead_filter_num)),dead_filter_num,label='dead filter')
    # plt.xlabel('layer')
    # plt.ylabel('number of filters')
    # plt.legend()
    # plt.show()
    # print()
if __name__ == "__main__":
    print()
    # dead_neural_rate()
    # plot_dead_filter_num_with_different_dft()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=resnet.resnet34(pretrained=True).to(device)
    val_loader=data_loader.create_validation_loader(batch_size=64,num_workers=8,dataset_name='imagenet')

    # checkpoint = torch.load('./baseline/resnet56_cifar10,accuracy=0.94230.tar')
    # net = resnet_copied.resnet56().to(device)
    # net.load_state_dict(checkpoint['state_dict'])
    # val_loader = data_loader.create_validation_loader(batch_size=1000, num_workers=6, dataset_name='cifar10')


    relu_list, neural_list = evaluate.check_ReLU_alive(net=net, neural_dead_times=50000, data_loader=val_loader)
    c={'relu_list':relu_list,'neural_list':neural_list,'net':net}
    torch.save(c,'/home/victorfang/Desktop/resnet34_imagenet_DeadReLU.tar')