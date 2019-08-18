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
import time
def dead_neural_rate():
    # checkpoint=torch.load('/home/victorfang/Desktop/vgg16_bn_imagenet_deadReLU.tar')
    # checkpoint=torch.load('/home/victorfang/Desktop/resnet34_imagenet_DeadReLU.tar')
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
    relu_list,neural_list=evaluate.check_ReLU_alive(net=net,neural_dead_times=10000,data_loader=val_loader)
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


def plot_dead_neuron_filter_number(neural_dead_times=8000,dataset_name='cifar10',):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load('./baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
    vgg16=checkpoint['net']
    vgg16.load_state_dict(checkpoint['state_dict'])

    checkpoint=torch.load('./baseline/resnet56_cifar10,accuracy=0.94230.tar')
    resnet56 = resnet_copied.resnet56().to(device)
    resnet56.load_state_dict(checkpoint['state_dict'])

    vgg16_imagenet=vgg.vgg16_bn(pretrained=True).to(device)
    checkpoint=torch.load('/home/victorfang/Desktop/vgg16_bn_imagenet_deadReLU.tar')
    relu_list_imagenet=checkpoint['relu_list']
    neural_list_imagenet=checkpoint['neural_list']


    loader=data_loader.create_validation_loader(batch_size=1000,num_workers=6,dataset_name=dataset_name)
    # loader=data_loader.create_validation_loader(batch_size=1000,num_workers=8,dataset_name='cifar10_trainset')

    relu_list_vgg,neural_list_vgg=evaluate.check_ReLU_alive(net=vgg16,neural_dead_times=neural_dead_times,data_loader=loader, max_data_to_test=10000)
    relu_list_resnet,neural_list_resnet=evaluate.check_ReLU_alive(net=resnet56,neural_dead_times=neural_dead_times,data_loader=loader, max_data_to_test=10000)


    def get_statistics(net,relu_list,neural_list,neural_dead_times,sample_num=10000):
        num_conv = 0  # num of conv layers in the net
        for mod in net.modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                num_conv += 1

        neural_dead_list=list()                                         #神经元死亡次数的列表
        filter_dead_list=list()                                         #卷积核死亡比率的列表
        dead_ratio=list()
        for i in range(num_conv):
            for relu_key in list(neural_list.keys()):
                if relu_list[i] is relu_key:  # find the neural_list_statistics in layer i+1
                    dead_times = copy.deepcopy(neural_list[relu_key])
                    neural_dead_list+=copy.deepcopy(dead_times).flatten().tolist()


                    neural_num = dead_times.shape[1] * dead_times.shape[2]  # neural num for one filter

                    # compute sum(dead_times)/(batch_size*neural_num) as label for each filter
                    dead_times = np.sum(dead_times, axis=(1, 2))
                    dead_ratio += (dead_times / (neural_num * sample_num)).tolist()

                    # # judge dead filter by neural_dead_times and dead_filter_ratio
                    # dead_times[dead_times < neural_dead_times] = 0
                    # dead_times[dead_times >= neural_dead_times] = 1
                    # dead_times = np.sum(dead_times, axis=(1, 2))  # count the number of dead neural for one filter
                    # dead_times = dead_times / neural_num
                    # filter_dead_list+=dead_times.tolist()
                    break
        active_ratio=1-np.array(dead_ratio)
        active_filter_list =1- np.array(filter_dead_list)
        neural_activated_list=(sample_num-np.array(neural_dead_list))/sample_num


        return neural_activated_list,active_ratio,#active_filter_list

    nal_vgg,afl_vgg=get_statistics(vgg16,relu_list_vgg,neural_list_vgg,neural_dead_times=neural_dead_times)
    nal_resnet, afl_resnet = get_statistics(resnet56,relu_list_resnet, neural_list_resnet,neural_dead_times=neural_dead_times)
    nal_imagenet,afl_imagenet=get_statistics(vgg16_imagenet,relu_list_imagenet,neural_list_imagenet,sample_num=50000,neural_dead_times=40000)

    #cdf_of_dead_neurons
    plt.figure()
    plt.hist([nal_vgg,nal_resnet,nal_imagenet],cumulative=True,histtype='step',bins=1000,density=True,)#linewidth=5.0) #cumulative=False为pdf，true为cdf
    # plt.hist(neural_activated_list,cumulative=True,histtype='bar',bins=20,density=True,rwidth=0.6) #cumulative=False为pdf，true为cdf
    plt.xlabel('Activation Ratio')
    plt.ylabel('Ratio of Neurons')
    plt.legend(['VGG-16 on CIFAR-10','ResNet-56 on CIFAR-10','VGG-16 on ImageNet'],loc='upper left')
    plt.savefig('0cdf_of_dead_neurons.jpg')
    plt.savefig('cdf_of_dead_neurons.eps',format='eps')
    plt.show()

    #cdf_of_inactive_filter
    plt.figure()
    plt.hist([afl_vgg,afl_resnet,afl_imagenet],cumulative=True,histtype='step',bins=1000,density=True,)#linewidth=5.0) #cumulative=False为pdf，true为cdf
    # plt.hist(neural_activated_list,cumulative=True,histtype='bar',bins=20,density=True,rwidth=0.6) #cumulative=False为pdf，true为cdf
    plt.xlabel('Activation Ratio')
    plt.ylabel('Ratio of Filters')
    plt.legend(['VGG-16 on CIFAR-10','ResNet-56 on CIFAR-10','VGG-16 on ImageNet'],loc='upper left')
    plt.savefig('0cdf_of_inactive_filter.jpg')
    plt.savefig('cdf_of_inactive_filter.eps',format='eps')
    plt.show()

    #pdf_of_dead_neurons
    plt.figure()
    hist_list = list()
    for nal in [nal_vgg,nal_resnet,nal_imagenet]:
        hist, bins = np.histogram(nal, bins=[0.1 * i for i in range(11)])
        hist_list.append(hist / np.sum(hist))
    x_tick = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    plt.figure()
    plt.bar(x_tick - 0.02, hist_list[0], color='blue', hatch='//', label='VGG-16 on CIFAR-10', align='center',
            width=0.02)
    plt.bar(x_tick, hist_list[1], color='grey', hatch='\\', label='ResNet-56 on CIFAR-10', align='center', width=0.02)
    plt.bar(x_tick + 0.02, hist_list[2], color='red', hatch='/', label='VGG-16 on ImageNet', align='center', width=0.02)
    plt.xticks(x_tick, x_tick)
    plt.xlabel('Activation Ratio')
    plt.ylabel('Ratio of Neurons')
    plt.legend(loc='upper right')
    plt.savefig('0pdf_of_dead_neurons.jpg')
    plt.savefig('pdf_of_dead_neurons.eps',format='eps')
    plt.show()


    #pdf_of_inactive_filter
    plt.figure()
    hist_list=list()
    for active_ratio in [afl_vgg,afl_resnet,afl_imagenet]:
        hist, bins = np.histogram(active_ratio, bins=[0.1 * i for i in range(11)])
        hist_list.append( hist / np.sum(hist))
    x_tick=np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95])
    plt.figure()
    plt.bar(x_tick-0.02, hist_list[0], color='blue', hatch='//', label='VGG-16 on CIFAR-10',align='center',width=0.02)
    plt.bar(x_tick, hist_list[1], color='grey', hatch='\\',  label='ResNet-56 on CIFAR-10',align='center',width=0.02)
    plt.bar(x_tick+0.02, hist_list[2], color='red', hatch='/',  label='VGG-16 on ImageNet',align='center',width=0.02)
    plt.xticks(x_tick, x_tick)
    plt.xlabel('Activation Ratio')
    plt.ylabel('Ratio of Filters')
    plt.legend(loc='upper right')
    plt.savefig('0pdf_of_inactive_filter.jpg')
    plt.savefig('pdf_of_inactive_filter.eps',format='eps')
    plt.show()

    print()


def plot_dead_filter_num_with_different_fdt():
    # print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # net=create_net.vgg_cifar10()
    checkpoint = torch.load('./baseline/resnet56_cifar10,accuracy=0.94230.tar')
    net = resnet_copied.resnet56().to(device)
    net.load_state_dict(checkpoint['state_dict'])

    val_loader=data_loader.create_validation_loader(batch_size=500,num_workers=6,dataset_name='cifar10')
    relu_list,neural_list=evaluate.check_ReLU_alive(net=net,neural_dead_times=8000,data_loader=val_loader)


    # net=vgg.vgg16_bn(pretrained=False)
    # checkpoint=torch.load('/home/victorfang/Desktop/vgg16_bn_imagenet_deadReLU.tar')
    # net=resnet.resnet34(pretrained=True)
    # checkpoint=torch.load('/home/victorfang/Desktop/resnet34_imagenet_DeadReLU.tar')
    # neural_list=checkpoint['neural_list']
    # relu_list=checkpoint['relu_list']

    neural_dead_times=8000
    # neural_dead_times=40000
    fdt_list=[0.001*i for i in range(1,1001)]
    dead_filter_num=list()
    for fdt in fdt_list:
        dead_filter_num.append(dead_filter_statistics(net=net,neural_list=neural_list,neural_dead_times=neural_dead_times,filter_dead_ratio=fdt,relu_list=relu_list))
        if fdt==0.8:
            print()
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

def speed_up():
    device=torch.device('cpu')#device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint=torch.load('./baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
    net_original=checkpoint['net']
    net_original.load_state_dict(checkpoint['state_dict'])

    checkpoint=torch.load('./model_saved/vgg16bn_cifar10_realdata_regressor6_大幅度/checkpoint/flop=39915982,accuracy=0.93200.tar')
    net_pruned=checkpoint['net']
    net_pruned.load_state_dict(checkpoint['state_dict'])




    # batch_size=[256,512,1024]
    num_workers=[i for i in range(4,5)]
    batch_size=[300,600,1000,1600]

    device_list=[torch.device('cuda'),torch.device('cpu')]
    for num_worker in num_workers:
        time_original = list()
        time_pruned = list()
        for d in device_list:
            for bs in batch_size:
                net_original.to(d)
                net_pruned.to(d)

                dl=data_loader.create_validation_loader(batch_size=bs,num_workers=num_worker,dataset_name='cifar10')
                start_time=time.time()
                evaluate.evaluate_net(net=net_original,data_loader=dl,save_net=False,device=d)
                end_time=time.time()
                time_original.append(end_time-start_time)
                del dl

                dl=data_loader.create_validation_loader(batch_size=bs,num_workers=num_worker,dataset_name='cifar10')
                start_time=time.time()
                evaluate.evaluate_net(net=net_pruned,data_loader=dl,save_net=False,device=d)
                end_time=time.time()
                time_pruned.append(end_time-start_time)
                del dl

        print('time before pruned:',time_original)
        print('time after pruned:',time_pruned)
        acceleration=np.array(time_original)/np.array(time_pruned)
        baseline=np.ones(shape=2*len(batch_size))
        x_tick=range(len(baseline))

        plt.figure()
        plt.bar(x_tick[:len(batch_size)],acceleration[:len(batch_size)],color='blue',hatch='//',label='GPU')
        plt.bar(x_tick[len(batch_size):], acceleration[len(batch_size):], color='grey', hatch='\\', label='CPU')
        plt.bar(x_tick,baseline,color='red',hatch='*',label='Baseline')
        plt.xticks(x_tick,batch_size+batch_size)
        for x,y in enumerate(list(acceleration)):
            plt.text(x,y+0.1,'%.2f x'%y,ha='center')
        plt.ylim([0,np.max(acceleration)+0.3])
        plt.xlabel('Batch-Size')
        plt.ylabel('Speed-Up')
        plt.legend(loc='upper left')
        plt.savefig(str(num_worker)+'speed_up.eps',format='eps')
        plt.savefig(str(num_worker)+'speed_up.jpg')
        plt.show()
        print()



if __name__ == "__main__":
    plot_dead_neuron_filter_number()
    # speed_up()
