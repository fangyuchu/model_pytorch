import os,sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
from framework import data_loader, evaluate
from network import storage,resnet_cifar,vgg,net_with_predicted_mask,resnet,mobilenet
from framework.measure_flops import measure_model,count_params
from filter_characteristic import predict_dead_filter
import matplotlib.pyplot as plt
import copy
import time
import os
from network.modules import conv2d_with_mask
from framework import config as conf
from framework.draw import draw_masked_net,draw_gat_attention
from framework.train import show_feature_map
import matplotlib.patches as mpathes


def dead_neural_rate():
    # checkpoint=torch.load('/home/victorfang/Desktop/vgg16_bn_imagenet_deadReLU.tar')
    # checkpoint=torch.load('/home/victorfang/Desktop/resnet34_imagenet_DeadReLU.tar')
    # neural_list=checkpoint['neural_list']
    # relu_list=checkpoint['relu_list']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint=torch.load('../data/baseline/resnet56_cifar10,accuracy=0.94230.tar')
    net = resnet_cifar.resnet56().to(device)
    net.load_state_dict(checkpoint['state_dict'])

    # net=create_net.vgg_cifar10()
    val_loader= data_loader.create_test_loader(batch_size=1000, num_workers=6, dataset_name='cifar10')
    # train_loader=data_loader.create_train_loader(batch_size=1600,num_workers=6,dataset_name='cifar10')
    #
    relu_list,neural_list= evaluate.check_ReLU_alive(net=net, neural_dead_times=10000, data_loader=val_loader)
    # ndt_list=[i for i in range(35000,51000,1000)]
    ndt_list=[i for i in range(6000,11000,1000)]
    dead_rate=[]
    for ndt in ndt_list:
        print(ndt)
        dead_rate.append(evaluate.cal_dead_neural_rate(neural_dead_times=ndt, neural_list_temp=neural_list))

    plt.figure()
    plt.title('df')
    plt.plot(ndt_list,dead_rate)
    plt.xlabel('neural dead times')
    plt.ylabel('neuron dead rate%')
    plt.legend()
    plt.show()


def plot_dead_neuron_filter_number(neural_dead_times=8000,dataset_name='cifar10',):
    fontsize = 17
    label_fontsize=24
    tick_fontsize=20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load('../data/baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
    vgg16=storage.restore_net(checkpoint)
    vgg16.load_state_dict(checkpoint['state_dict'])

    checkpoint=torch.load('../data/baseline/resnet56_cifar10,accuracy=0.94230.tar')
    resnet56 = resnet_cifar.resnet56().to(device)
    resnet56.load_state_dict(checkpoint['state_dict'])

    vgg16_imagenet= vgg.vgg16_bn(pretrained=True).to(device)
    checkpoint=torch.load('/home/victorfang/Desktop/vgg16_bn_imagenet_deadReLU.tar')
    relu_list_imagenet=checkpoint['relu_list']
    neural_list_imagenet=checkpoint['neural_list']


    loader= data_loader.create_test_loader(batch_size=100, num_workers=1, dataset_name=dataset_name)
    # loader=data_loader.create_test_loader(batch_size=1000,num_workers=8,dataset_name='cifar10_trainset')

    relu_list_vgg,neural_list_vgg= evaluate.check_ReLU_alive(net=vgg16, neural_dead_times=neural_dead_times, data_loader=loader, max_data_to_test=10000)
    relu_list_resnet,neural_list_resnet= evaluate.check_ReLU_alive(net=resnet56, neural_dead_times=neural_dead_times, data_loader=loader, max_data_to_test=10000)


    def get_statistics(net,relu_list,neural_list,neural_dead_times,sample_num=10000):
        num_conv = 0  # num of conv layers in the net
        for mod in net.modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                num_conv += 1

        neural_dead_list=[]                                         #神经元死亡次数的列表
        filter_dead_list=[]                                         #卷积核死亡比率的列表
        FIRE=[]
        for i in range(num_conv):
            for relu_key in list(neural_list.keys()):
                if relu_list[i] is relu_key:  # find the neural_list_statistics in layer i+1
                    dead_times = copy.deepcopy(neural_list[relu_key])
                    neural_dead_list+=copy.deepcopy(dead_times).flatten().tolist()


                    neural_num = dead_times.shape[1] * dead_times.shape[2]  # neural num for one filter

                    # compute sum(dead_times)/(batch_size*neural_num) as label for each filter
                    dead_times = np.sum(dead_times, axis=(1, 2))
                    FIRE += (dead_times / (neural_num * sample_num)).tolist()

                    # # judge dead filter by neural_dead_times and dead_filter_ratio
                    # dead_times[dead_times < neural_dead_times] = 0
                    # dead_times[dead_times >= neural_dead_times] = 1
                    # dead_times = np.sum(dead_times, axis=(1, 2))  # count the number of dead neural for one filter
                    # dead_times = dead_times / neural_num
                    # filter_dead_list+=dead_times.tolist()
                    break
        active_ratio=1-np.array(FIRE)
        active_filter_list =1- np.array(filter_dead_list)
        neural_activated_list=(sample_num-np.array(neural_dead_list))/sample_num


        return neural_activated_list,active_ratio,#active_filter_list

    nal_vgg,afl_vgg=get_statistics(vgg16,relu_list_vgg,neural_list_vgg,neural_dead_times=neural_dead_times)
    nal_resnet, afl_resnet = get_statistics(resnet56,relu_list_resnet, neural_list_resnet,neural_dead_times=neural_dead_times)
    nal_imagenet,afl_imagenet=get_statistics(vgg16_imagenet,relu_list_imagenet,neural_list_imagenet,sample_num=50000,neural_dead_times=40000)

    # #cdf_of_dead_neurons
    # plt.figure()
    # plt.hist([nal_vgg,nal_resnet,nal_imagenet],cumulative=True,histtype='step',bins=1000,density=True,)#linewidth=5.0) #cumulative=False为pdf，true为cdf
    # # plt.hist(neural_activated_list,cumulative=True,histtype='bar',bins=20,density=True,rwidth=0.6) #cumulative=False为pdf，true为cdf
    # plt.xlabel('Activation Ratio',fontsize = fontsize)
    # plt.ylabel('Ratio of Neurons',fontsize = fontsize)
    # plt.legend(['VGG-16 on CIFAR-10','ResNet-56 on CIFAR-10','VGG-16 on ImageNet'],loc='upper left',fontsize = fontsize)
    # # plt.savefig('0cdf_of_dead_neurons.jpg')
    # plt.savefig('cdf_of_dead_neurons.eps',format='eps')
    # plt.show()
    #
    # #cdf_of_inactive_filter
    # plt.figure()
    # plt.hist([afl_vgg,afl_resnet,afl_imagenet],cumulative=True,histtype='step',bins=1000,density=True,)#linewidth=5.0) #cumulative=False为pdf，true为cdf
    # # plt.hist(neural_activated_list,cumulative=True,histtype='bar',bins=20,density=True,rwidth=0.6) #cumulative=False为pdf，true为cdf
    # plt.xlabel('Activation Ratio',fontsize = fontsize)
    # plt.ylabel('Ratio of Filters',fontsize = fontsize)
    # plt.legend(['VGG-16 on CIFAR-10','ResNet-56 on CIFAR-10','VGG-16 on ImageNet'],loc='upper left',fontsize = fontsize)
    # # plt.savefig('0cdf_of_inactive_filter.jpg')
    # plt.savefig('cdf_of_inactive_filter.eps',format='eps')
    # plt.show()

    #pdf_of_dead_neurons
    plt.figure()
    hist_list = []
    for nal in [nal_vgg,nal_resnet,nal_imagenet]:
        hist, bins = np.histogram(nal, bins=[0.1 * i for i in range(11)])
        hist_list.append(100*hist / np.sum(hist))
    x_tick = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
    plt.figure()
    plt.bar(x_tick - 2, hist_list[0], color='coral', edgecolor='black', label='VGG-16 on CIFAR-10', align='center',
            width=2)
    plt.bar(x_tick, hist_list[1], color='cyan', edgecolor='black', label='ResNet-56 on CIFAR-10', align='center', width=2)
    plt.bar(x_tick + 2, hist_list[2], color='mediumslateblue', edgecolor='black', label='VGG-16 on ImageNet', align='center', width=2)
    plt.xticks(x_tick, x_tick,size=tick_fontsize)
    plt.yticks(size=tick_fontsize)
    plt.xlabel('Activation Ratio (%)',fontsize = label_fontsize)
    plt.ylabel('% of Neurons',fontsize = label_fontsize)
    plt.legend(loc='upper right',fontsize = fontsize)
    # plt.savefig('0pdf_of_dead_neurons.jpg')
    plt.savefig('pdf_of_dead_neurons.eps',format='eps',bbox_inches='tight')
    plt.show()


    #pdf_of_inactive_filter
    plt.figure()
    hist_list=[]
    for active_ratio in [afl_vgg,afl_resnet,afl_imagenet]:
        hist, bins = np.histogram(active_ratio, bins=[0.1 * i for i in range(11)])
        hist_list.append( 100*hist / np.sum(hist))
    x_tick = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
    plt.figure()
    plt.bar(x_tick-2, hist_list[0], color='coral', edgecolor='black', label='VGG-16 on CIFAR-10',align='center',width=2)
    plt.bar(x_tick, hist_list[1], color='cyan', edgecolor='black',  label='ResNet-56 on CIFAR-10',align='center',width=2)
    plt.bar(x_tick+2, hist_list[2], color='mediumslateblue', edgecolor='black',  label='VGG-16 on ImageNet',align='center',width=2)
    plt.xticks(x_tick, x_tick,size=tick_fontsize)
    plt.yticks(size=tick_fontsize)
    plt.xlabel('Activation Ratio (%)',fontsize = label_fontsize)
    plt.ylabel('% of Filters',fontsize = label_fontsize)
    plt.legend(loc='upper right',fontsize = fontsize)
    # plt.savefig('0pdf_of_inactive_filter.jpg')
    plt.savefig('pdf_of_inactive_filter.eps',format='eps',bbox_inches='tight')
    plt.show()

    print()


def plot_dead_filter_num_with_different_fdt():
    # print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # net=create_net.vgg_cifar10()
    checkpoint = torch.load('../data/baseline/resnet56_cifar10,accuracy=0.94230.tar')
    net = resnet_cifar.resnet56().to(device)
    net.load_state_dict(checkpoint['state_dict'])

    val_loader= data_loader.create_test_loader(batch_size=500, num_workers=6, dataset_name='cifar10')
    relu_list,neural_list= evaluate.check_ReLU_alive(net=net, neural_dead_times=8000, data_loader=val_loader)


    # net=vgg.vgg16_bn(pretrained=False)
    # checkpoint=torch.load('/home/victorfang/Desktop/vgg16_bn_imagenet_deadReLU.tar')
    # net=resnet.resnet34(pretrained=True)
    # checkpoint=torch.load('/home/victorfang/Desktop/resnet34_imagenet_DeadReLU.tar')
    # neural_list=checkpoint['neural_list']
    # relu_list=checkpoint['relu_list']

    neural_dead_times=8000
    # neural_dead_times=40000
    fdt_list=[0.001*i for i in range(1,1001)]
    dead_filter_num=[]
    for fdt in fdt_list:
        dead_filter_num.append(dead_filter_statistics(net=net,neural_list=neural_list,neural_dead_times=neural_dead_times,filter_FIRE=fdt,relu_list=relu_list))
        if fdt==0.8:
            print()
    plt.figure()
    plt.title('df')
    plt.plot(fdt_list,dead_filter_num)
    plt.xlabel('filter activation ratio')
    plt.ylabel('number of filters')
    plt.legend()
    plt.show()

def dead_filter_statistics(net,relu_list,neural_list,neural_dead_times,filter_FIRE):



    dead_filter_num=[]                                                                      #num of dead filters in each layer
    filter_num=[]                                                                           #num of filters in each layer
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
                dead_filter_index = np.where(dead_relu_list >= neural_num * filter_FIRE)[0].tolist()
                dead_filter_num.append(len(dead_filter_index))
                filter_num.append(len(neural_list[relu_key]))

    dead_filter_num_sum=np.sum(dead_filter_num)
    return dead_filter_num_sum
    # plt.figure()
    # plt.title('statistics of dead filter\nneural_dead_time={},filter_FIRE={}'.format(neural_dead_times,filter_FIRE))
    # plt.bar(range(len(filter_num)),filter_num,label='filter')
    # plt.bar(range(len(dead_filter_num)),dead_filter_num,label='dead filter')
    # plt.xlabel('layer')
    # plt.ylabel('number of filters')
    # plt.legend()
    # plt.show()
    # print()

def speed_up_pruned_net():
    fontsize=15

    checkpoint=torch.load('../data/baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
    net_original=storage.restore_net(checkpoint)

    checkpoint=torch.load('../data/baseline/resnet56_cifar10,accuracy=0.93280.tar')
    net_original= resnet_cifar.resnet56()
    net_original.load_state_dict(checkpoint['state_dict'])

    checkpoint=torch.load('../data/model_saved/vgg16bn_cifar10_realdata_regressor6_大幅度/checkpoint/flop=39915982,accuracy=0.93200.tar')

    checkpoint=torch.load('../data/model_saved/resnet56_cifar10_regressor_prunedBaseline2/checkpoint/flop=36145802,accuracy=0.92110.tar')
    net_pruned=storage.restore_net(checkpoint)
    net_pruned.load_state_dict(checkpoint['state_dict'])




    # batch_size=[256,512,1024]
    num_workers=[i for i in range(4,5)]
    batch_size=[300,600,1000,1600]

    device_list=[torch.device('cuda')]#
    # device_list=[torch.device('cpu')]
    for num_worker in num_workers:
        time_original = []
        time_pruned = []
        for d in device_list:
            for bs in batch_size:
                net_original.to(d)
                net_pruned.to(d)

                dl= data_loader.create_test_loader(batch_size=bs, num_workers=num_worker, dataset_name='cifar10')
                start_time=time.time()
                evaluate.evaluate_net(net=net_original, data_loader=dl, save_net=False, device=d)
                end_time=time.time()
                time_original.append(end_time-start_time)
                del dl

                dl= data_loader.create_test_loader(batch_size=bs, num_workers=num_worker, dataset_name='cifar10')
                start_time=time.time()
                evaluate.evaluate_net(net=net_pruned, data_loader=dl, save_net=False, device=d)
                end_time=time.time()
                time_pruned.append(end_time-start_time)
                del dl

        print('time before pruned:',time_original)
        print('time after pruned:',time_pruned)
        acceleration=np.array(time_original)/np.array(time_pruned)
        baseline=np.ones(shape=len(batch_size))
        x_tick=range(len(baseline))

        plt.figure()
        plt.bar(x_tick,acceleration,color='blue',hatch='//')#,label='GPU')
        # plt.bar(x_tick[len(batch_size):], acceleration[len(batch_size):], color='grey', hatch='\\', label='CPU')
        # plt.bar(x_tick,baseline,color='red',hatch='*',label='Baseline')
        plt.xticks(x_tick,batch_size,fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        for x,y in enumerate(list(acceleration)):
            plt.text(x,y+0.1,'%.2f x'%y,ha='center',fontsize=fontsize)
        plt.ylim([0,np.max(acceleration)+0.5])
        plt.xlabel('Batch-Size',fontsize=fontsize)
        plt.ylabel('Speed-Up',fontsize=fontsize)
        # plt.legend(loc='upper left')
        plt.savefig('resnet_gpu_speed_up.eps',format='eps')
        # plt.savefig(str(num_worker)+'speed_up.jpg')
        plt.show()
        print()


def speed_up_regressor():
    path='/home/victorfang/PycharmProjects/model_pytorch/model_saved/vgg16bn_tinyimagenet_prune/checkpoint'
    predictor = predict_dead_filter.predictor(name='random_forest')
    predictor.load(path='/home/disk_new/model_saved/vgg16bn_tinyimagenet_prune/')

    file_list=os.listdir(path)
    file_list.sort()
    regressor_time=[]
    real_data_time=[]
    for file in file_list:
        print(file)
        checkpoint_path=os.path.join(path,file)
        checkpoint=torch.load(checkpoint_path)
        net=storage.restore_net(checkpoint)
        net.load_state_dict(checkpoint['state_dict'])
        #time for regressor
        start_time = time.time()
        evaluate.find_useless_filters_regressor_version(net=net,
                                                        predictor=predictor,
                                                        percent_of_inactive_filter=0.1,
                                                        max_filters_pruned_for_one_time=0.2, )
        end_time=time.time()
        regressor_time.append(end_time-start_time)
        #time for sampled data
        start_time = time.time()
        evaluate.find_useless_filters_data_version(net=net,
                                                   batch_size=24,
                                                   dataset_name='tiny_imagenet',
                                                   percent_of_inactive_filter=0.1,
                                                   max_data_to_test=50000,
                                                   )
        end_time=time.time()
        real_data_time.append(end_time - start_time)

    print(regressor_time)
    print(real_data_time)

def tolerance():
    fontsize=14

    flop=[125491556,90655076,83650916,64924004,54178148,42989924,41515364,39082340]
    pruned_rate=100*(125491556-np.array(flop))/125491556
    acc_drop=[0,0.0058,0.0095,0.0128,0.0203,0.0263,0.0318,0.0359]
    acc_drop=100*np.array(acc_drop)


    plt.figure()
    plt.plot(acc_drop,pruned_rate,'bo--')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)

    plt.xlabel('Tolerance Accuracy Drop%', fontsize=fontsize)
    plt.ylabel('Pruned Rate%', fontsize=fontsize)
    plt.savefig('resnet56_cifar100_tolerance.eps', format='eps')
    plt.show()

def plot_layer_structure(baseline_net,net1,net_name):
    structure=[[],[],[]]
    for name,mod in baseline_net.named_modules():
        if isinstance(mod,nn.Conv2d) and 'downsample' not in name and mod.groups==1:
            structure[0]+=[mod.out_channels]
    for name,mod in net1.named_modules():
        if isinstance(mod,nn.Conv2d) and 'downsample' not in name and mod.groups==1:
            structure[1]+=[mod.out_channels]
    fig, ax = plt.subplots(figsize=(24, 6))
    # for name,mod in net2.named_modules:
    #     if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
    #         structure[2]+=[mod.out_channels]
    if net_name=='resnet50':
        net1_name='DualPrune,Acc=70.74%,FLOPs=625M,Param=4.95M'
        # net1_name='DualPrune,FLOPs=625M,Param=4.95M'
        # net2_name='PFS,FLOPs=645M,Param=5.44M'
        net2_name='PFS,Acc=63.64%,FLOPs=645M,Param=5.44M'
        structure[2]=[64, 6, 7, 256, 2, 5, 256, 5, 5, 256, 12, 10, 512, 13, 8, 512, 8, 11, 512, 10, 9, 512, 19, 20, 1024, 20, 19, 1024, 19, 20, 1024, 16, 19, 1024, 17, 18, 1024, 22, 17, 1024, 24, 22, 2048, 17, 29, 2048, 22, 28, 2048]
    elif net_name=='mobilenet_v1':
        net1_name='DualPrune,Acc=61.99%,FLOPs=89M,Param=0.84M'
        # net1_name='DualPrune,FLOPs=89M,Param=0.84M'
        net2_name='PFS,Acc=44.62%,FLOPs=95M,Param=0.56M'
        # net2_name='PFS,FLOPs=95M,Param=0.56M'
        structure[2]=[22, 38, 71, 68, 125, 118, 198, 159, 117, 65, 38, 11, 11, 427]
        ax.set_xticks([0,2,4,6,8,10,13])
        ax.set_yscale('log')
    font_size=40    #for small figure
    marker_size=15
    line_width=3
    marker_list = ['v', 'o', '*', 'd', '.', '+', 's']

    # structure[1]=np.array(structure[1])/np.array(structure[0])
    # structure[2]=np.array(structure[2])/np.array(structure[0])
    # structure[0]=np.array(structure[0])/np.array(structure[0])
    # rect = mpathes.Rectangle((0,0),height=1,width=len(structure[0]-2), color='g',alpha=0.3)
    # ax.add_patch(rect)

    ax.bar(list(range(len(structure[0]))),structure[0],label='Full Network',color='g',alpha=0.3)
    ax.plot(list(range(len(structure[1]))),structure[1] , marker=marker_list[0], label=net1_name, markersize=marker_size,linewidth=line_width)
    ax.plot(list(range(len(structure[2]))),structure[2] , marker=marker_list[1], label=net2_name, markersize=marker_size,linewidth=line_width)
    ax.set_xlim(right=len(structure[0]))
    ax.set_xlabel('Layer Index',fontsize=font_size)
    ax.set_ylabel('Filter Number',fontsize=font_size)
    ax.tick_params(labelsize=font_size)
    ax.legend(fontsize=font_size,loc='best',frameon=False)

    bwith = 2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    # ax.set_ylim(bottom=5)

    plt.show()


def acc_pruneratio(acc_list,prune_ratio,legends,exp_name):
    '''

    :param acc_list: one/two-dimension lists
    :param prune_ratio: one-dimension lists
    :param exp_name:
    :return:
    '''
    # plt.style.use('seaborn-whitegrid')
    # font_size=20    # for motivation
    font_size=20    #for small figure
    font_size=25 #for part module
    marker_size=15
    # marker_list=['v','+','*','.','d','s','o',] # for baselines
    marker_list=['v','o','*','d','.','+','s']
    if type(acc_list[0]) is not list:
        acc_list=[acc_list]
    if acc_list[0][0]<1:
        acc_list=np.array(acc_list)*100
    if prune_ratio[0]<1:
        prune_ratio=np.array(prune_ratio)*100
    x=prune_ratio
    fig,ax=plt.subplots(figsize=(8, 5))
    # plt.axes(yscale="log")
    for i,y in enumerate(acc_list):
        ax.plot(x,y,marker=marker_list[i],label=legends[i],markersize=marker_size)
    # plt.ylim(0.6,1)
    ax.set_xlabel('Pruned Flops%',fontsize=font_size)
    ax.set_ylabel('Accuracy%',fontsize=font_size)
    # plt.yticks([80,85,90],('80','85','90'))
    # ax.set_ylim(bottom=10)
    # ax.set_ylim(bottom=30)
    # ax.set_ylim(bottom=50)
    # ax.set_ylim(bottom=55)
    # ax.set_yticks([30,40,60,80,90,100],('0','40','60','80','90','100'),)
    # ax.set_yticks([50,60,80,90,100],('50','60','80','90','100'),)
    # ax.set_ylim(bottom=79)
    # ax.set_yticks([30,70,90],('30','70','90'))

    ax.tick_params(labelsize=font_size)
    ax.grid()

    bwith=2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    # plt.yticks([60,70,80,85,90],('60','70','80','85','90'))
    # plt.yticks([70,80,90],('70','80','90'))
    # plt.title(exp_name)
    ax.legend(fontsize=font_size,loc='best')
    plt.savefig('/home/victorfang/'+exp_name+'.png',dpi=fig.dpi)
    plt.show()

def layer_collapse():
    font_size = 25
    marker_size = 23
    ratio = [50,65,78.125,84.375,89.0625,93.75,96.875]
    acc = [93.3,93.2,93.0,92.68,92.3,91.01,83.83]
    flop_reduction =[9.04,11.57,14.12,15.25,16.10,16.94,17.19]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ratio, acc,  markersize=marker_size,marker = 'o')
    # ax.plot(ratio, acc,  marker = 'o')

    for i,flop in enumerate(flop_reduction):
        if i != len(acc)-1:
            ax.annotate(str(flop)+'%',xy=(ratio[i],acc[i]),xytext = (ratio[i]-2,acc[i]-1.5),fontsize = font_size-2)
        else:
            ax.annotate(str(flop)+'%',xy=(ratio[i],acc[i]),xytext = (ratio[i]-7,acc[i]+0.5),fontsize = font_size-2)

    ax.set_xlabel('Pruned Filter Ratio%',fontsize=font_size)
    ax.set_ylabel('Accuracy%',fontsize=font_size)
    # ax.set_xlabel('Pruned Filter Ratio%')
    # ax.set_ylabel('Accuracy%')
    ax.tick_params(labelsize=font_size)
    plt.show()

if __name__ == "__main__":
    layer_collapse()

    # # # motivation
    # acc_pruneratio(acc_list=[
    #     [0.927733, 0.9172, 0.907667, 0.903967, 0.904667, 0.896867, 0, 0, 0],
    #     [0.9283, 0.9245, 0.918933, 0.913867, 0.89, 0.8726, 0.8407, 0.6393, 0.4423],
    #     [0.9412, 0.9374, 0.9312, 0.9162, 0.9071, .836033, 0.8532, 0.720933, 0],
    #     [0.9209, 0.9044, 0.8879, 0.8743, 0.87, 0.855833, 0.8259, 0.701967, 0.4044],
    #     # [0.932, 0.9308, 0.9201, 0.915, 0.9139, 0.9064, 0, 0, 0],
    #     # [0.9317,0.9188,0.9146,0.9072,0.9034,0.8658,0,0,0]
    #
    #     # [0.9228, 0.9174, 0.9145, 0.912, 0.9107, 0.8903, 0.8871, 0.8613, 0.8096],
    # ],
    #
    #     prune_ratio=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98],
    #     # legends=['PFS', 'EB-Tickets', 'Rethink', 'SFP', 'DPFPS','HRank'],
    #     legends=['PFS', 'EB-Tickets', 'Rethink', 'SFP'],
    #     exp_name='motivation_resnet56_cifar10')
    #
    # acc_pruneratio(acc_list=[
    #     [0.711, 0.702933, 0.6963, 0.691567, 0.682667, 0.6694, 0.654066667,  0.5441,0.4133],
    #                          [0.6584,0.6181,0.5699,0.6109,0.5156,0.5535,0.4379,0.4643666,0.2771],
    #                          [0.62095,0.6134,0.528,0.54595,0.53205,.3951,0.34245,0.3141,0.147],
    #                          [0.7149,0.7133,0.7008,0.6968,0.6874,0.6668,0.61545,0.42025,0.1464],
    #     # [0.9228, 0.9174, 0.9145, 0.912, 0.9107, 0.8903, 0.8871, 0.8613, 0.8096],
    #
    #                          ],
    #
    #                prune_ratio=[0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.98],
    #                legends=['PFS','EB-Tickets','Rethink','SFP'],
    #                exp_name='motivation_vgg16_cifar100')



    # # # # Pruning ResNet-56 on CIFAR-10
    # acc_pruneratio(acc_list=[[0.9107,0.8984,0.8903,0.8916,0.8937,0.876,0.8613,0.8096],
    #                          [0.904667, 0.89613, 0.896867, 0.886733, 0, 0, 0, 0],
    #                          [0.89, 0.8842, 0.8726, 0.8643, 0.8407, 0.7488, 0.6393, 0.4423],
    #                          [0.871133,0.8438,0.836033,0.80906667,0.7579,0.785,0.720933,0],
    #                          [0.874367,0.8573,0.855833,0.84,0.8259,0.7791,0.701967,0.404383],
    #                          [0.9139,0.9134,0.9064,0.9027,0,0,0,0],
    #                         [0.9034,0.8795,0.8658,0,0,0,0,0]
    #                          ],
    #                prune_ratio=[0.8,0.83,0.85,0.87,0.9,0.93,0.95,0.98],#[0.7,0.75,0.8,0.85,0.9,0.95],
    #                legends=['DualPrune','PFS','EB-Tickets','Rethink','SFP','DPFPS','HRank'],
    #                exp_name='Pruning_ResNet-56_on_CIFAR-10')

    # # # Pruning VGG-16 on CIFAR-10
    # acc_pruneratio(acc_list=[[0.9264, 0.9245, 0.924, 0.9204, 0.9169, 0.9135, 0.9052, 0.8655],
    #                          [0.9144, 0.9147, 0.9101, 0.908167, 0.8978, 0.895967, 0.8816, 0],
    #                          [0.919933, 0.9102, 0.9115, 0.9044, 0.901167, 0.897, 0.881567, 0.566033],
    #                          [ 0.865566667, 0.8399, 0.851433333, 0.875833333, 0.850033333, 0.857266667,0.749133333, 0.543666667],
    #                          [0.9179,0.9149,0.91235,0.91305,0.90595,0.896,0.8723,0.59765],
    #                          [0.8876,0.871,0.8599,0.8426,0.7974,0.7245,0.6038,0]
    #                          ],
    #                prune_ratio=[0.8,0.83,0.85,0.87,0.9,0.93,0.95,0.98],
    #                legends=['DualPrune','PFS','EB-Tickets','Rethink','SFP','HRank'],
    #                exp_name='Pruning_VGG-16_on_CIFAR-10')
    #
    # #plot structure
    # # #resnet50
    # checkpoint = torch.load('/home/disk_new/model_saved_4gpu/model_saved/gat_resnet50_predicted_mask_and_variable_shortcut_net_newinner_newtrain_85_7/checkpoint/flop=625447007,accuracy=0.70741.pth')
    # net = checkpoint['net'].module.cuda()
    # plot_layer_structure(resnet.resnet50(),net,'resnet50')
    # # #mobilenet_v1
    # checkpoint = torch.load('/home/disk_new/model_saved_4gpu/model_saved/gat_mobilenet_v1_predicted_mask_and_variable_shortcut_net_newinner_newtrain_85_3/checkpoint/flop=89121118,accuracy=0.61999.pth')
    #
    # net = checkpoint['net'].module.cuda()
    # plot_layer_structure(mobilenet.MobileNet(n_class=1000), net, 'mobilenet_v1')


    # # # Pruning VGG-16 on CIFAR-100
    # acc_pruneratio(acc_list=[[0.704,0.7002,0.6948,0.6835,0.6802,0.6544,0.6307,0.5429],
    #                          [0.682667,0.679367,0.6694,0.663167,0.654067,0.636767,0.5441,0.4133],
    #                          [0.5156,0.54867,0.55353,0.5317,0.43793,0.50293,0.464367,0.2717],
    #                          [0.53205,0.41285,0.3951,0.4863,0.34245,0.4291,0.3141,0.147],
    #                          [0.6874,0.6794,0.6668,0.6629,0.61545,0.55025,0.42025,0.1464]
    #                          ],
    #
    #                prune_ratio=[0.8,0.83,0.85,0.87,0.9,0.93,0.95,0.98],
    #                legends=['DualPrune','PFS','EB-Tickets','Rethink','SFP'],
    #                exp_name='Pruning_VGG-16_on_CIFAR-100')

    # # # Pruning ResNet-56 on CIFAR-100
    # acc_pruneratio(acc_list=[[0.6579, 0.6659, 0.6527, 0.6457, 0.619, 0.5897, 0.5585, 0.4197],
    #                          [0.643, 0.6319, 0.6272, 0.61365, 0.596, 0, 0, 0],
    #                          [0.6437, 0.6360333333, 0.6229333333, 0.6069, 0.5615333333, 0.49486666669999996, 0.43573333329999997, 0.18109999999999998],
    #                          [0.5213, 0.48755, 0.4587, 0.4381, 0.42025, 0, 0, 0],
    #                          [0.62945, 0.6123, 0.608, 0.5983, 0.52245, 0.5352, 0.4929, 0.07775],
    #                          [0.6579,0.65,0,0,0,0,0,0]
    #                          ],
    #                prune_ratio=[0.8,0.83,0.85,0.87,0.9,0.93,0.95,0.98],#[0.7,0.75,0.8,0.85,0.9,0.95],
    #                legends=['DualPrune','PFS','EB-Tickets','Rethink','SFP','DPFPS'],
    #                exp_name='Pruning_ResNet-56_on_CIFAR-100')

    #ablation study part module
    # acc_pruneratio(acc_list=[[0.9107,0.8984,0.8903,0.8916,0.8871,0.876,0.8613,0.8096],
    #                         [0.8964, 0.8873, 0.8897, 0.8855, 0.8707, 0.862, 0.8405,0.602],
    #                          [0.9011, 0.8918, 0.8964, 0.8849, 0, 0, 0,0],
    #                          ],
    #                prune_ratio=[0.8,0.83,0.85,0.87,0.9,0.93,0.95,0.98],
    #                legends=['DualPrune','w/o Graph Attention', 'w/o Side-path'],
    #                exp_name='effect_of_two_modules'
    #                )

    # acc_pruneratio(acc_list=[[0.704,0.7002,0.6948,0.6835,0.6802,0.6544,0.6307,0.5429],
    #                          [0.6946,0.6838,0.6782,0.6722,0.6579,0.6458,0.6167,0.3187],
    #                          [0.6936, 0.6864, 0.6897, 0.6733, 0.6695, 0.64, 0.6195, 0.5488],
    #                          ],
    #                prune_ratio=[0.8,0.83,0.85,0.87,0.9,0.93,0.95,0.98],
    #                legends=['DualPrune','w/o Graph Attention', 'w/o Side-path'],
    #                exp_name='effect_of_two_modules'
    #                )


    # #draw the side-attention of the net
    # # resnet56
    # net = resnet_cifar.resnet56(num_classes=10).cuda()
    # net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
    #                                                                        net_name='resnet56',
    #                                                                        dataset_name='cifar10',
    #                                                                        mask_update_epochs=900,
    #                                                                        mask_update_freq=1000,
    #                                                                        flop_expected=126550666 * (1 - 0.9),
    #                                                                        gcn_layer_num=2,
    #                                                                        mask_training_start_epoch=1,
    #                                                                        mask_training_stop_epoch=80,
    #                                                                        batch_size=128,
    #                                                                        add_shortcut_ratio=0.9
    #                                                                        )
    # net = net.cuda()
    # i = 3
    # checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet56', str(i) + '.pth'),map_location='cpu')
    #
    # for key in list(checkpoint['state_dict'].keys()):
    #     if 'zero_vec' in key or 'eye_mat' in key or 'gat_layers.0.adj' in key or 'gat_layers.1.adj' in key:
    #         checkpoint['state_dict'].pop(key)
    # net.load_state_dict(checkpoint['state_dict'])
    # mask = net.extractor(net)  # predict mask using extractor
    # mask=mask.abs()
    # lo = hi = 0
    # last_conv_mask = None
    # for name, mod in net.net.named_modules():
    #     if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
    #         hi += mod.out_channels
    #         mod.set_mask(mask[lo:hi].view(-1))  # update mask for each conv
    #         lo = hi
    #         last_conv_mask = mod.mask
    # fig=draw_masked_net(net,pic_name='resnet56_'+str(i),path='/home/victorfang/')
    # # fig = draw_gat_attention(net, pic_name='resnet56_gat_' + str(i), path='/home/victorfang/')
    # print()

    # total_flop = 314017290
    # prune_ratio = 0.9
    # flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
    # num_epochs = 160 * 1 + 20
    # learning_rate = {'default': 0.1, 'extractor': 0.0001}
    # weight_decay = {'default': 5e-4, 'extractor': 0}
    # net = vgg.vgg16_bn(dataset_name='cifar10').cuda()
    # net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
    #                                                                        net_name='vgg16_bn',
    #                                                                        dataset_name='cifar10',
    #                                                                        mask_update_epochs=900,
    #                                                                        mask_update_freq=1000,
    #                                                                        flop_expected=flop_expected,
    #                                                                        mask_training_start_epoch=1,
    #                                                                        mask_training_stop_epoch=80,
    #                                                                        batch_size=128,
    #                                                                        add_shortcut_ratio=0.9
    #                                                                        )
    # net = net.cuda()
    # i=11
    # # checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'vgg16', str(i) + '.pth'), map_location='cpu')
    # checkpoint = torch.load('/home/victorfang/model_pytorch/data/model_saved/gat_vgg16bn_predicted_mask_and_variable_shortcut_net_mask_newinner_meanstd_sameparam_11/checkpoint/masked_net.pth')
    # net.load_state_dict(checkpoint['state_dict'])
    # mask = net.extractor(net)  # predict mask using extractor
    # mask = mask.abs()
    # lo = hi = 0
    # last_conv_mask = None
    # for name, mod in net.net.named_modules():
    #     if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
    #         hi += mod.out_channels
    #         mod.set_mask(mask[lo:hi].view(-1))  # update mask for each conv
    #         lo = hi
    #         last_conv_mask = mod.mask
    # fig = draw_masked_net(net, pic_name='vgg16_' + str(i), path='/home/victorfang/')
    # print()

    # #draw the gat attention of the network
    # #draw the side-attention of the net
    # # resnet56
    # net = resnet_cifar.resnet56(num_classes=10).cuda()
    # net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
    #                                                                        net_name='resnet56',
    #                                                                        dataset_name='cifar10',
    #                                                                        mask_update_epochs=900,
    #                                                                        mask_update_freq=1000,
    #                                                                        flop_expected=126550666 * (1 - 0.9),
    #                                                                        gcn_layer_num=2,
    #                                                                        mask_training_start_epoch=1,
    #                                                                        mask_training_stop_epoch=80,
    #                                                                        batch_size=128,
    #                                                                        add_shortcut_ratio=0.9
    #                                                                        )
    # net = net.cuda()
    # i = 14
    # checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet56', str(i) + '.pth'),map_location='cpu')
    # for key in list(checkpoint['state_dict'].keys()):
    #     if 'zero_vec' in key or 'eye_mat' in key or 'gat_layers.0.adj' in key or 'gat_layers.1.adj' in key:
    #         checkpoint['state_dict'].pop(key)
    # net.load_state_dict(checkpoint['state_dict'])
    # mask = net.extractor(net)  # predict mask using extractor
    # mask=mask.abs()
    # lo = hi = 0
    # last_conv_mask = None
    # for name, mod in net.net.named_modules():
    #     if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
    #         hi += mod.out_channels
    #         mod.set_mask(mask[lo:hi].view(-1))  # update mask for each conv
    #         lo = hi
    #         last_conv_mask = mod.mask
    # fig=draw_gat_attention(net,pic_name='resnet56_gat_'+str(i),path='/home/victorfang/')
    # print()

    # #draw the feature map
    # # resnet56
    # net = resnet_cifar.resnet56(num_classes=10).cuda()
    # net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
    #                                                                        net_name='resnet56',
    #                                                                        dataset_name='cifar10',
    #                                                                        mask_update_epochs=900,
    #                                                                        mask_update_freq=1000,
    #                                                                        flop_expected=126550666 * (1 - 0.9),
    #                                                                        gcn_layer_num=2,
    #                                                                        mask_training_start_epoch=1,
    #                                                                        mask_training_stop_epoch=80,
    #                                                                        batch_size=128,
    #                                                                        add_shortcut_ratio=0.9
    #                                                                        )
    # net = net.cuda()
    # i = 3
    # checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet56', str(i) + '.pth'), map_location='cpu')
    # for key in list(checkpoint['state_dict'].keys()):
    #     if 'zero_vec' in key or 'eye_mat' in key or 'gat_layers.0.adj' in key or 'gat_layers.1.adj' in key:
    #         checkpoint['state_dict'].pop(key)
    # dl, _ = data_loader.create_train_loader(batch_size=64,
    #                                         num_workers=0,
    #                                         dataset_name='cifar10',
    #                                         )
    # net.load_state_dict(checkpoint['state_dict'])
    # show_feature_map(net,
    #                  data_loader=dl,
    #                  layer_indexes=0,
    #                  num_image_show=16,
    #                  col=4)
