import torch
import torch.nn as nn
import numpy as np
from framework import data_loader, evaluate
from network import vgg, resnet_cifar,storage
from filter_characteristic import predict_dead_filter
import matplotlib.pyplot as plt
import copy
import time
import os


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

def acc_pruneratio(acc_list,prune_ratio,legends,exp_name):
    '''

    :param acc_list: one/two-dimension lists
    :param prune_ratio: one-dimension lists
    :param exp_name:
    :return:
    '''
    marker_list=['o','*','+']
    if type(acc_list[0]) is not list:
        acc_list=[acc_list]
    if acc_list[0][0]<1:
        acc_list=np.array(acc_list)*100
    if prune_ratio[0]<1:
        prune_ratio=np.array(prune_ratio)*100
    x=prune_ratio
    plt.figure()
    plt.axes(yscale="log")
    for i,y in enumerate(acc_list):
        plt.plot(x,y,marker=marker_list[i],label=legends[i])
    # plt.ylim(0.6,1)
    plt.xlabel('Pruned Flops%')
    plt.ylabel('Accuracy%')
    # plt.yticks([80,85,90],('80','85','90'))
    plt.ylim(bottom=60)
    plt.yticks([60,70,80,85,90],('60','70','80','85','90'))
    # plt.yticks([60,70,80,85,90],('60','70','80','85','90'))
    # plt.title(exp_name)
    plt.legend()
    plt.savefig('/home/victorfang/'+exp_name+'.eps')
    plt.show()


if __name__ == "__main__":

    # Pruning ResNet-56 on CIFAR-10
    acc_pruneratio(acc_list=[[0.909867,0.9079,0.9041,0.899567,0.89653,0.88683,0.881333,0.8641,0.726466],
                             [0.903967,0.904667,0.89613,0.896867,0.886733,0,0,0,0],
                             [0.918933,0.89,0.8842,0.8726,0.8643,0.8407,0.7488,0.6393,0.4423]],#[[0.9132,0.9099,0.9079,0.8996,0.8870,0.8641],
                             #[0.9077,0.9040,0.9047,0.8969,0,0],
                             #[0.9189,0.9139,0.8900,0.8726,0.8407,0.6393]],
                   prune_ratio=[0.75,0.8,0.83,0.85,0.87,0.9,0.93,0.95,0.98],#[0.7,0.75,0.8,0.85,0.9,0.95],
                   legends=['Ours','PFS','EB-Tickets'],
                   exp_name='Pruning_ResNet-56_on_CIFAR-10')

    # # Pruning VGG-16 on CIFAR-10
    # acc_pruneratio(acc_list=[[0.9249,0.917867,0.91753,0.9126,0.90705,0.9047,0.894133,0.859467],
    #                          [0.9144,0.9147,0.9101,0.908167,0.8978,0.895967,0.8816,0],
    #                          [0.919933,0.9102,0.9115,0.9074,0.901167,0.897,0.881567,0.566033]],
    #                prune_ratio=[0.8,0.83,0.85,0.87,0.9,0.93,0.95,0.98],
    #                legends=['Ours','PFS','EB-Tickets'],
    #                exp_name='Pruning_VGG-16_on_CIFAR-10')

    # #ablation study part module
    # acc_pruneratio(acc_list=[[0.9004, 0.8908, 0.8865, 0.8849, 0.8748, 0.832, 0.8076],
    #                          [0.9001, 0.8915, 0.8872, 0.8792, 0.867, 0.8312, 0.798],
    #                          [0.9054, 0.9038, 0.9012, 0.895, 0.8847, 0.8743, 0.8656]],
    #                prune_ratio=[0.796654646, 0.829760074, 0.846118297, 0.869739508, 0.896758742, 0.92985532,0.946386375],
    #                legends=['Only Graph Embedding', 'Only Filter Embedding', 'Filter2Vec'],
    #                exp_name='effect_of_filter2vec'
    #                )

    # fontsize=14
    # pruned_flop=[9.48,31.63,63.72,84.54,88.30]
    # acc_drop=[0.1,0.2,0.5,1,2]
    # plt.figure()
    # plt.plot(acc_drop, pruned_flop, 'bo--')
    # plt.yticks(fontsize=fontsize)
    # plt.xticks(acc_drop,acc_drop,fontsize=fontsize-2)
    #
    # plt.xlabel('Tolerance Accuracy Drop%', fontsize=fontsize)
    # plt.ylabel('Pruned FLOPs%', fontsize=fontsize)
    # plt.savefig('/home/victorfang/Desktop/vgg16_cifar10_tolerance.eps', format='eps')
    # plt.show()


