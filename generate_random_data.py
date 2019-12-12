import torch
import torch.nn as nn
from datetime import datetime
from framework import evaluate
import numpy as np
from network import create_net


def random_normal(num,dataset_name=None,size=[3,32,32],mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],is_image=True):
    '''
    generate random normal data to imitate image data
    :param num:number of images created
    :param size:size of image
    :param mean:mean of normal distribution
    :param std:std of normal distribution
    :return:
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name is 'cifar10':
        size=[3,32,32]
    if dataset_name is 'imagenet':
        size=[3,224,224]

    def one_tensor(is_image):
        simulated_tensor = np.random.normal(loc=mean, scale=std, size=(size[1],size[2],size[0]))   #generate data



        #this might change the distribution of the output image
        # if is_image is True:
        #     while simulated_tensor.max() >= 1 or simulated_tensor.min() < 0:            #ensure all data are in range[0,1]
        #         for i in range(3):
        #             simulated_tensor[:, :, i][simulated_tensor[:, :, i] >= 1] = np.random.normal(loc=mean[i], scale=std[i])
        #             simulated_tensor[:, :, i][simulated_tensor[:, :, i] <0] = np.random.normal(loc=mean[i], scale=std[i])


        simulated_tensor=simulated_tensor.swapaxes(0,2)
        for i in range(3):                                      #normalize the data
            simulated_tensor[i,:,:]=(simulated_tensor[i,:,:]-mean[i])/std[i]
        if is_image is False:
            simulated_tensor[simulated_tensor<0]=0
        return simulated_tensor

    out=np.zeros(shape=[num,size[0],size[1],size[2]],dtype=np.float)
    for i in range(num):
        out[i]=one_tensor(is_image=is_image)
    out=torch.from_numpy(out).type(torch.FloatTensor).to(device)
    return out

if __name__ == "__main__":

    random_normal(num=1000,dataset_name='cifar10',is_image=True)


    a=random_normal(100)
    net= create_net.vgg_cifar10(net_name='vgg16_bn')
    # loader=data_loader.create_validation_loader(batch_size=300,num_workers=2,dataset_name='cifar10')
    # evaluate.check_ReLU_alive(network=network,neural_dead_times=8000,data_loader=loader)
    relu_list,neural_list= evaluate.check_ReLU_alive(net=net, neural_dead_times=80, data=a)

    neural_dead_times = 80
    filter_dead_ratio = 0.9
    neural_dead_times_decay = 0.95
    filter_dead_ratio_decay = 0.98
    filter_preserve_ratio = 0.1
    max_filters_pruned_for_one_time = 0.3
    target_accuracy = 0.931
    batch_size = 300
    num_epoch = 300
    checkpoint_step = 1600


    num_conv = 0  # num of conv layers in the network
    filter_num_lower_bound=list()
    filter_num=list()
    for mod in net.features:
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv += 1
            filter_num_lower_bound.append(int(mod.out_channels*filter_preserve_ratio))
            filter_num.append(mod.out_channels)


    round=0
    while True:
        success=False
        round+=1
        print('{} start round {} of filter pruning.'.format(datetime.now(),round))
        print('{} current filter_dead_ratio:{},neural_dead_times:{}'.format(datetime.now(),filter_dead_ratio,neural_dead_times))

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