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


neural_dead_times=7000
filter_dead_ratio=0.85




def read_data(total_num,
              path='/home/victorfang/Desktop/pytorch_model/vgg16_bn_cifar10_dead_neural_pruned/dead_neural',
              balance=False):
    dead_filter = np.empty(shape=[0, 3, 3, 3])
    living_filter = np.empty(shape=[0, 3, 3, 3])
    for Round in range(1,total_num+1):
        checkpoint=torch.load(path+'/round '+str(Round)+'.tar')
        net=checkpoint['net']
        net.load_state_dict(checkpoint['state_dict'])
        neural_list=checkpoint['neural_list']
        relu_list=checkpoint['relu_list']

        num_conv = 0  # num of conv layers in the net
        filter_num=list()
        filters=list()
        for mod in net.features:
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                num_conv += 1
                filter_num.append(mod.out_channels)
                filters.append(mod)

        for i in range(1):
            for relu_key in list(neural_list.keys()):
                if relu_list[i] is relu_key:                                    #find the neural_list_statistics in layer i+1
                    dead_relu_list=neural_list[relu_key]
                    neural_num=dead_relu_list.shape[1]*dead_relu_list.shape[2]  #neural num for one filter

                    # judge dead filter by neural_dead_times and dead_filter_ratio
                    dead_relu_list[dead_relu_list<neural_dead_times]=0
                    dead_relu_list[dead_relu_list>=neural_dead_times]=1
                    dead_relu_list=np.sum(dead_relu_list,axis=(1,2))            #count the number of dead neural for one filter
                    dead_filter_index=np.where(dead_relu_list>neural_num*filter_dead_ratio)[0].tolist()
                    living_filter_index=[i for i in range(filter_num[i]) if i not in dead_filter_index]
                    dead_filter=np.append(dead_filter,filters[i].weight.data.cpu().numpy()[dead_filter_index],axis=0)
                    living_filter=np.append(living_filter,filters[i].weight.data.cpu().numpy()[living_filter_index],axis=0)
                    print(Round)
    dead_filter=np.reshape(dead_filter,(-1,27))
    living_filter=np.reshape(living_filter,(-1,27))
    x=np.vstack((dead_filter,living_filter))
    y=np.zeros(x.shape[0])
    y[[i for i in range(dead_filter.shape[0])]]=1

    if balance is True:
        x=x[:dead_filter.shape[0]*2]
        y=y[:dead_filter.shape[0]*2]
    return x,y

if __name__ == "__main__":
    train_x,train_y=read_data(total_num=41,
                              balance=True,
                              path='/home/victorfang/Desktop/pytorch_model/vgg16_bn_cifar10_dead_neural_pruned/dead_neural')
    validation_x,validation_y=read_data(total_num=22,
                                        path='/home/victorfang/Desktop/pytorch_model/test1/dead_neural')
    svc=svm.SVC()
    svc.fit(train_x,train_y)
    predict_y=svc.predict(validation_x)
    print()



