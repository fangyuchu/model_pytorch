import torch
from filter_characteristic.graph_convolutional_network import gcn
import torch.nn as nn
import transform_conv
from network import vgg
import network.vgg as vgg
import os
import numpy as np
from network import storage
from framework import train
from datetime import datetime
from framework import config as conf
from random import shuffle

import copy
from framework import evaluate

class extractor(nn.Module):
    def __init__(self,feature_len,gcn_rounds=2):
        super(extractor, self).__init__()
        self.gcn=gcn(in_features=27,out_features=feature_len)
        self.feature_len=feature_len
        self.gcn_rounds=gcn_rounds
        self.network=nn.Sequential(
            nn.Linear(feature_len * 2,128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128,1)
        )
        
    def forward(self,net ):
        crosslayer_features=self.gcn.forward(net=net,rounds=self.gcn_rounds)

        filter_num=[]
        singular_value_list=[]
        for mod in net.modules():                                                   #mod is a copy
            if isinstance(mod,nn.Conv2d):
                filter_num+=[mod.out_channels]
                weight= transform_conv.conv_to_matrix(copy.deepcopy(mod))
                u, s, v = torch.svd(weight)
                singular_value_list+=[s[:self.feature_len]]

        features=torch.zeros((sum(filter_num),self.feature_len*2)).cuda()
        start=0
        for i in range(len(filter_num)):
            stop = start+filter_num[i]
            features[start:stop]=torch.cat((crosslayer_features[i],singular_value_list[i].repeat(filter_num[i],1)),dim=1)
            start=stop

        return self.network(features)

def load(path):
    checkpoint=torch.load(path)
    feature_len=checkpoint['feature_len']
    gcn_rounds=checkpoint['gcn_rounds']
    net=extractor(feature_len=feature_len,gcn_rounds=gcn_rounds)
    net.load_state_dict(checkpoint['state_dict'])
    return net

def read_data(path='/home/victorfang/model_pytorch/data/最少样本测试/训练集',
              num_images=None):
    sample=list()
    file_list = os.listdir(path)
    for file_name in file_list:
        if '.tar' in file_name:
            checkpoint=torch.load(os.path.join(path,file_name))
            net=storage.restore_net(checkpoint)
            net.load_state_dict(checkpoint['state_dict'])
            neural_list=checkpoint['neural_list']
            try:
                module_list=checkpoint['module_list']
            except KeyError:
                module_list=checkpoint['relu_list']


            num_conv = 0  # num of conv layers in the network
            filter_weight=list()
            layers=list()
            for mod in net.modules():
                if isinstance(mod, torch.nn.modules.conv.Conv2d):
                    num_conv += 1
                    conv=mod
                elif isinstance(mod,torch.nn.ReLU):                             #ensure the conv are followed by relu
                    if layers != [] and layers[-1] == num_conv - 1:             # get rid of the influence from relu in fc
                        continue
                    filter_weight.append(conv.weight.data.cpu().numpy())
                    layers.append(num_conv-1)

            filter_layer=list()
            filter_label=list()
            for i in range(len(filter_weight)):
                for module_key in list(neural_list.keys()):
                    if module_list[i] is module_key:                                    #find the neural_list_statistics in layer i+1
                        dead_times=neural_list[module_key]
                        neural_num=dead_times.shape[1]*dead_times.shape[2]              #neural num for one filter

                        #compute sum(dead_times)/(num_images*neural_num) as label for each filter
                        dead_times=np.sum(dead_times,axis=(1,2))
                        prediction=dead_times/(neural_num*num_images)
                        filter_label+=prediction.tolist()
                        filter_layer+=[layers[i] for j in range(filter_weight[i].shape[0])]
            sample.append({'net':net,'filter_label':filter_label,'filter_layer':filter_layer})

    return sample

def train_extractor(path,epoch=1001,feature_len=27,gcn_rounds=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor_model=extractor(feature_len=feature_len,gcn_rounds=gcn_rounds).to(device)
    # extractor_model=torch.nn.DataParallel(extractor_model)
    sample_list=read_data(path=path,num_images=10000)
    optimizer=train.prepare_optimizer(net=extractor_model,optimizer=torch.optim.Adam,learning_rate=1e-2,weight_decay=0)
    # optimizer=train.prepare_optimizer(net=extractor_model,optimizer=torch.optim.SGD,learning_rate=1e-3,weight_decay=0)

    # criterion=torch.nn.MSELoss()
    criterion=torch.nn.L1Loss()

    checkpoint_path = conf.root_path + 'filter_feature_extractor' + '/checkpoint/l1norm/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
    order=[i for i in range(len(sample_list))]
    for i in range(epoch):
        total_loss=[0 for k in range(len(sample_list))]
        shuffle(order)
        # print(order)
        for j in order:

            sample=sample_list[j]
            net=sample['net']

            filter_label=sample['filter_label']
            label=torch.Tensor(filter_label).reshape((-1,1)).to(device)
            optimizer.zero_grad()

            output=extractor_model.forward(net)

            loss=criterion(output,label)
            loss.backward()

            total_loss[j]=float(loss)
            optimizer.step()

        print('{}  Epoch:{}. loss is {}. Sum:'.format(datetime.now(),i, total_loss),end='')
        print(sum(total_loss))
        if i%50==0 and i!=0:
            checkpoint={'feature_len':feature_len,
                        'gcn_rounds':gcn_rounds,
                        'state_dict':extractor_model.state_dict()}

            torch.save(checkpoint,os.path.join(checkpoint_path,str(i)+'.tar'))
            print(i)
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_extractor('/home/victorfang/model_pytorch/data/最少样本测试/训练集')


    test_net=vgg.vgg16_bn(pretrained=False,dataset_name='cifar10').to(device)
    path='/home/victorfang/model_pytorch/data/model_saved/filter_feature_extractor/checkpoint/l1norm/600.tar'
    extractor_model=load(path).to(device)
    sample_list=read_data(path='/home/victorfang/model_pytorch/data/最少样本测试/训练集',num_images=10000)
    criterion=torch.nn.MSELoss()
    for sample in sample_list:
        net = sample['net']
        if net.features[0].weight.shape[0]!=64:
            continue
        filter_label = sample['filter_label']
        label = torch.Tensor(filter_label).reshape((-1, 1)).to(device)

        output = extractor_model.forward(test_net)

        loss = criterion(output, label)

        from filter_characteristic import predict_dead_filter
        predict_dead_filter.filter_inactive_rate_ndcg(np.array(filter_label),output.data.detach().cpu().numpy().reshape(-1),0.3)

        print(float(loss))
        print()



    # read_data(num_images=10000)
    #
    #
    # net= vgg.vgg16_bn(pretrained=True).to(device)
    # model=extractor(net=net,feature_len=15,gcn_rounds=3).to(device)
    # c=model.forward()
    # d=torch.sum(c)
    # d.backward()
    # print()