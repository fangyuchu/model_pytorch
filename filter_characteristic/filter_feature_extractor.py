import os,sys
sys.path.append('../')
import torch
from filter_characteristic.graph_convolutional_network import gcn,gat
from transform_conv import conv_to_matrix,pca
import torch.nn as nn
import transform_conv
from network import vgg,net_with_predicted_mask
import network.vgg as vgg
import numpy as np
from network import storage
from framework import train
from datetime import datetime
from framework import config as conf
from random import shuffle
import copy
from filter_characteristic import predict_dead_filter
class extractor(nn.Module):
    def __init__(self,net,feature_len=20,layer_num=2):
        super(extractor, self).__init__()
        self.gat=gat(net,layer_num,feature_len)
        self.network = nn.Sequential(
            nn.Linear(feature_len, 1, bias=True),
            nn.BatchNorm1d(1, track_running_stats=False),
            nn.Tanh(),
        )

        # to be compatible with the older version
        self.only_gcn=True
        self.only_inner_features=False
        self.feature_len=feature_len
        self.gcn_layer_num=layer_num
    def forward(self,net,*args):
        hidden_states=self.gat(net)
        return self.network(hidden_states)


class extractor_cvpr(nn.Module):
    def __init__(self,feature_len,gcn_layer_num=2,only_gcn=False,only_inner_features=False):
        super(extractor_cvpr, self).__init__()
        self.only_gcn = only_gcn
        if only_gcn:  # only use gcn for prediction
            self.gcn = gcn(in_features=feature_len, out_features=1)
        else:
            self.gcn = gcn(in_features=feature_len, out_features=feature_len)
        self.only_inner_features = only_inner_features
        self.feature_len = feature_len
        self.gcn_layer_num = gcn_layer_num
        if not only_inner_features:
            in_features = feature_len + feature_len
        else:
            in_features = feature_len
        self.network = nn.Sequential(
            # nn.Linear(in_features,128),
            # nn.BatchNorm1d(128,track_running_stats=False),
            # nn.ReLU(True),
            nn.Linear(in_features,1,bias=True),
            nn.BatchNorm1d(1,track_running_stats=False),
            nn.Tanh(),
        )
        self.normalization=nn.BatchNorm1d(num_features=in_features,track_running_stats=False)
        
    def forward(self,net,net_name,dataset_name ):
        crosslayer_features=self.gcn.forward(net=net,rounds=self.gcn_layer_num,net_name=net_name,dataset_name=dataset_name)
        if self.only_gcn:                                                                      #only use gcn for prediction
            return crosslayer_features

        innerlayer_features=self.extract_innerlayer_features(net)
        if not self.only_inner_features:
            features=torch.cat((crosslayer_features,innerlayer_features),dim=1)
        else:
            features=innerlayer_features

        features=self.normalization(features)

        out = self.network(features)
        return out

    def extract_innerlayer_features(self, net):
        device = list(net.parameters())[0].device
        num_conv=0
        for name,mod in net.named_modules():
            if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
                num_conv+=mod.out_channels
        features=torch.ones((num_conv,self.feature_len)).to(device)
        lo=hi=0
        for name,mod in net.named_modules():
            if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
                hi+=mod.out_channels
                features[lo:hi]=pca(conv_to_matrix(mod),dim=self.feature_len)
                lo=hi
        return features


    # def extract_innerlayer_features(self,net):
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     channel_num=[]
    #     filter_num=[]
    #     for name, mod in net.named_modules():
    #         if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
    #             filter_num += [mod.out_channels]
    #             channel_num += [mod.in_channels]
    #
    #     # singular values are not used for efficiency concern
    #     # singular_values = torch.zeros((sum(filter_num), self.feature_len - 5)).to(device)
    #     mean = torch.zeros(sum(filter_num)).to(device)
    #     max = torch.zeros(sum(filter_num)).to(device)
    #     std = torch.zeros(sum(filter_num)).to(device)
    #     start=0
    #     i=0
    #     for name,mod in net.named_modules():
    #         if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
    #             stop = start + filter_num[i]
    #             weight= transform_conv.conv_to_matrix(mod)
    #             mean[start:stop]=torch.mean(weight,dim=1)
    #             max[start:stop]=torch.max(weight,dim=1)[0]
    #             std[start:stop]=torch.std(weight,dim=1)
    #             #singular values are not used for efficiency concern
    #             # u, s, v = torch.svd(weight)
    #             # singular_values[start:stop]=s[:self.feature_len-5].repeat(filter_num[i],1)
    #             start = stop
    #             i+=1
    #
    #     innerlayer_features = torch.zeros((sum(filter_num),5 )).to(device)
    #     start=0
    #     for i in range(len(filter_num)):
    #         stop = start+filter_num[i]
    #         innerlayer_features[start:stop][:,0]=i+1                                                                            #layer
    #         innerlayer_features[start:stop][:,1]=channel_num[i]                                                                  #channel_num
    #         innerlayer_features[start:stop][:,2]=mean[start:stop]                                                                        #mean
    #         innerlayer_features[start:stop][:,3]=max[start:stop]                                                                         #max
    #         innerlayer_features[start:stop][:,4]=std[start:stop]                                                                         #standard deviation
    #         # innerlayer_features[start:stop][:,5:]=singular_values[start:stop]                               #top k singuar value
    #         start=stop
    #
    #     return innerlayer_features

# class weighted_MSELoss(torch.nn.MSELoss):
#     def __init__(self):
#         super(weighted_MSELoss,self).__init__()
#
#     def forward(self, input, target):
#         device=input.device
#         ret = (input - target) ** 2
#         weight=torch.zeros(target.shape).to(device)
#         weight[target<0.3]=0.3
#         weight[target>=0.3]=target[target>=0.3]
#         ret = ret * weight
#         if self.reduction != 'none':
#             ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
#         return ret





def read_data(path,
              num_images=None):
    sample=[]
    file_list = os.listdir(path)
    file_list.sort()
    print(file_list)
    for file_name in file_list:
        if '.tar' in file_name:
            checkpoint=torch.load(os.path.join(path,file_name))
            net=storage.restore_net(checkpoint,pretrained=True)
            # from framework import measure_flops
            # measure_flops.measure_model(net, 'cifar10',print_flop=True)

            neural_list=checkpoint['neural_list']
            try:
                module_list=checkpoint['module_list']
            except KeyError:
                module_list=checkpoint['relu_list']

            num_conv = 0  # num of conv layers in the network
            filter_weight=[]
            layers=[]
            for mod in net.modules():
                if isinstance(mod, torch.nn.modules.conv.Conv2d):
                    num_conv += 1
                    conv=mod
                elif isinstance(mod,torch.nn.ReLU):                             #ensure the conv are followed by relu
                    if layers != [] and layers[-1] == num_conv - 1:             # get rid of the influence from relu in fc
                        continue
                    filter_weight.append(conv.weight.data.cpu().numpy())
                    layers.append(num_conv-1)

            filter_layer=[]
            filter_label=[]
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
            sample.append({'net':net,'filter_label':filter_label,'filter_layer':filter_layer,
                           'net_name':checkpoint['net_name'],'dataset_name':checkpoint['dataset_name']})

    return sample




if __name__ == "__main__":


    # read_data(num_images=10000)
    #
    #
    # net= vgg.vgg16_bn(pretrained=True).to(device)
    # model=extractor(net=net,feature_len=15,gcn_rounds=3).to(device)
    # c=model.forward()
    # d=torch.sum(c)
    # d.backward()
    print()