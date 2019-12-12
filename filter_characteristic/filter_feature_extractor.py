import torch
from filter_characteristic.graph_convolutional_network import gcn
import torch.nn as nn
import transform_conv
from network import vgg
import network.vgg as vgg
import os
import numpy as np
from network import storage

class extractor(nn.Module):
    def __init__(self,net,feature_len,gcn_rounds=2):
        super(extractor, self).__init__()
        self.gcn=gcn(in_features=27,out_features=feature_len)
        self.feature_len=feature_len
        self.net=net
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
        
    def forward(self ):
        crosslayer_features=self.gcn.forward(net=self.net,rounds=self.gcn_rounds)

        filter_num=[]
        singular_value_list=[]
        for mod in self.net.modules():                                                   #mod is a copy
            if isinstance(mod,nn.Conv2d):
                filter_num+=[mod.out_channels]
                weight= transform_conv.conv_to_matrix(mod)
                u, s, v = torch.svd(weight)
                singular_value_list+=[s[:self.feature_len]]

        features=torch.zeros((sum(filter_num),self.feature_len*2)).cuda()
        start=0
        for i in range(len(filter_num)):
            stop = start+filter_num[i]
            features[start:stop]=torch.cat((crosslayer_features[i],singular_value_list[i].repeat(filter_num[i],1)),dim=1)
            start=stop
        return self.network(features)

def read_data(path='/home/victorfang/PycharmProjects/model_pytorch/data/最少样本测试/训练集',
              batch_size=None):
    nets=list()
    filter_layer=list()
    filter_label=list()

    file_list = os.listdir(path)
    for file_name in file_list:
        if '.tar' in file_name:
            checkpoint=torch.load(os.path.join(path,file_name))
            net=storage.restore_net(checkpoint)
            net.load_state_dict(checkpoint['state_dict'])
            nets+=[net]
            neural_list=checkpoint['neural_list']
            try:
                module_list=checkpoint['module_list']
            except KeyError:
                module_list=checkpoint['relu_list']
            if batch_size is None:
                batch_size=checkpoint['batch_size']

            num_conv = 0  # num of conv layers in the network
            filters=list()
            layers=list()
            for mod in net.modules():
                if isinstance(mod, torch.nn.modules.conv.Conv2d):
                    num_conv += 1
                    conv=mod
                elif isinstance(mod,torch.nn.ReLU):                             #ensure the conv are followed by relu
                    if layers != [] and layers[-1] == num_conv - 1:             # get rid of the influence from relu in fc
                        continue
                    filters.append(conv)
                    layers.append(num_conv-1)

            for i in range(len(filters)):
                for module_key in list(neural_list.keys()):
                    if module_list[i] is module_key:                                    #find the neural_list_statistics in layer i+1
                        dead_times=neural_list[module_key]
                        neural_num=dead_times.shape[1]*dead_times.shape[2]              #neural num for one filter
                        filter_weight = filters[i].weight.data.cpu().numpy()
                        #compute sum(dead_times)/(batch_size*neural_num) as label for each filter
                        dead_times=np.sum(dead_times,axis=(1,2))
                        prediction=dead_times/(neural_num*batch_size)
                        filter_label+=prediction.tolist()
                        filter_layer+=[layers[i] for j in range(filter_weight.shape[0])]

        return nets,filter_label,filter_layer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    read_data(batch_size=10000)


    net= vgg.vgg16_bn(pretrained=True).to(device)
    model=extractor(net=net,feature_len=15,gcn_rounds=3).to(device)
    c=model.forward()
    d=torch.sum(c)
    d.backward()
    print()