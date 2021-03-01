import os,sys
sys.path.append('../')
import torch
import transform_conv
from transform_conv import conv_to_matrix,pca
import torch.nn as nn
from network import vgg,resnet_cifar,resnet,net_with_predicted_mask
import copy
from network.modules import block_with_mask_shortcut,block_with_mask_weighted_shortcut
import math
import torch.nn.functional as F

from framework import evaluate

class gat(nn.Module):
    def __init__(self,net,layer_num,embedding_feature_len):
        # self.adj
        # self.w
        # self.gat_layers
        super(gat, self).__init__()

        filter_num=[]
        filter_weight_num={}
        for name,mod in net.named_modules():
            if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
                filter_num+=[mod.out_channels]
                filter_weight_num[mod.in_channels*mod.kernel_size[0]*mod.kernel_size[1]]=1
        self.embedding_feature_len=embedding_feature_len
        self.register_buffer('initial_h',torch.ones((sum(filter_num),embedding_feature_len)))
        self.register_buffer('adj',torch.zeros((sum(filter_num),sum(filter_num))))
        row=last_num=filter_num[0]
        for i,num in enumerate(filter_num,start=1):
            self.adj[row:row+num,row-last_num:row]=1
            row+=num
            last_num=num
        for i in range(sum(filter_num)):
            self.adj[i][i]=0

        self.w={}
        for key in filter_weight_num.keys():
            self.w[str(key)]=nn.Linear(in_features=key,out_features=embedding_feature_len,bias=False)
        self.w=nn.ModuleDict(self.w)

        gat_layers=[GAT_layer(self.adj,embedding_feature_len,embedding_feature_len) for i in range(layer_num)]
        self.gat_layers=nn.Sequential(*gat_layers)

    def forward(self,net):
        # self.initial_h_list=[]
        self.initial_h=torch.ones_like(self.initial_h)
        i=0
        for name,mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                weight=conv_to_matrix(mod)
                filter_weight_num=mod.in_channels*mod.kernel_size[0]*mod.kernel_size[1]
                # self.initial_h[i:i+mod.out_channels]=pca(weight,dim=self.embedding_feature_len)
                self.initial_h[i:i+mod.out_channels]=self.w[str(filter_weight_num)](weight)
                i+=mod.out_channels
        embedding_features=self.gat_layers(self.initial_h)
        return embedding_features



class GAT_layer(nn.Module):
    def __init__(self,adjacent_matrix,in_feature_len,out_feature_len):
        super(GAT_layer, self).__init__()
        self.register_buffer('adj',adjacent_matrix)
        self.linear=nn.Linear(in_feature_len,out_feature_len)
        self.A=None
        self.register_buffer('zero_vec',-9e15*torch.ones_like(self.adj)) #so the output of softmax will be 0 for unconnected node
        self.register_buffer('eye_mat',torch.eye(self.adj.shape[0]))

    def forward(self,h):
        # compute attention matrix
        attention = h.mm(h.T) / math.sqrt(h.shape[1])
        self.A = torch.where(self.adj > 0, attention, self.zero_vec)
        self.A = F.softmax(self.A, dim=1)

        self.A = (self.A + self.eye_mat)/2  # set the attention of the node itself to 0.5.(the sum of neighbor's attention is also 0.5)

        num_filter_1layer=torch.sum(self.adj[:,0]!=0)
        self.A[:num_filter_1layer]=self.eye_mat[:num_filter_1layer]


        # for i in range(len(h)):


        #forward propagation
        h_new=self.linear(h)
        h_new=F.relu(self.A.mm(h_new))
        return h_new



class gcn(nn.Module):
    def __init__(self,in_features,out_features):
        super(gcn, self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.network=nn.Sequential(
            nn.Linear(in_features,128),
            nn.BatchNorm1d(128, track_running_stats=False),
            nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(128, 128),
            # nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(128,out_features),
            nn.BatchNorm1d(out_features, track_running_stats=False)
        )
        self.normalization = nn.BatchNorm1d(num_features=in_features,track_running_stats=False,affine=True)
    def forward(self, net,net_name,dataset_name, rounds=2):
        if 'vgg' in net_name:
            return self.forward_vgg(net,rounds)
        elif 'resnet' in net_name:
            return self.forward_resnet(net,rounds)

    def forward_vgg(self, net, rounds):
        '''

        :param net:
        :param rounds:
        :return: extracted-features representing the cross layer relationship for each filter
        '''
        device = self.network[0].weight.device
        if isinstance(net,net_with_predicted_mask.predicted_mask_net):
            net=net.copy()
        else:
            net=copy.deepcopy(net)
        conv_list=[]
        filter_num=[]
        for name,mod in net.named_modules():
            if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
                filter_num += [mod.out_channels]
                conv_list+=[mod]


        while rounds>0:
            rounds-=1
            mean = torch.zeros(3, 1).to(device)                      #initialize mean for first layer
            self.aggregate_convs(conv_list,mean)

        weight_list=[]
        for conv in conv_list:
            weight_list+=[conv_to_matrix(conv)]

        gcn_feature_in=[]
        for i in range(len(weight_list)):
            gcn_feature_in+=[pca(weight_list[i],dim=self.in_features)]              #reduce the dimension of all filters to same value

        features=torch.zeros((sum(filter_num),self.in_features)).to(device)
        start=0
        for i in range(len(filter_num)):
            stop = start+filter_num[i]
            features[start:stop]=gcn_feature_in[i]
            start=stop
        features = self.normalization(features)
        gcn_feature_out=self.network(features)

        return gcn_feature_out                                                     #each object represents one conv

    def forward_resnet(self,net,rounds):
        if isinstance(net,net_with_predicted_mask.predicted_mask_net):
            net=net.copy()
        else:
            net=copy.deepcopy(net)
        while rounds>0:                                                                     #aggregate information in convs
            rounds-=1
            first_conv=True
            for name,mod in net.named_modules():
                if first_conv and isinstance(mod,nn.Conv2d):                                #first conv in the ResNet
                    weight=conv_to_matrix(mod)
                    information_at_last = weight.mean(dim=1).reshape([-1, 1])                 # calculate the mean of current layer
                    if isinstance(mod, block_with_mask_shortcut):                           #first block_with_mask_shortcut has conv in downsample
                        downsample=mod.downsample[0].weight.mean(dim=1).reshape([-1,1])
                        information_at_last=information_at_last/2+downsample/2
                    first_conv = False
                if isinstance(mod,resnet.Bottleneck):
                    _,information_at_last=self.aggregate_block(mod,information_at_last)
                elif isinstance(mod,resnet_cifar.BasicBlock):
                    _,information_at_last=self.aggregate_block(mod,information_at_last)

        weight_list=[]
        for name,mod in net.named_modules():
            if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
                weight_list+=[conv_to_matrix(mod)]

        gcn_feature_in = []
        for i in range(len(weight_list)):
            gcn_feature_in += [
                pca(weight_list[i], dim=self.in_features)]  # reduce the dimension of all filters to same value

        features=gcn_feature_in[0]
        for i in range(1,len(gcn_feature_in)):
            features=torch.cat((features,gcn_feature_in[i]),dim=0)

        features=self.normalization(features)
        features=self.network(features)

        return features  # each object represents one conv

    def aggregate_block(self,block,information_in_front):
        '''
        aggregate basicblock and bottleneck in resnet
        :param block: block module
        :param conv_in_front: conv module in front of block
        :return:
        '''
        device = self.network[0].weight.device
        weight_downsample=None
        zero_padding=False
        conv_list=[]
        for name, mod in block.named_modules():
            if 'downsample' in name and isinstance(mod, resnet_cifar.DownsampleA):  # for basic block
                zero_padding = True
            if isinstance(mod, nn.Conv2d):
                if 'downsample' in name:  # shortcut conv for bottleneck or for block_with_mask_shortcut
                    if 'conv' not in name:  # shortcut conv for bottleneck
                        weight_downsample = conv_to_matrix(mod)
                    continue
                conv_list += [mod]  # a list containing 2-d conv weight matrix

        _, information_at_last = self.aggregate_convs(conv_list, information_in_front)

        # shortcut
        if weight_downsample is not None:                                                       #with 1x1 conv
            weight_downsample =(weight_downsample+ information_in_front.repeat(1, 1).view(-1))/2
            information_at_last=(information_at_last+ weight_downsample.mean(dim=1).reshape([-1, 1]))/2
        elif zero_padding is True:                                                              #with zero padding
            pad_length=information_at_last.shape[0]-information_in_front.shape[0]
            information_at_last =(information_at_last+ torch.cat((information_in_front, torch.zeros(pad_length,1).to(device)), 0))/2
        else:                                                                                   #identity map
            information_at_last=(information_at_last+information_in_front)/2

        return conv_list,information_at_last

    def aggregate_convs(self,conv_list,information_in_front):
        '''
        aggregate information for convs which were directely linked with each other
        :param conv_list: list containing conv modules.
                            Warnning: This method will modified the weights in conv. So if you don't want to change the
                            weights in the original network, make sure you input a deepcopy of conv to this method.
        :param information_in_front:
        :return:
        '''
        mean=information_in_front
        weight_list=[]
        for conv in conv_list:
            weight_list+=[conv_to_matrix(conv)]

        # for i in range(len(conv_list)):
        #     kernel_size=conv_list[i].kernel_size[0]*conv_list[i].kernel_size[1]
        #     old_mean=mean
        #     mean = mean.repeat(1, kernel_size).view(-1)  # expand each value for 9 times.
        #     weight_list[i] += mean  # aggregate the mean from previous layer
        #     if isinstance(conv_list[i],block_with_mask_shortcut):
        #         if len(conv_list[i].downsample) == 0:  # direct shortcut w/o 1x1 conv
        #             mean = weight_list[i].mean(dim=1).reshape([-1, 1]) + old_mean
        #         else:  # shortcut with 1x1 conv
        #             weight_downsample = conv_to_matrix(conv_list[i].downsample[0])
        #             downsample_mean = (weight_downsample + old_mean.view(-1)).mean(dim=1)
        #             mean = (weight_list[i].mean(dim=1) + downsample_mean).reshape([-1, 1])
        #     else:
        #         mean = weight_list[i].mean(dim=1).reshape([-1, 1])  # calculate the mean of current layer

        for i in range(len(conv_list)):
            kernel_size=conv_list[i].kernel_size[0]*conv_list[i].kernel_size[1]
            old_mean=mean
            mean = mean.repeat(1, kernel_size).view(-1)  # expand each value for 9 times.
            weight_list[i] += mean  # aggregate the mean from previous layer
            weight_list[i]=weight_list[i]/2+mean/2
            if isinstance(conv_list[i], block_with_mask_shortcut):
                if len(conv_list[i].downsample) == 0:  # direct shortcut w/o 1x1 conv
                    mean = weight_list[i].mean(dim=1).reshape([-1, 1])/2 + old_mean/2
                else:  # shortcut with 1x1 conv
                    weight_downsample = conv_to_matrix(conv_list[i].downsample[0])
                    downsample_mean = (weight_downsample + old_mean.view(-1)).mean(dim=1)/2
                    mean = (weight_list[i].mean(dim=1) + downsample_mean).reshape([-1, 1])/2
            else:
                mean = weight_list[i].mean(dim=1).reshape([-1, 1])  # calculate the mean of current layer

        information_at_last=mean
        return conv_list, information_at_last  # information from the last conv







def normalize(tensor):
    mean=tensor.mean(dim=0)
    std=tensor.std(dim=0)
    tensor=(tensor-mean)/std
    return tensor








if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # net= vgg.vgg16_bn(pretrained=True).to(device)
    net1= resnet_cifar.resnet56().to(device)

    i=0
    for mod in net1.modules():
        if isinstance(mod,torch.nn.Conv2d):
            i+=1

    net2=resnet.resnet50().to(device)
    net3=vgg.vgg16_bn().to(device)


    test=gcn(in_features=10,out_features=10).to(device)


    test.forward(net=net1,net_name='resnet56',dataset_name='cifar10',rounds=2)

    c=test.forward(net=net2,rounds=2,net_name='resnet50',dataset_name='imagenet')
    print()
    # for name, module in network.named_modules():
    #     if isinstance(module,torch.nn.Conv2d):
    #         w=module.weight.data
    #         w[0,0,0,0]=1000
    #         print(name)
