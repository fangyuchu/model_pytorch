import torch
import transform_conv
from transform_conv import conv_to_matrix
import torch.nn as nn
from network import vgg,resnet_cifar,resnet
import copy
from framework import evaluate


class gcn(nn.Module):
    def __init__(self,in_features,out_features):
        super(gcn, self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.network=nn.Sequential(
            nn.Linear(in_features,128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128,out_features)
        )
    def forward(self, net,net_name,dataset_name, rounds=2):
        if 'vgg' in net_name:
            return self.forward_vgg(net,rounds)
        elif 'resnet' in net_name:
            return self.forward_resnet(net,rounds)

    def forward_resnet(self,net,rounds):
        first_conv=True
        for name,mod in net.named_modules():
            if first_conv and isinstance(mod,nn.Conv2d):
                weight=conv_to_matrix(copy.deepcopy(mod))
                information = weight.mean(dim=1).reshape([-1, 1])               # calculate the mean of current layer
                first_conv = False
            if isinstance(mod,resnet.Bottleneck):
                information=self.aggregate_bottleneck(mod,information)          #conv at present is the last conv in front of bottleneck


    def aggregate_bottleneck(self,bottleneck,information_in_front):
        '''

        :param bottleneck: bottleneck module
        :param conv_in_front: conv module in front of bottleneck
        :return:
        '''
        information_downsample=None
        weight_dowmsample=None
        conv_list=[]
        for name,mod in bottleneck.named_modules():

            if isinstance(mod,nn.Conv2d):
                if 'downsample' in name:
                    weight_dowmsample = conv_to_matrix(copy.deepcopy(mod))
                    information_downsample=weight_dowmsample.mean(dim=1).reshape([-1, 1])
                    conv_downsample=copy.deepcopy(mod)
                    continue
                conv_list+=[copy.deepcopy(mod)]      #a list containing 2-d conv weight matrix, parameters in original nets will not be updated

        if weight_dowmsample is not None:
            weight_dowmsample+=information_in_front.repeat(1, 1).view(-1)
        self.aggregate_convs(conv_list,information_in_front)
        weight_list=[]
        for conv in conv_list:
            weight_list+=[conv_to_matrix(conv)]
        information_at_last=weight_list[-1].mean(dim=1).reshape([-1, 1])
        if weight_dowmsample is not None:
            information_at_last+=weight_dowmsample.mean(dim=1).reshape([-1, 1])

        print()
        return 0

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

        for i in range(len(conv_list)):
            kernel_size=conv_list[i].kernel_size[0]*conv_list[i].kernel_size[1]
            mean = mean.repeat(1, kernel_size).view(-1)  # expand each value for 9 times.
            # this implies that the default size of kernel is 3x3
            weight_list[i] += mean  # aggregate the mean from previous layer
            mean = weight_list[i].mean(dim=1).reshape([-1, 1])  # calculate the mean of current layer

    def forward_vgg(self, net, rounds):
        '''

        :param net:
        :param rounds:
        :return: extracted-features representing the cross layer relationship for each filter
        '''
        weight_list=[]
        for mod in net.modules():                                                   #mod is a copy
            if isinstance(mod,nn.Conv2d):
                # weight_list+=[conv_to_matrix(mod)]                   #a list containing 2-d conv weight matrix
                weight_list+=[conv_to_matrix(copy.deepcopy(mod))]      #a list containing 2-d conv weight matrix, parameters in original nets will not be updated

        while rounds>0:
            rounds-=1
            mean = torch.zeros(3, 1).to(weight_list[0].device)                      #initialize mean for first layer
            for i in range(len(weight_list)):
                mean=mean.repeat(1,9).view(-1)                                      #expand each value for 9 times.
                                                                                    #this implies that the default size of kernel is 3x3
                weight_list[i]+=mean                                                #aggregate the mean from previous layer
                mean=weight_list[i].mean(dim=1).reshape([-1,1])                     #calculate the mean of current layer

        gcn_feature_in=[]
        for i in range(len(weight_list)):
            gcn_feature_in+=[pca(weight_list[i],dim=self.in_features)]              #reduce the dimension of all filters to same value

        gcn_feature_out=[]
        for i in range(len(gcn_feature_in)):
            gcn_feature_out+=[self.network(gcn_feature_in[i])]                      #foward propagate

        return gcn_feature_out                                                      #each object represents one conv



def pca(tensor_2d,dim):
    '''

    :param tensor_2d: each row is a piece of data
    :param dim:
    :return: tensor after dimension reduction,each row is a piece of data
    '''
    u,s,v=torch.svd(tensor_2d)
    projection_matrix=v[:,:dim]
    return torch.matmul(tensor_2d,projection_matrix)









if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # net= vgg.vgg16_bn(pretrained=True).to(device)
    net1= resnet_cifar.resnet56()
    net2=resnet.resnet50()

    test=gcn(in_features=27,out_features=10).to(device)
    c=test.forward(net=net2,rounds=2,net_name='resnet50',dataset_name='imagenet')
    print()
    # for name, module in network.named_modules():
    #     if isinstance(module,torch.nn.Conv2d):
    #         w=module.weight.data
    #         w[0,0,0,0]=1000
    #         print(name)
