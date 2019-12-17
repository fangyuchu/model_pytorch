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
        net=copy.deepcopy(net)
        while rounds>0:                                                                     #aggregate information in convs
            rounds-=1
            first_conv=True
            for name,mod in net.named_modules():
                if first_conv and isinstance(mod,nn.Conv2d):                                #first conv in the ResNet
                    weight=conv_to_matrix(mod)
                    information_at_last = weight.mean(dim=1).reshape([-1, 1])                 # calculate the mean of current layer
                    first_conv = False
                if isinstance(mod,resnet.Bottleneck):
                    _,information_at_last=self.aggregate_bottleneck(mod,information_at_last)

        weight_list=[]
        for mod in net.modules():
            if isinstance(mod,nn.Conv2d):
                weight_list+=[conv_to_matrix(mod)]

        gcn_feature_in = []
        for i in range(len(weight_list)):
            gcn_feature_in += [
                pca(weight_list[i], dim=self.in_features)]  # reduce the dimension of all filters to same value

        gcn_feature_out = []
        for i in range(len(gcn_feature_in)):
            gcn_feature_out += [self.network(gcn_feature_in[i])]  # foward propagate

        return gcn_feature_out  # each object represents one conv

    def aggregate_bottleneck(self,bottleneck,information_in_front):
        '''

        :param bottleneck: bottleneck module
        :param conv_in_front: conv module in front of bottleneck
        :return:
        '''
        weight_dowmsample=None
        conv_list=[]
        for name,mod in bottleneck.named_modules():

            if isinstance(mod,nn.Conv2d):
                if 'downsample' in name:
                    weight_dowmsample = conv_to_matrix(mod)
                    continue
                conv_list+=[mod]      #a list containing 2-d conv weight matrix, parameters in original nets will not be updated

        _,information_at_last=self.aggregate_convs(conv_list,information_in_front)

        # shortcut
        if weight_dowmsample is not None:                                                       #with 1x1 conv
            weight_dowmsample += information_in_front.repeat(1, 1).view(-1)
            information_at_last+=weight_dowmsample.mean(dim=1).reshape([-1, 1])
        else:                                                                                   #without 1x1 conv
            information_at_last+=information_in_front

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

        for i in range(len(conv_list)):
            kernel_size=conv_list[i].kernel_size[0]*conv_list[i].kernel_size[1]
            mean = mean.repeat(1, kernel_size).view(-1)  # expand each value for 9 times.
            weight_list[i] += mean  # aggregate the mean from previous layer
            mean = weight_list[i].mean(dim=1).reshape([-1, 1])  # calculate the mean of current layer
        information_at_last=mean
        return conv_list,information_at_last                    #information from the last conv

    def forward_vgg(self, net, rounds):
        '''

        :param net:
        :param rounds:
        :return: extracted-features representing the cross layer relationship for each filter
        '''
        net=copy.deepcopy(net)
        conv_list=[]
        for mod in net.modules():
            if isinstance(mod,nn.Conv2d):
                conv_list+=[mod]
        while rounds>0:
            rounds-=1
            mean = torch.zeros(3, 1).to(net.features[0].weight.device)                      #initialize mean for first layer
            self.aggregate_convs(conv_list,mean)

        weight_list=[]
        for mod in net.modules():
            if isinstance(mod,nn.Conv2d):
                weight_list+=[conv_to_matrix(mod)]

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
    if dim>tensor_2d.shape[1]:
        raise Exception('Required dim is larger than feature len.(dim:{}>tensor_2d.shape[1]:{})'.format(dim,tensor_2d.shape[1]))
    u,s,v=torch.svd(tensor_2d)
    projection_matrix=v[:,:dim]
    return torch.matmul(tensor_2d,projection_matrix)









if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # net= vgg.vgg16_bn(pretrained=True).to(device)
    net1= resnet_cifar.resnet56().to(device)
    net2=resnet.resnet50().to(device)
    net3=vgg.vgg16_bn().to(device)


    test=gcn(in_features=27,out_features=10).to(device)


    test.forward(net=net3,net_name='vgg16_bn',dataset_name='imagenet',rounds=2)

    c=test.forward(net=net2,rounds=2,net_name='resnet50',dataset_name='imagenet')
    print()
    # for name, module in network.named_modules():
    #     if isinstance(module,torch.nn.Conv2d):
    #         w=module.weight.data
    #         w[0,0,0,0]=1000
    #         print(name)
