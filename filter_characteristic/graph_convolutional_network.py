import torch
import transform_conv
from transform_conv import conv_to_matrix
import torch.nn as nn
from network import vgg,resnet_cifar,resnet,net_with_predicted_mask
import copy
from network.modules import conv2d_with_mask_shortcut,block_with_mask_shortcut
from framework import evaluate


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
        self.normalization = nn.BatchNorm1d(num_features=in_features,track_running_stats=False)
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
                    if isinstance(mod,conv2d_with_mask_shortcut):                           #first conv2d_with_mask_shortcut has conv in downsample
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
        output=self.network(features)

        return output  # each object represents one conv

    def aggregate_block(self,block,information_in_front):
        '''
        aggregate basicblock and bottleneck in resnet
        :param block: block module
        :param conv_in_front: conv module in front of block
        :return:
        '''
        device = self.network[0].weight.device
        weight_dowmsample=None
        zero_padding=False
        conv_list=[]
        for name, mod in block.named_modules():
            if 'downsample' in name and isinstance(mod, resnet_cifar.DownsampleA):  # for basic block
                zero_padding = True
            if isinstance(mod, nn.Conv2d):
                if 'downsample' in name:  # shortcut conv for bottleneck or for conv2d_with_mask_shortcut
                    if 'conv' not in name:  # shortcut conv for bottleneck
                        raise Exception('check if this is right. why downsample inside conv?')
                        weight_dowmsample = conv_to_matrix(mod)
                    continue
                conv_list += [mod]  # a list containing 2-d conv weight matrix

        _, information_at_last = self.aggregate_convs(conv_list, information_in_front)

        # shortcut
        if weight_dowmsample is not None:                                                       #with 1x1 conv
            weight_dowmsample =(weight_dowmsample+ information_in_front.repeat(1, 1).view(-1))/2
            information_at_last=(information_at_last+ weight_dowmsample.mean(dim=1).reshape([-1, 1]))/2
            raise Exception('check if this is right for bottleneck')
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
        #     if isinstance(conv_list[i],conv2d_with_mask_shortcut):
        #         if len(conv_list[i].downsample) == 0:  # direct shortcut w/o 1x1 conv
        #             mean = weight_list[i].mean(dim=1).reshape([-1, 1]) + old_mean
        #         else:  # shortcut with 1x1 conv
        #             weight_downsample = conv_to_matrix(conv_list[i].downsample[0])
        #             dowmsample_mean = (weight_downsample + old_mean.view(-1)).mean(dim=1)
        #             mean = (weight_list[i].mean(dim=1) + dowmsample_mean).reshape([-1, 1])
        #     else:
        #         mean = weight_list[i].mean(dim=1).reshape([-1, 1])  # calculate the mean of current layer

        for i in range(len(conv_list)):
            kernel_size=conv_list[i].kernel_size[0]*conv_list[i].kernel_size[1]
            old_mean=mean
            mean = mean.repeat(1, kernel_size).view(-1)  # expand each value for 9 times.
            weight_list[i] += mean  # aggregate the mean from previous layer
            weight_list[i]=weight_list[i]/2+mean/2
            if isinstance(conv_list[i],conv2d_with_mask_shortcut):
                if len(conv_list[i].downsample) == 0:  # direct shortcut w/o 1x1 conv
                    mean = weight_list[i].mean(dim=1).reshape([-1, 1])/2 + old_mean/2
                else:  # shortcut with 1x1 conv
                    weight_downsample = conv_to_matrix(conv_list[i].downsample[0])
                    dowmsample_mean = (weight_downsample + old_mean.view(-1)).mean(dim=1)/2
                    mean = (weight_list[i].mean(dim=1) + dowmsample_mean).reshape([-1, 1])/2
            else:
                mean = weight_list[i].mean(dim=1).reshape([-1, 1])  # calculate the mean of current layer

        information_at_last=mean
        return conv_list, information_at_last  # information from the last conv





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
