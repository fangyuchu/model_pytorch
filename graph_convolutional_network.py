import torch
import transform_conv
import torch.nn as nn
import vgg

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


    def forward(self, net, rounds=2):
        '''

        :param net:
        :param rounds:
        :return: extracted-features representing the cross layer relationship for each filter
        '''
        weight_list=[]
        for mod in net.modules():                                                   #mod is a copy
            if isinstance(mod,nn.Conv2d):
                weight_list+=[transform_conv.conv_to_matrix(mod)]                   #a list containing 2-d conv weight matrix

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

    net=vgg.vgg16_bn(pretrained=True).to(device)
    test=gcn(in_features=27,out_features=10).to(device)
    c=test.forward(net=net,rounds=2)
    print()
    # for name, module in net.named_modules():
    #     if isinstance(module,torch.nn.Conv2d):
    #         w=module.weight.data
    #         w[0,0,0,0]=1000
    #         print(name)
