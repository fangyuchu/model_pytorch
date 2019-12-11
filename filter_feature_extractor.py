import torch
from graph_convolutional_network import gcn
import torch.nn as nn
import transform_conv
import vgg

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
            #todo:这一步自动求梯度了？
            features[start:stop]=torch.cat((crosslayer_features[i],singular_value_list[i].repeat(filter_num[i],1)),dim=1)
            start=stop
        return self.network(features)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # from torch.autograd import Variable
    # d=torch.zeros(2).cuda()
    # #为什么放到cuda上就没梯度了？？？
    # #答案：https://pytorch.org/docs/stable/autograd.html#torch.Tensor.is_leaf
    # a=Variable(torch.Tensor([1]).cuda(),requires_grad=True)
    # b=Variable(torch.Tensor([1]).cuda(),requires_grad=True)
    # c=a*a
    # d[:]=torch.cat((c,b))
    # e=torch.sum(d)
    # e.backward()
    #
    # import torch.optim as optim
    #
    # optimizer=optim.SGD([a,b],lr=1)
    # optimizer.step()


    net=vgg.vgg16_bn(pretrained=True).to(device)
    model=extractor(net=net,feature_len=15,gcn_rounds=3).to(device)
    c=model.forward()
    d=torch.sum(c)
    d.backward()
    print()