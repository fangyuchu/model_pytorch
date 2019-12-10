import torch
import transform_conv
import torch.nn as nn
import vgg
import transform_conv

class gcn(nn.Module):
    def __init__(self,in_features,out_features):
        super(gcn, self).__init__()
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
        while rounds>0:
            rounds-=1
            for name, module in net.named_modules():
                if isinstance(module,nn.Conv2d):
                    weight=transform_conv.conv_to_matrix(module)
                    filter_sum=weight.sum(dim=[1])
                    #todo:这样加上去原来的权重不就变了吗？
        print()

def no_name():
    a = torch.nn.Conv2d(in_channels=5, out_channels=64, kernel_size=(3, 3))
    b = transform_conv.conv_to_matrix(a)
    c = b.sum(dim=[1])
    b = a.view(-1)
    b[0] = 3
    print()






if __name__ == "__main__":
    net=vgg.vgg16_bn(pretrained=False)
    for name, module in net.named_modules():
        if isinstance(module,torch.nn.Conv2d):
            w=module.weight.data
            w[0,0,0,0]=1000
            print(name)
    no_name()