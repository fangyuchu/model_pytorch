from network import vgg,resnet
from torch import nn
import torch
from prune.prune_module import get_module
class vgg16_bn_paper(nn.Module):
    def __init__(self,dataset_name,**kwargs):
        super(vgg16_bn_paper,self).__init__()
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg=vgg.vgg16_bn(dataset_name=dataset_name,**kwargs)
        self.mask=[torch.ones(3).to(device)]                                                                                #for rgb channels

        for name,mod in self.vgg.named_modules():
            if isinstance(mod,nn.Conv2d):

                self.mask+=[torch.ones(mod.out_channels).to(device)]
                _modules = get_module(model=self.vgg, name=name)
                _modules[name.split('.')[-1]] = Conv2d_mask(mod,mask=self.mask[-1],front_mask=self.mask[-2]).to(device)
                self.mask[-1][0]=0

    def forward(self, input):
        return self.vgg(input)

class Conv2d_mask(nn.Conv2d):
    def __init__(self,conv,mask,front_mask):
        super(Conv2d_mask,self).__init__( conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                 padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=(conv.bias is not None))
        self.weight=conv.weight
        if self.bias is not None:
            self.bias=conv.bias
        self.mask=mask
        self.front_mask=front_mask

    def forward(self, input):
        masked_weight=self.weight*self.mask.reshape((-1,1,1,1))
        masked_weight=self.weight*self.channel_weight                 #add weight for each channel of the filter






if __name__ == "__main__":
    a=vgg16_bn_paper(dataset_name='cifar10',pretrained=False)
    from framework import evaluate,data_loader
    dl=data_loader.create_validation_loader(batch_size=16,num_workers=2,dataset_name='cifar10')
    evaluate.evaluate_net(a,dl,save_net=False)
