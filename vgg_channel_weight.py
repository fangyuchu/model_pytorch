import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

__all__ = [
    'VGG_weighted_channel', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class conv2d_weighted_channel(nn.modules.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(conv2d_weighted_channel, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.channel_weight=nn.Parameter(torch.ones((out_channels,in_channels,1,1)),requires_grad=True)


    def forward(self, input):
        weighted_weight=self.weight*self.channel_weight                 #add weight for each channel of the filter
        out=F.conv2d(input, weighted_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return out

    def prune_channel_weight(self,percent):
        #set the least percent% channel weights to zero
        weight=self.channel_weight.view(-1).abs()
        split_point=weight.sort(descending=False)[0][int(weight.shape[0]*percent)]
        self.channel_weight[torch.abs(self.channel_weight)<=split_point]=0

    def copy_conv2d(self,conv2d):
        if self.in_channels!=conv2d.in_channels or self.out_channels!=conv2d.out_channels \
                or self.padding!=conv2d.padding or self.stride!=conv2d.stride \
                or self.dilation!=conv2d.dilation or self.groups!=conv2d.groups:
            raise ArithmeticError
        self.weight=conv2d.weight
        self.bias=conv2d.bias



class CrossEntropyLoss_weighted_channel(nn.CrossEntropyLoss):
    def __init__(self, net,penalty=1e-5,weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean',piecewise=1):
        super(CrossEntropyLoss_weighted_channel,self).__init__(
            weight,size_average,ignore_index,reduce,reduction
        )
        self.net=net
        self.penalty=penalty
        self.piecewise=piecewise

    def forward(self, input, target):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        regularization_loss=0
        for mod in self.net.modules():
            if isinstance(mod,conv2d_weighted_channel):
                
                weight = mod.channel_weight.view(-1).abs().detach().to(device)
                weight = weight.sort(descending=False)[0]                                   #sort the weight array
                piecewise_split = [0]
                for i in range(1,self.piecewise):
                    piecewise_split += [weight[i * int(weight.shape[0] /self.piecewise)]]   #find the split point
                piecewise_split += [ weight[weight.shape[0]-1] ]

                penalty=torch.FloatTensor(mod.channel_weight.shape).detach().to(device)
                weight=mod.channel_weight.abs().detach().to(device)
                for i in range(len(piecewise_split)-1):
                    penalty[(weight>=piecewise_split[i]).mul(weight<=piecewise_split[i+1])]=self.penalty/(100**i)

                regularization_loss+=torch.sum(torch.abs(penalty*mod.channel_weight))


                # regularization_loss+=torch.sum(torch.abs(mod.channel_weight))

        loss=self.penalty * regularization_loss + F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
        return loss



class VGG_weighted_channel(nn.Module):

    def __init__(self, features, init_weights=True,dataset='imagenet'):
        super(VGG_weighted_channel, self).__init__()
        self.features = features
        if dataset is 'imagenet':
            self.classifier = nn.Sequential(
                nn.Linear(512*7*7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 1000),
            )
        elif dataset is 'cifar10':
            self.classifier = nn.Sequential(
                nn.Linear(512 , 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 10),
            )



        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def train_channel_weight(self,if_train=True):
        for mod in self.modules():
            if isinstance(mod, conv2d_weighted_channel):
                mod.channel_weight.requires_grad=if_train

    def prune_channel_weight(self,percent):
        i=0
        for mod in self.modules():
            if isinstance(mod,conv2d_weighted_channel):
                mod.prune_channel_weight(percent[i])
                i+=1

def reform_net(mod):
    #change conv2d in net to conv2d_weighted_channel
    for k, v in mod._modules.items():
        reform_net(v)
        if isinstance(v, nn.Conv2d):
            in_channels=v.in_channels
            out_channels=v.out_channels
            stride=v.stride
            kernel_size=v.kernel_size
            padding=v.padding
            dilation=v.dilation
            groups=v.groups
            new_conv=conv2d_weighted_channel(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                                             stride=stride,padding=padding,
                                             dilation=dilation,groups=groups)
            new_conv.copy_conv2d(v)
            mod._modules[k] = new_conv


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = conv2d_weighted_channel(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG_weighted_channel 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_weighted_channel(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG_weighted_channel 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_weighted_channel(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG_weighted_channel 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_weighted_channel(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG_weighted_channel 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_weighted_channel(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG_weighted_channel 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_weighted_channel(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG_weighted_channel 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_weighted_channel(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG_weighted_channel 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_weighted_channel(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG_weighted_channel 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_weighted_channel(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model