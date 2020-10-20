import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
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


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True,dataset_name='imagenet'):
        super(VGG, self).__init__()
        self.features = features
        if dataset_name == 'imagenet':
            self.classifier = nn.Sequential(
                nn.Linear(512*7*7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 1000),
            )
        elif dataset_name == 'cifar10':
            self.classifier = nn.Sequential(
                # nn.Dropout(),
                nn.Linear(512 , 512),
                nn.BatchNorm1d(512,track_running_stats=True),
                # nn.ReLU(True),
                # nn.Dropout(),
                # nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, 10),
            )
        elif dataset_name == 'tiny_imagenet':
            self.classifier = nn.Sequential(
                nn.Linear(512*7*7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 200),
            )
        elif dataset_name == 'cifar100':
            self.classifier = nn.Sequential(
                # nn.Linear(512, 4096),
                # nn.BatchNorm1d(4096,track_running_stats=True),
                # nn.ReLU(True),
                # nn.Linear(4096, 100),

                nn.Linear(512,100)
            )
        else:
            raise Exception("Please input right dataset_name")
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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)

    # i=0
    # from collections import OrderedDict
    # layers = OrderedDict()
    # in_channels = 3
    # total=len(cfg)
    # for v in cfg:
    #     i+=1
    #     if v == 'M':
    #         layers['MaxPool2d'+str(i)]= nn.MaxPool2d(kernel_size=2, stride=2)
    #     else:
    #         conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
    #         if batch_norm:
    #             layers['conv2d'+str(i)]=conv2d
    #             layers['BatchNorm2d'+str(i)]=nn.BatchNorm2d(v)
    #             if i!=17:
    #                 layers['ReLU'+str(i)]=nn.ReLU(inplace=True)
    #             else:
    #                 layers['ReLU' + str(i)] = nn.ReLU(inplace=False)
    #         else:
    #             layers['conv2d' + str(i)] = conv2d
    #             layers['ReLU' + str(i)] = nn.ReLU(inplace=True)
    #         in_channels = v
    return nn.Sequential(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model


# if __name__ == "__main__":
#     import torch
#     from network import storage
#     from framework import data_loader,evaluate
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16_cifar100_extractor_static/checkpoint/flop=194985856,accuracy=0.71950.tar')
#     c_sample=torch.load('/home/victorfang/model_pytorch/data/baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
#
#     # net=checkpoint['net']
#     # # net=vgg16(dataset_name='cifar100').to(device)
#     # net.load_state_dict(checkpoint['state_dict'])
#     # checkpoint.pop('state_dict')
#     # checkpoint['state_dict']=net.state_dict()
#     # checkpoint.update(storage.get_net_information(net=net,dataset_name='cifar100',net_name='vgg16'))
#     # # checkpoint.pop('net')
#     # torch.save(checkpoint,'/home/victorfang/model_pytorch/data/model_saved/vgg16_cifar100_extractor_static/checkpoint/flop=194985856,accuracy=0.71950.tar')
#     net=storage.restore_net(checkpoint=checkpoint,pretrained=True)
#
#
#     # net=nn.DataParallel(net)
#     evaluate.evaluate_net(net=net,
#                           data_loader=data_loader.create_test_loader(512,8,'cifar100'),
#                           save_net=False,
#                           dataset_name='cifar100')
#
#     print()
