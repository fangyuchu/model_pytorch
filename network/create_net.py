import torch
import torch.nn as nn
import re
import datetime
from network import vgg

def vgg_cifar10(net_name='vgg16_bn',pretrained=True,model_path='../data/baseline/vgg16_bn_cifar10,accuracy=0.941.tar'):
    if pretrained:
        checkpoint = torch.load(model_path)
        net = checkpoint['net']
        net.load_state_dict(checkpoint['state_dict'])
    else:
        temp = re.search(r'(\d+)', net_name).span()[0]
        net = net_name[:temp]  # name of the network.ex: vgg16,vgg11_bn...

        # define the network
        net = getattr(globals()[net], net_name)(pretrained=False)
        net.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        net = net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return net

# def vgg_tiny_imagenet(net_name='vgg16_bn',pretrained=True,model_path='../data/baseline/vgg16_bn_cifar10,accuracy=0.941.tar'):
#     # if pretrained:
#     #     checkpoint = torch.load(model_path)
#     #     network = checkpoint['net']
#     #     network.load_state_dict(checkpoint['state_dict'])
#     # else:
#     temp = re.search(r'(\d+)', net_name).span()[0]
#     network = net_name[:temp]  # name of the network.ex: vgg16,vgg11_bn...
#
#     # define the network
#     network = getattr(globals()[network], net_name)(pretrained=False)
#     network.classifier = nn.Sequential(
#         nn.Dropout(),
#         nn.Linear(512, 512),
#         nn.ReLU(True),
#         nn.Dropout(),
#         nn.Linear(512, 512),
#         nn.ReLU(True),
#         nn.Linear(512, 200),
#     )
#     for m in network.modules():
#         if isinstance(m, nn.Linear):
#             nn.init.normal_(m.weight, 0, 0.01)
#             nn.init.constant_(m.bias, 0)
#     network = network.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#     return network

def net_on_imagenet(net_name,pretrained):
    temp = re.search(r'(\d+)', net_name).span()[0]
    net = net_name[:temp]  # name of the network.ex: vgg,resnet...

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the network
    net = getattr(globals()[net], net_name)(pretrained=pretrained).to(device)
    print('{} Net {} created,'.format(datetime.now(),net_name),end='')
    if pretrained:
        print('using pretrained weight.')
    else:
        print('initiate weight.')
    return net


