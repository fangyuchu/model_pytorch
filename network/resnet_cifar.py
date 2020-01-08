'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict
from framework import data_loader, evaluate

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.planes = planes  # changed
        self.relu2 = nn.ReLU(True)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.downsample = LambdaLayer(self.tmp_func)
                # self.shortcut = LambdaLayer(lambda x:
                #                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def tmp_func(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes // 4, self.planes // 4), "constant", 0)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers=OrderedDict()
        block_ind=0
        for stride in strides:
            layers['block'+str(block_ind)]=block(self.in_planes, planes, stride)
            self.in_planes = planes * block.expansion
            block_ind+=1
        # layers = []
        # for stride in strides:
        #     layers.append(block(self.in_planes, planes, stride))
        #     self.in_planes = planes * block.expansion

        return nn.Sequential(layers)

    def forward(self, x):
        out = self.relu1((self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9],num_classes=num_classes)


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


# def checkpoint_conversion(path):
#     '''
#     convert checkpoint downloaded to fit network which does not support parallel training.
#     :param path:
#     :return:
#     '''
#     c_original = torch.load(path)
#     new_state_dict = OrderedDict()
#     for k, v in c_original['state_dict'].items():
#         if 'module.' not in k:
#             return
#         name = k[7:]  # remove module.
#         new_state_dict[name] = v
#
#     c_new = {'highest_accuracy': c_original['best_prec1'],
#              'state_dict': new_state_dict}
#
#     torch.save(c_new, path)
#
# def checkpoint_conversion_with_block(path,net):
#     '''
#     convert checkpoint downloaded to fit network which does not support parallel training.
#     :param path:
#     :return:
#     '''
#     c_original = torch.load(path)
#     new_state_dict = OrderedDict()
#     for k, v in c_original['state_dict'].items():
#         if 'layer'  in k:
#             list_k=list(k)
#             list_k.insert(7,'block')
#             k=''.join(list_k)
#         new_state_dict[k] = v
#
#     c_new = {'highest_accuracy': c_original['highest_accuracy'],
#              'state_dict': new_state_dict}
#
#     torch.save(c_new, path)

if __name__ == "__main__":
    import torch
    from network import storage
    from framework import evaluate,data_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint=torch.load('/home/victorfang/model_pytorch/data/baseline/resnet56_cifar100_0.71580.tar')
    # c_sample=torch.load('/home/victorfang/model_pytorch/data/baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
    net=resnet56(num_classes=200)
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)

    # checkpoint.update(storage.get_net_information(net=net,dataset_name='tiny_imagenet',net_name='resnet18'))
    # checkpoint.pop('net')
    # torch.save(checkpoint,'/home/victorfang/model_pytorch/data/baseline/resnet18_tinyimagenet_v2_0.72990.tar')
    # net=storage.restore_net(checkpoint=checkpoint,pretrained=True)
    # net=nn.DataParallel(net)
    evaluate.evaluate_net(net=net,
                          data_loader=data_loader.create_validation_loader(512,8,'cifar100'),
                          save_net=False,
                          dataset_name='cifar100')

    print()