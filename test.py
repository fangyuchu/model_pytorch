import torch
from torch import nn
from framework import data_loader, evaluate,measure_flops,train
from network import vgg_channel_weight, vgg,storage,resnet,net_with_predicted_mask,resnet_cifar,modules,resnet_cifar,modules
from framework import config as conf
import os,sys
from filter_characteristic import filter_feature_extractor,predict_dead_filter
import numpy as np
from torch import optim
import math
from prune import prune_module
import matplotlib.pyplot as plt
import logger
import copy
#ssh -L 16006:127.0.0.1:6006 -p 20029 victorfang@210.28.133.13
# import torchsnooper
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class resnet(nn.Module):
    def __init__(self, depth=164, dataset='cifar10', cfg=None):
        super(resnet, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9
        block = Bottleneck

        if cfg is None:
            # Construct config variable.
            cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n, cfg = cfg[0:3*n])
        self.layer2 = self._make_layer(block, 32, n, cfg = cfg[3*n:6*n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg = cfg[6*n:9*n], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg[-1], 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg[-1], 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


net=resnet()
print()




torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net=resnet_cifar.resnet56().cuda()
measure_flops.measure_model(net,'cifar10')

# net=vgg.vgg16_bn(dataset_name='imagenet').cuda()
net=resnet.resnet50().cuda()
measure_flops.measure_model(net,'imagenet')

optimizer_net = optim.SGD
optimizer_extractor = optim.SGD
learning_rate = {'default': 0.1, 'extractor': 0.001}
weight_decay = {'default': 1e-4, 'extractor': 5e-4}
momentum = {'default': 0.9, 'extractor': 0.9}
batch_size = 128
# 网络参数
add_shortcut_ratio = 0.9  # 不是这儿！！！
mask_update_freq = 1000
mask_update_epochs = 900
mask_training_start_epoch = 1
mask_training_stop_epoch = 80

exp_name = 'resnet56_predicted_mask_and_variable_shortcut_net_mask_newinner_80epoch_7'
description = exp_name + '  ' + '专门训练mask,没有warmup，训练20epoch'

total_flop = 126550666
prune_ratio = 0.9
flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
gradient_clip_value = None
learning_rate_decay_epoch = [mask_training_stop_epoch + 1 * i for i in [80, 120]]
num_epochs = 160 * 1 + mask_training_stop_epoch
#
net = resnet_cifar.resnet56(num_classes=10).to(device)
net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                       net_name='resnet56',
                                                                       dataset_name='cifar10',
                                                                       mask_update_epochs=mask_update_epochs,
                                                                       mask_update_freq=mask_update_freq,
                                                                       flop_expected=flop_expected,
                                                                       gcn_rounds=2,
                                                                       mask_training_start_epoch=mask_training_start_epoch,
                                                                       mask_training_stop_epoch=mask_training_stop_epoch,
                                                                       batch_size=batch_size,
                                                                       add_shortcut_ratio=add_shortcut_ratio
                                                                       )
net = net.to(device)

i = 4
checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)


checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet56', str(i) + '.tar'), map_location='cpu')
net.load_state_dict(checkpoint['state_dict'])
net.mask_net()
net.print_mask()
net.prune_net()
net.print_in_out_channels()
net.current_epoch = net.mask_training_stop_epoch + 1
# measure_flops.measure_model(net, dataset_name='cifar10', print_flop=True)
measure_flops.measure_model(net.net,dataset_name='cifar10',print_flop=True)
print()
