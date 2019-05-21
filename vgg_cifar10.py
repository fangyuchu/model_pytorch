import train
import vgg
import torch.nn as nn
import os
import torch

net=vgg.vgg16_bn(pretrained=False)
net.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

checkpoint_path='./model_best.pth.tar'
if os.path.isfile(checkpoint_path):
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    net.load_state_dict(checkpoint['state_dict'])
    print("loaded checkpoint")
else:
    print("=> no checkpoint found at '{}'".format(checkpoint_path))
