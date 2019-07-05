import torch
import train
import config as conf
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import math
import prune_and_train
import measure_flops
import evaluate
import numpy as np
import data_loader
from sklearn import svm
import vgg


a=torch.normal(mean=torch.Tensor([0.485, 0.456, 0.406]),std=torch.Tensor([0.229, 0.224, 0.225]))
print(a)


# net = vgg.vgg16_bn(pretrained=True)
# net.classifier = nn.Sequential(
#     nn.Dropout(),
#     nn.Linear(512, 512),
#     nn.ReLU(True),
#     nn.Dropout(),
#     nn.Linear(512, 512),
#     nn.ReLU(True),
#     nn.Linear(512, 10),
# )
# for m in net.modules():
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, 0, 0.01)
#         nn.init.constant_(m.bias, 0)
# net = net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#
# # checkpoint = torch.load('/home/victorfang/Desktop/vgg16_bn_cifar10,accuracy=0.941.tar')
# checkpoint = torch.load(
#     '/home/victorfang/Desktop/vgg16_bn_cifar10,accuracy=0.941.tar')
#
# net = checkpoint['net']
# net.load_state_dict(checkpoint['state_dict'])
# print(checkpoint['highest_accuracy'])
# relu_list,neural_list=evaluate.check_ReLU_alive(net=net,
#                           data_loader=data_loader.create_validation_loader(dataset_path=conf.cifar10['validation_set_path'],
#                                                                  default_image_size=32,
#                                                                  mean=conf.cifar10['mean'],
#                                                                  std=conf.cifar10['std'],
#                                                                  batch_size=1024,
#                                                                  num_workers=4,
#                                                                  dataset_name='cifar10',
#                                                                  ),
#                           neural_dead_times=8000)










# print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# checkpoint=torch.load('/home/victorfang/Desktop/vgg19_imagenet_deadReLU.tar')
# relu_list=checkpoint['relu_list']
# neural_list=checkpoint['neural_list']
# net=checkpoint['net']
#
#
# evaluate.plot_dead_filter_statistics(net,relu_list,neural_list,40000,1)
# print()


