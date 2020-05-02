import torch
from torch import nn
from framework import data_loader, train, evaluate,measure_flops
from network import vgg_channel_weight, vgg,storage,resnet,net_with_predicted_mask,resnet_cifar
from framework import config as conf
import os,sys
from filter_characteristic import filter_feature_extractor,predict_dead_filter
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import cgd
#ssh -L 16006:127.0.0.1:6006 -p 20029 victorfang@210.28.133.13
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
# a=resnet_cifar.resnet56().to(device)
# # train.add_forward_hook(torch.nn.Conv2d,a)
# b=resnet.resnet50().to(device)
# # train.add_forward_hook(torch.nn.Conv2d,b)
# c=vgg.vgg16_bn(dataset_name='cifar10').to(device)
# # train.add_forward_hook(torch.nn.Conv2d,c)
# # train.add_forward_hook(nn.BatchNorm1d,c)
# # net=net_with_mask_shortcut.mask_and_shortcut_net(c,'vgg16_bn','cifar10').to(device)
# net=net_with_mask_shortcut.mask_and_shortcut_net(a,'resnet56','cifar10').to(device)
# # net.load_state_dict(torch.load('/home/victorfang/model_pytorch/data/model_saved/tmp/checkpoint/flop=261822474,accuracy=0.92380.tar')['state_dict'])
# net=a
# net.to(device)
# train.train(net=net,
#             net_name='resnet56',
#             exp_name='resnet56_baseline_newversion',
#             dataset_name='cifar10',
#             # optimizer=cgd.CGD,
#             optimizer=optim.SGD,
#             # weight_decay={'default':1e-6,'extractor':0},
#             # momentum={'default':0.9,'extractor':0},
#             weight_decay=1e-4,
#             momentum=0.9,
#
#             learning_rate=0.1,
#             num_epochs=160,
#             batch_size=128,
#             evaluate_step=5000,
#             load_net=False,
#             test_net=True,
#             num_workers=8,
#             # weight_decay=5e-4,
#             learning_rate_decay=True,
#             learning_rate_decay_epoch=[80,120],
#             learning_rate_decay_factor=0.1,
#             scheduler_name='MultiStepLR',
#             top_acc=1,
#             data_parallel=False,
#             paint_loss=False,
#             )
# print()



# net=vgg.vgg16_bn(dataset_name='cifar10')
# b=net.parameters()
#
# a=nn.Parameter(torch.ones(2))
# c=nn.Parameter(torch.Tensor([1,2]))
# opt=optim.SGD([a,c],lr=0.1)
#
# b=a+c
# b=torch.sum(b)
# b.backward()
# opt.step()
# print()

















mask_update_freq = 4000
mask_update_steps = 2
net=vgg.vgg16_bn(dataset_name='cifar10')
net = net_with_predicted_mask.predicted_mask_and_shortcut_net(net,
                                                 net_name='vgg16_bn',
                                                 dataset_name='cifar10',
                                                 mask_update_steps=mask_update_steps,
                                                 mask_update_freq=mask_update_freq,
                                                              flop_expected=5e7)
checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16bn_mask_shortcut_2/checkpoint/flop=261559818,accuracy=0.93360.tar')
# checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16bn_mask_shortcut_3_flop5/checkpoint/flop=49822218,accuracy=0.10000.tar')
net.load_state_dict(checkpoint['state_dict'])
net=net.to(device)

measure_flops.measure_model(net,'cifar10')
print()


# net.eval()

# evaluate.evaluate_net(net,data_loader=data_loader.create_train_loader(batch_size=1024,num_workers=2,dataset_name='cifar10'),save_net=False)
# train_loader=data_loader.create_validation_loader(batch_size=1024,num_workers=2,dataset_name='cifar10')
# for step, data in enumerate(train_loader, 0):
#     images, labels = data
#     images, labels = images.to(device), labels.to(device)
#     net(images)
#     break


# net.train()
# #
# import cgd
#
# net=vgg.vgg16_bn(dataset_name='cifar10').to(device)
train.train(net=net,
            net_name='vgg16_bn',
            exp_name='test',
            dataset_name='cifar10',
            # optimizer=cgd.CGD,
            optimizer=optim.SGD,
            weight_decay={'default':1e-6,'extractor':0},
            momentum={'default':0.9,'extractor':0},
            # weight_decay=0,
            # momentum=0,

            learning_rate={'default': 0.1, 'extractor': 0.1},
            num_epochs=1000000,
            batch_size=128,
            evaluate_step=5000,
            load_net=False,
            test_net=False,
            num_workers=8,
            # weight_decay=5e-4,
            learning_rate_decay=False,
            learning_rate_decay_epoch=[100,250],
            learning_rate_decay_factor=0.1,
            scheduler_name='MultiStepLR',
            top_acc=1,
            data_parallel=False,
            paint_loss=True,
            )