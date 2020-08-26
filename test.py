import torch
from torch import nn
from framework import data_loader, evaluate,measure_flops,train
from network import vgg_channel_weight, vgg,storage,resnet,net_with_predicted_mask,resnet_cifar,modules,resnet_cifar,modules
from framework import config as conf
import os,sys
from filter_characteristic import filter_feature_extractor,predict_dead_filter
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import cgd
import logger
import copy
#ssh -L 16006:127.0.0.1:6006 -p 20029 victorfang@210.28.133.13
# import torchsnooper
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


net=resnet.resnet50().to(device)
net2=resnet_cifar.resnet56()
# measure_flops.measure_model(net)
print()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#训练参数
optimizer_net = optim.SGD
optimizer_extractor = optim.SGD
learning_rate = {'default': 0.1, 'extractor': 0.001}
weight_decay = {'default':1e-4,'extractor':1e-4}
momentum = {'default':0.9,'extractor':0.9}
batch_size=256
#网络参数
add_shortcut_ratio=0.9#不是这儿！！！
mask_update_freq = 1000
mask_update_epochs = 900
mask_training_start_epoch=1
mask_training_stop_epoch=2


exp_name='resnet50'
description=exp_name+'  '+'专门训练mask,没有warmup，训练20epoch'

total_flop=4133641192
prune_ratio=0.95
flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
gradient_clip_value=None
learning_rate_decay_epoch = [mask_training_stop_epoch+1*i for i in [80,120]]
num_epochs=160*1+mask_training_stop_epoch

net=resnet.resnet50(pretrained=False)
net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                       net_name='resnet50',
                                                                       dataset_name='imagenet',
                                                                       mask_update_epochs=mask_update_epochs,
                                                                       mask_update_freq=mask_update_freq,
                                                                       flop_expected=flop_expected,
                                                                       gcn_rounds=2,
                                                                       mask_training_start_epoch=mask_training_start_epoch,
                                                                       mask_training_stop_epoch=mask_training_stop_epoch,
                                                                       batch_size=batch_size,
                                                                       add_shortcut_ratio=add_shortcut_ratio
                                                                       )
# net=resnet.resnet50(pretrained=False)
torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
net=net.to(device)
net = nn.parallel.DistributedDataParallel(net,find_unused_parameters=True)
# eval_loader = data_loader.create_test_loader(batch_size=batch_size, num_workers=8, dataset_name='imagenet')
# evaluate.evaluate_net(net, eval_loader, save_net=False,dataset_name='imagenet')

print( weight_decay, momentum, learning_rate, mask_update_freq, mask_update_epochs, flop_expected, gradient_clip_value)
train.train_extractor_network(net=net,
                              net_name='resnet50',
                              exp_name='test',
                              description=description,
                              dataset_name='imagenet',

                              optim_method_net=optimizer_net,
                              optim_method_extractor=optimizer_extractor,
                              weight_decay=weight_decay,
                              momentum=momentum,
                              learning_rate=learning_rate,

                              num_epochs=num_epochs,
                              batch_size=batch_size,
                              evaluate_step=1000,
                              load_net=False,
                              test_net=False,
                              num_workers=8,
                              # weight_decay=5e-4,
                              learning_rate_decay=True,
                              learning_rate_decay_epoch=learning_rate_decay_epoch,
                              learning_rate_decay_factor=0.1,
                              scheduler_name='MultiStepLR',
                              top_acc=1,
                              data_distributed=True,
                              paint_loss=True,
                              save_at_each_step=False,
                              gradient_clip_value=gradient_clip_value
                              )










optimizer_net = optim.SGD
# optimizer_extractor = optim.Adam
optimizer_extractor=optim.SGD
learning_rate = {'default': 0.1, 'extractor': 0.001}
momentum = {'default':0.9,'extractor':0.9}
weight_decay = {'default':1e-4,'extractor':5e-4}
# optimizer = optim.Adam
# learning_rate = {'default': 0.01, 'extractor': 0.01}
exp_name='test2'

mask_update_freq = 1000
mask_update_epochs = 900
batch_size=1024
mask_training_start_epoch=1
mask_training_stop_epoch=20

num_epochs=160*2+mask_training_stop_epoch
learning_rate_decay_epoch = [mask_training_stop_epoch+2*i for i in [80,120]]

gradient_clip_value=1

save_at_each_step=True

# learning_rate_decay_epoch = [mask_training_stop_epoch+2*i for i in [80,120]]
# num_epochs=160*2+mask_training_stop_epoch


# net=resnet_cifar.resnet56(num_classes=10).to(device)
# # net=resnet.resnet50(pretrained=False).to(device)
# # total_flop=4111413224
# total_flop=125485706
# flop_expected=total_flop*0.5
#
# net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
#                                                                        net_name='resnet56',
#                                                                        dataset_name='cifar10',
#                                                                        mask_update_epochs=mask_update_epochs,
#                                                                        mask_update_freq=mask_update_freq,
#                                                                        flop_expected=flop_expected,
#                                                                        gcn_rounds=2,
#                                                                        mask_training_start_epoch=mask_training_start_epoch,
#                                                                        mask_training_stop_epoch=mask_training_stop_epoch,
#                                                                        batch_size=batch_size,
#                                                                        add_shortcut_ratio=0.9,
#                                                                        )
# net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
#                                                                        net_name='resnet50',
#                                                                        dataset_name='imagenet',
#                                                                        mask_update_epochs=mask_update_epochs,
#                                                                        mask_update_freq=mask_update_freq,
#                                                                        flop_expected=flop_expected,
#                                                                        gcn_rounds=2,
#                                                                        mask_training_start_epoch=mask_training_start_epoch,
#                                                                        mask_training_stop_epoch=mask_training_stop_epoch,
#                                                                        batch_size=batch_size,
#                                                                        add_shortcut_ratio=0.9,
#                                                                        )

# net.measure_self_flops()
# net.measure_self_flops()
# c=torch.load('/home/victorfang/model_pytorch/data/masked_net/3.tar')
# net.load_state_dict(c['state_dict'])
# net.update_mask()
# #todo:以上都是对的
# net.mask_net()

path='/home/victorfang/PycharmProjects/model_pytorch/data/model_saved/test2/crash/'
state_dict=torch.load(path+'state_dict.pt')
images=torch.load(path+'images.pt')

net.load_state_dict(state_dict)
net(images)
net.measure_self_flops()

train.train_extractor_network(net=net,
                              net_name='vgg16_bn',
                              exp_name=exp_name,
                              dataset_name='cifar10',

                              optim_method_net=optimizer_net,
                              optim_method_extractor=optimizer_extractor,
                              weight_decay=weight_decay,
                              momentum=momentum,
                              learning_rate=learning_rate,

                              num_epochs=num_epochs,
                              batch_size=batch_size,
                              evaluate_step=5000,
                              load_net=False,
                              test_net=False,
                              num_workers=0,
                              # weight_decay=5e-4,
                              learning_rate_decay=True,
                              learning_rate_decay_epoch=learning_rate_decay_epoch,
                              learning_rate_decay_factor=0.1,
                              scheduler_name='MultiStepLR',
                              top_acc=1,
                              data_distributed=False,
                              paint_loss=True,
                              save_at_each_step=save_at_each_step,
                              gradient_clip_value=gradient_clip_value,

                              )
