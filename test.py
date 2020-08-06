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
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# net=resnet.resnet50().to(device)
net=resnet_cifar.resnet56()#,vgg.vgg16_bn()
num=0
for name,mod in net.named_modules():
    if isinstance(mod,nn.Conv2d):
        num+=mod.out_channels
print(num)
print()
#
#
# measure_flops.measure_model(net)
# net = net_with_predicted_mask.predicted_mask_and_shortcut_net(net,
#                                                                       net_name='resnet50',
#                                                                       dataset_name='imagenet',
#                                                                       mask_update_epochs=1,
#                                                                       mask_update_freq=3,
#                                                                       flop_expected=4133641192*0.5,
#                                                                       gcn_rounds=2,
#                                                                       mask_training_start_epoch=5,
#                                                                       mask_training_stop_epoch=35,
#                                                                       batch_size=512
#                                                                       )
# num_conv=0
# num_s=0
# for name,mod in net.named_modules():
#     if isinstance(mod,modules.block_with_mask_shortcut):
#         if len(mod.downsample)!=0:
#             num_conv+=1
#         else:
#             num_s+=1
#
# print(num_conv,num_s)
# measure_flops.measure_model(net)
# net=resnet.resnet50(pretrained=False).cuda()
# measure_flops.measure_model(net=net,dataset_name='imagenet',print_flop=True)
# print()

optimizer_net = optim.SGD
# optimizer_extractor = optim.Adam
optimizer_extractor=optim.SGD
learning_rate = {'default': 0.1, 'extractor': 0.001}
momentum = {'default':0.9,'extractor':0.9}
weight_decay = {'default':1e-4,'extractor':5e-4}
# optimizer = optim.Adam
# learning_rate = {'default': 0.01, 'extractor': 0.01}
exp_name='tmp'

mask_update_freq = 1000
mask_update_epochs = 900
batch_size=1024
mask_training_start_epoch=1
mask_training_stop_epoch=4

num_epochs=160*2+mask_training_stop_epoch
learning_rate_decay_epoch = [mask_training_stop_epoch+2*i for i in [80,120]]

flop_expected=0.6e7#1.25e7#2.5e7
# flop_expected = 126550666
gradient_clip_value=1

save_at_each_step=False

# learning_rate_decay_epoch = [mask_training_stop_epoch+2*i for i in [80,120]]
# num_epochs=160*2+mask_training_stop_epoch


net=resnet_cifar.resnet56(num_classes=10).to(device)
net=resnet.resnet50(pretrained=False).to(device)
total_flop=4111413224
flop_expected=total_flop*0.5

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
                                                                       add_shortcut_ratio=0.9,
                                                                       )
net.to(device)
# net.measure_self_flops()
# c=torch.load('/home/victorfang/model_pytorch/data/masked_net/3.tar')
# net.load_state_dict(c['state_dict'])
net.update_mask()
#todo:以上都是对的
net.mask_net()
# net.print_mask()
# net.prune_net()
# f=net.measure_self_flops()
print()
# net.current_epoch=30
# train.add_forward_hook(net,module_name='extractor.gcn.network.0')
# net.update_mask()
# flop=net.measure_self_flops()
# print()

# net_tmp = net_with_predicted_mask.predicted_mask_and_shortcut_net(net,
#                                                               net_name='resnet56',
#                                                               dataset_name='cifar10',
#                                                               mask_update_epochs=mask_update_epochs,
#                                                               mask_update_freq=mask_update_freq,
#                                                               flop_expected=flop_expected,
#                                                               gcn_rounds=2,
#                                                               mask_training_start_epoch=mask_training_start_epoch,
#                                                               mask_training_stop_epoch=mask_training_stop_epoch,
#                                                               batch_size=batch_size,
#                                                               add_shortcut_ratio=0.9
#
#                                                               )
#
#
# net=torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_predicted_mask_and_variable_shortcut_net_seperate_train_2/crash/net.pt')
# state_dict=torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_predicted_mask_and_variable_shortcut_net_seperate_train_2/crash/state_dict.pt')
# net.load_state_dict(state_dict)

# checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_predicted_mask_and_variable_shortcut_net_seperate_train_2/checkpoint/flop=25471882,accuracy=0.56560.tar')
# net=storage.restore_net(checkpoint)
# net.prune_net_tmp()
# net.to(device)


#

# net.update_mask()
# f1=net.measure_self_flops()
# net.prune_net()
# f2=net.measure_self_flops()
# net.print_mask()
# print()
# #
# net.prune_net()
# check=net.state_dict()
# for k,v in check.items():
#     if v.device.type != 'cuda':
#         print()
#
# net.current_epoch=1000
# train.add_forward_hook(net,module_name='net.conv_1_3x3')
# images=torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_predicted_mask_and_variable_shortcut_net_seperate_train_3/crash/images.pt')
# labels=torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_predicted_mask_and_variable_shortcut_net_seperate_train_3/crash/labels.pt')
# orgin_outputs=torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_predicted_mask_and_shortcut_net_prune70_4/crash/outputs.pt')
# images, labels = images.to(device), labels.to(device)
# outputs = net(images)
# criterion=nn.CrossEntropyLoss()
# loss = criterion(outputs, labels)
# loss.backward()
# train.add_forward_hook(net,module_name='stage')
# train.add_forward_hook(net,module_name='net.conv_1_3x3')
# for mod in net.modules():
#     if isinstance(mod,nn.BatchNorm2d):
#         mod.track_running_stats=False
# net.train()
# net.current_epoch=1000
# # net.prune_zero_block()
# output=net(images)
# print()
# net=resnet_cifar.resnet56().cuda()

net.to(device)
# train.train_extractor_network(net=net,
#             net_name='resnet56',
#             exp_name=exp_name,
#             description='test',
#             dataset_name='cifar10',
#
#             optimizer=optimizer,
#             weight_decay=weight_decay,
#             momentum=momentum,
#             learning_rate=learning_rate,
#
#             num_epochs=320,
#             batch_size=batch_size,
#             evaluate_step=5000,
#             load_net=False,
#             test_net=False,
#             num_workers=0,
#             # weight_decay=5e-4,
#             learning_rate_decay=True,
#             learning_rate_decay_epoch=[160,240],
#             learning_rate_decay_factor=0.1,
#             scheduler_name='MultiStepLR',
#             top_acc=1,
#             data_parallel=False,
#             paint_loss=True,
#             save_at_each_step=True,
#             gradient_clip_value=gradient_clip_value
#             )



# checkpoint = torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_predicted_mask_shortcut_with_weight_0.5blpenalty_prune40_bs512_4/checkpoint/flop=81012362,accuracy=0.91480.tar')
# net.load_state_dict(checkpoint['state_dict'])
# net=net.to(device)
# net.detach_mask()
# evaluate.evaluate_net(net,data_loader=data_loader.create_test_loader(batch_size=512,num_workers=0,dataset_name='cifar10'),save_net=False)
#
# checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
# # save the output to log
# print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
# if not os.path.exists(checkpoint_path):
#     os.makedirs(checkpoint_path, exist_ok=True)
# sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
# sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
#
# print(optimizer, weight_decay, momentum, learning_rate, mask_update_freq, mask_update_epochs, flop_expected, gradient_clip_value)
#
# # train.add_forward_hook(net,module_name='net.features.1')
#
train.train_extractor_network(net=net,
                              net_name='resnet56',
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
                              data_parallel=False,
                              paint_loss=True,
                              save_at_each_step=save_at_each_step,
                              gradient_clip_value=gradient_clip_value
                              )
