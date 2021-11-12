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
from PIL import Image
#ssh -L 16006:127.0.0.1:6006 -p 20029 victorfang@210.28.133.13
# import torchsnooper
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

net = mobilenet.MobileNetV2(n_class=1000)
net.cuda()
measure_flops.measure_model(net)
print()

# checkpoint = torch.load('/home/victorfang/model_pytorch/data/baseline/resnet56_cifar10,accuracy=0.94230.tar')
# # checkpoint = torch.load('/home/victorfang/semantic_adversarial_nas/data/model_saved/finetune_withclean_gaussian_noise_tenth_fullnet_lr0.0001_backbone_BasicBlock_conv3x3BN_64outChannels_56depth_flop=128516746_acc=0.9391_sgd_noaugmix_cosinedecay/checkpoint/flop=128516746,accuracy=0.85693.pth')
# # checkpoint = torch.load('/home/disk_new/semantic_adversarial_nas/model_saved_8gpu/model_saved/finetune_withclean_gaussian_noise_tenth_fullnet_lr0.0001_backbone_BasicBlock_conv3x3BN_64outChannels_56depth_flop=128516746_acc=0.9391_sgd_noaugmix_cosinedecay/checkpoint/flop=128516746,accuracy=0.85693.pth')
# # net = checkpoint['net'].cuda()
# net = resnet_cifar.resnet56()
# net.load_state_dict(checkpoint['state_dict'])
# test_loader=data_loader.create_test_loader(512,0,'cifar10')
# evaluate.evaluate_net(net, test_loader, False)
#
# net=resnet.resnet50()
# measure_flops.count_params(net)
# # net2=resnet_cifar.resnet56()
# checkpoint = torch.load('/home/victorfang/PycharmProjects/model_pytorch/data/model_saved/gat_vgg16bn_cifar100_predicted_mask_and_variable_shortcut_net_newinner_doubleschedule_98_5/checkpoint/flop=7285428,accuracy=0.54290.tar')
# net = checkpoint['net'].cuda()
# measure_flops.count_params(net)
# net2 = vgg.vgg16_bn(dataset_name='cifar100')
# net2 = mobilenet.mobilenet_v1(num_class=1000)
# measure_flops.count_params(net2)
#
# evaluate.evaluate_net(net,data_loader.create_test_loader(batch_size=1,num_workers=4,dataset_name='cifar10'),save_net=False)
# print()
# net = resnet_cifar.resnet56()
# net=resnet.resnet18(num_classes=10).cuda()
# measure_flops.measure_model(net,'cifar10')
#
# add_shortcut_ratio = 0.9  # 不是这儿！！！
# mask_update_freq = 1000
# mask_update_epochs = 900
# mask_training_start_epoch = 1
# # mask_training_stop_epoch=20
# mask_training_stop_epoch = 161
# total_flop = 126550666  # 125485706
# prune_ratio = 0.9
# flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
# gradient_clip_value = 5
# batch_size=128
# # learning_rate_decay_epoch = [mask_training_stop_epoch+1*i for i in [80,120]]
# # num_epochs=160*1+mask_training_stop_epoch
# #
# net = resnet_cifar.resnet56(num_classes=10).cuda()
# net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
#                                                                        net_name='resnet56',
#                                                                        dataset_name='cifar10',
#                                                                        mask_update_epochs=mask_update_epochs,
#                                                                        mask_update_freq=mask_update_freq,
#                                                                        flop_expected=flop_expected,
#                                                                        mask_training_start_epoch=mask_training_start_epoch,
#                                                                        mask_training_stop_epoch=mask_training_stop_epoch,
#                                                                        batch_size=batch_size,
#                                                                        add_shortcut_ratio=add_shortcut_ratio,
#                                                                        gcn_layer_num=2
#                                                                        )
#
# net = net.cuda()
# checkpoint = torch.load('/home/victorfang/PycharmProjects/model_pytorch/data/model_saved/gat_resnet56_predicted_mask_and_variable_shortcut_net_mask_newinner_bn_meanstd_crosstrain_22/19/checkpoint/final_model_flop=126550666,accuracy=0.86970.pth')
# net.load_state_dict(checkpoint['state_dict'])
#
# train.train(net=net.net,
#             net_name='resnet56',
#             exp_name='gat_resnet56_crosstrain_convNet_finetune',
#             # description=description+'gat_extractor',
#             dataset_name='cifar10',
#             optimizer=optim.SGD,
#             weight_decay=5e-4,
#             momentum=0.9,
#             learning_rate=0.01,
#             num_epochs=50,
#             batch_size=batch_size,
#             evaluate_step=5000,
#             resume=True,
#             test_net=True,
#             num_workers=4,
#             learning_rate_decay=True,
#             learning_rate_decay_epoch=[10],
#             learning_rate_decay_factor=0.1,
#             scheduler_name='MultiStepLR',
#             top_acc=1,
#             data_parallel=False,
#             paint_loss=True,
#             save_at_each_step=False,
#             gradient_clip_value=gradient_clip_value,
#             )
print()
# net=mobilenet.mobilenet_v1(num_class=1000).cuda()
# measure_flops.measure_model(net)
# extractor=filter_feature_extractor.extractor(net,feature_len=9).cuda()
# extractor.forward(net)
# # net=resnet_cifar.resnet56(num_classes=100)
# a=np.ndarray([0,1,0])
# b=torch.tensor([0,0,0])
# b[a>0]=1
# print()
# net=mobilenet.mobilenet_v1(num_class=1000).cuda()
# measure_flops.measure_model(net,)
#
#
# optimizer_net = optim.SGD
# optimizer_extractor = optim.SGD
# learning_rate = {'default': 0.1, 'extractor': 0.001}
# weight_decay = {'default': 5e-4, 'extractor': 5e-4}
# momentum = {'default': 0.9, 'extractor': 0.9}
# batch_size = 128
# # 网络参数
# add_shortcut_ratio = 0.9  # 不是这儿！！！
# mask_update_freq = 1000
# mask_update_epochs = 900
# mask_training_start_epoch = 1
# mask_training_stop_epoch = 80
#
# net=mobilenet.mobilenet_v1(num_class=1000)
# # dl=data_loader.create_train_loader(batch_size=10,num_workers=1)
# # for (i,data) in enumerate(data_loader):
# net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
#                                                                        net_name='mobilenet_v1',
#                                                                        dataset_name='imagenet',
#                                                                        mask_update_epochs=mask_update_epochs,
#                                                                        mask_update_freq=mask_update_freq,
#                                                                        flop_expected=578826728*0.1,
#                                                                        gcn_rounds=2,
#                                                                        mask_training_start_epoch=mask_training_start_epoch,
#                                                                        mask_training_stop_epoch=mask_training_stop_epoch,
#                                                                        batch_size=batch_size,
#                                                                        add_shortcut_ratio=add_shortcut_ratio,
#                                                                        feature_len=9,
#                                                                        )
# net.cuda()
# measure_flops.measure_model(net)
# net.mask_net()
# net.print_mask()
# net.prune_net()
print()
# total_flop=126550666#125485706
# prune_ratio=0.93
# flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
# gradient_clip_value=None
# learning_rate_decay_epoch = [mask_training_stop_epoch+1*i for i in [80,120]]
# num_epochs=160*1+mask_training_stop_epoch
#
# #
# net=resnet_cifar.resnet56(num_classes=10).cuda()
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
#                                                                        add_shortcut_ratio=add_shortcut_ratio
#                                                                        )
# net=net.cuda()
# i = 4
#
# checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet56',str(i) + '.tar'),map_location='cpu')
# net.load_state_dict(checkpoint['state_dict'])
# net.mask_net()
# net.print_mask()
# net.prune_net()
# net.current_epoch = net.mask_training_stop_epoch + 1
# net.eval()
# dl=data_loader.create_test_loader(batch_size=1,num_workers=0,dataset_name='cifar10',shuffle=True)




# optimizer_net = optim.SGD
# learning_rate = {'default': 0.1, 'extractor': 0.0001}
# weight_decay = {'default': 1e-4, 'extractor': 1e-4}
# momentum = {'default': 0.9, 'extractor': 0.9}
# batch_size = 256
# # 网络参数
# add_shortcut_ratio = 0.9  # 不是这儿！！！
# mask_update_freq = 1000
# mask_update_epochs = 900
# mask_training_start_epoch = 1
# mask_training_stop_epoch = 3
#
# exp_name = 'resnet50_predicted_mask_and_variable_shortcut_net_mask_newinner_5'
# description = exp_name + '  ' + '专门训练mask,没有warmup，训练20epoch'
#
# total_flop = 4133641192
# prune_ratio = 0.5
# flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
# gradient_clip_value = None
# learning_rate_decay_epoch = [mask_training_stop_epoch + 1 * i for i in [30, 60]]
# num_epochs = 90 * 1 + mask_training_stop_epoch
#
# net = resnet.resnet50(pretrained=False)
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
#                                                                        add_shortcut_ratio=add_shortcut_ratio
#                                                                        )
#
# net = net.cuda()
#
# i = 6
#
# checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet50', str(i) + '.tar'),
#                         map_location='cpu')
# net.load_state_dict(checkpoint['state_dict'])
#
#
# dl=data_loader.create_test_loader(batch_size=1,num_workers=0,dataset_name='imagenet')



# train.show_feature_map(net,dl,0,16)


