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
from sklearn import manifold
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# net=resnet_cifar.resnet56(num_classes=100)






net=vgg.vgg16_bn(dataset_name='cifar10').cuda()
net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                       net_name='vgg16_bn',
                                                                       dataset_name='cifar10',
                                                                       mask_update_epochs=1,
                                                                       mask_update_freq=20,
                                                                       flop_expected=3.13e6*0.1,
                                                                       gcn_layer_num=2,
                                                                       mask_training_start_epoch=1,
                                                                       mask_training_stop_epoch=20,
                                                                       batch_size=128,
                                                                       add_shortcut_ratio=0.9
                                                                       )
net=net.cuda()





checkpoint = torch.load('/home/victorfang/model_pytorch/data/model_saved/gat_vgg16bn_predicted_mask_and_variable_shortcut_net_mask_newinner_3/checkpoint/flop=314570250,accuracy=0.41610.tar',map_location='cpu')
net.track_running_stats(False)
net.load_state_dict(checkpoint['state_dict'])
net.mask_net()
net.print_mask()
net.prune_net()
        # net.current_epoch = net.mask_training_stop_epoch + 1
        # learning_rate_decay_epoch = [1*i for i in [80,120]]
        # num_epochs = 160*1
        # train.train(net=net,
        #             net_name='vgg16_bn',
        #             exp_name=exp_name,
        #             description=description,
        #             dataset_name='cifar10',
        #             optimizer=optim.SGD,
        #             weight_decay=weight_decay,
        #             momentum=momentum,
        #             learning_rate=learning_rate,
        #             num_epochs=num_epochs,
        #             batch_size=batch_size,
        #             evaluate_step=5000,
        #             resume=False,
        #             test_net=False,
        #             num_workers=2,
        #             learning_rate_decay=True,
        #             learning_rate_decay_epoch=learning_rate_decay_epoch,
        #             learning_rate_decay_factor=0.1,
        #             scheduler_name='MultiStepLR',
        #             top_acc=1,
        #             data_parallel=False,
        #             paint_loss=True,
        #             save_at_each_step=False,
        #             gradient_clip_value=gradient_clip_value
        #             )


net.t_sne()


print()

