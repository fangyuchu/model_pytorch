import train
import vgg
import torch.nn as nn
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import config as conf
import torch.optim as optim
import logger
import sys
import resnet_copied
import data_loader
import measure_flops
import prune_and_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoint=torch.load('./baseline/resnet56_cifar100_0.71580.tar')
# checkpoint=torch.load('/home/zengyao/fang/model_pytorch/model_saved/resnet56_cifar100_regressor3/checkpoint/flop=90655076,accuracy=0.71000.tar')
# net=checkpoint['net']
# net.load_state_dict(checkpoint['state_dict'])
# net.to(device)
# measure_flops.measure_model(net,'cifar100')
# prune_and_train.prune_inactive_neural_with_regressor_resnet(net=net,
#                                                             net_name='resnet56_cifar100_regressor3',
#                                                             prune_rate=0.05,
#                                                             load_regressor=True,
#                                                             dataset_name='cifar100',
#                                                             filter_preserve_ratio=0.15,
#                                                             max_filters_pruned_for_one_time=0.2,
#                                                             target_accuracy=0.708,
#                                                             tar_acc_gradual_decent=True,
#                                                             flop_expected=4e7,
#                                                             batch_size=128,
#                                                             num_epoch=250,
#                                                             checkpoint_step=3000,
#                                                             use_random_data=False,
#                                                             round_for_train=3,
#                                                             round=5,
#                                                             # optimizer=optim.Adam,
#                                                             # learning_rate=1e-3,
#                                                             # weight_decay=0
#
#                                                             optimizer=optim.SGD,
#                                                             learning_rate=0.1,
#                                                             learning_rate_decay=True,
#                                                             learning_rate_decay_epoch=[15,35,50,80, 120,160],
#                                                             learning_rate_decay_factor=0.1,
#                                                             weight_decay=1e-4,
#                                                             momentum=0.9,
#                                                             )

# checkpoint = torch.load('./baseline/vgg16bn_cifar100_0.72940.tar')
# checkpoint=torch.load('/home/zzj/fang/model_pytorch/model_saved/vgg16_cifar100_regressor/checkpoint/flop=216253868,accuracy=0.71900.tar')
#
# net = checkpoint['net'].to(device)
#
# net.load_state_dict(checkpoint['state_dict'])
# print(checkpoint['highest_accuracy'])
#
# measure_flops.measure_model(net, 'cifar100', print_flop=True)
# a=[0.05 for i in range(13)]
# a[12]=0.3
# prune_and_train.prune_inactive_neural_with_regressor(net=net,
#                                                      net_name='vgg16_cifar100_regressor',
#                                                      prune_rate=0.15,
#                                                      load_regressor=True,
#                                                      dataset_name='cifar100',
#                                                      filter_preserve_ratio=0.15,
#                                                      max_filters_pruned_for_one_time=a,
#                                                      target_accuracy=0.7177,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=1e8,
#                                                      batch_size=128,
#                                                      num_epoch=250,
#                                                      checkpoint_step=8000,
#                                                      use_random_data=False,
#                                                      round_for_train=2,
#                                                      round=3,
#                                                      # optimizer=optim.Adam,
#                                                      # learning_rate=1e-3,
#                                                      # weight_decay=0
#
#                                                      optimizer=optim.SGD,
#                                                      learning_rate=0.01,  # 标准baseline
#                                                      # learning_rate=0.001,
#                                                      learning_rate_decay=True,
#                                                      learning_rate_decay_epoch=[30, 50, 120, 160, 200],  # 标准baseline
#                                                      # learning_rate_decay_epoch=[10,20,60,90,160],
#                                                      learning_rate_decay_factor=0.2,
#                                                      weight_decay=5e-4,
#                                                      momentum=0.9,
#                                                             )

# checkpoint = torch.load('./baseline/vgg16bn_tinyimagenet_0.73150.tar')
# checkpoint=torch.load('/home/victorfang/PycharmProjects/model_pytorch/model_saved/vgg16bn_tinyimagenet_prune/checkpoint/flop=9501473860,accuracy=0.70140.tar')
#
# net = checkpoint['net'].to(device)
#
# net.load_state_dict(checkpoint['state_dict'])
# print(checkpoint['highest_accuracy'])
#
# measure_flops.measure_model(net, 'tiny_imagenet', print_flop=True)
# a=[0.03 for i in range(13)]
# a[8]=0.2
# # a[0]=a[2]=a[4]=a[6]=a[8]=a[10]=a[12]=0.3
# # a[1]=a[3]=a[5]=a[7]=a[9]=a[11]=0.3
# prune_and_train.prune_inactive_neural_with_regressor(net=net,
#                                                      net_name='vgg16bn_tinyimagenet_prune',
#                                                      prune_rate=0.1,
#                                                      load_regressor=True,
#                                                      dataset_name='tiny_imagenet',
#                                                      filter_preserve_ratio=0.15,
#                                                      max_filters_pruned_for_one_time=a,
#                                                      # [0.11,0.11,0.11,0.11,0.11,0.11,0.08,0.11,0.11,0.11,0.2,0.2,0.2],
#                                                      target_accuracy=0.725,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=7e9,
#                                                      batch_size=24,
#
#                                                      num_epoch=40,
#                                                      checkpoint_step=1000,
#                                                      use_random_data=False,
#                                                      round_for_train=2,
#                                                      round=5,
#
#
#                                                      optimizer=optim.SGD,
#                                                      learning_rate=0.001,
#                                                      # weight_decay=0.0006,
#                                                      momentum=0.9,
#                                                      learning_rate_decay=True,
#                                                      learning_rate_decay_epoch=[5, 14, 21, 28, 35],
#                                                      learning_rate_decay_factor=0.1,
#                                      )

# checkpoint = torch.load('/home/disk_new/model_saved/vgg16bn_cifar10_realdata_regressor6_大幅度/checkpoint/flop=39915982,accuracy=0.93200.tar')
# checkpoint=torch.load('/home/victorfang/PycharmProjects/model_pytorch/model_saved/vgg16bn_cifar10_inactiveFilter_tolerance/checkpoint/flop=39045282,accuracy=0.92880.tar')
#
# net = checkpoint['net'].to(device)
#
# net.load_state_dict(checkpoint['state_dict'])
# print(checkpoint['highest_accuracy'])
#
# measure_flops.measure_model(net, 'cifar10', print_flop=True)
#
#
# prune_and_train.prune_inactive_neural_with_regressor(net=net,
#                                                      net_name='vgg16bn_cifar10_inactiveFilter_tolerance',
#                                                      dataset_name='cifar10',
#                                                      prune_rate=0.02,
#
#                                                      load_regressor=True,
#                                                      round_for_train=2,
#                                                      round=2,
#
#                                                         max_training_iteration=2,
#
#
#                                                      filter_preserve_ratio=0.1,
#                                                      max_filters_pruned_for_one_time=0.3,
#                                                      target_accuracy=0.931,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=1e7,
#                                                      batch_size=1600,
#                                                      num_epoch=450,
#                                                      checkpoint_step=1600,
#                                                      use_random_data=False,
#                                                      # optimizer=optim.Adam,
#                                                      # learning_rate=1e-3,
#                                                      # weight_decay=0
#                                                      optimizer=optim.SGD,
#                                                      learning_rate=0.001,
#                                                      learning_rate_decay=True,
#                                                      learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
#                                                      learning_rate_decay_factor=0.5,
# )


net=vgg.vgg16_bn(pretrained=True).to(device)

# import evaluate
# evaluate.evaluate_net(net=net,data_loader=data_loader.create_validation_loader(batch_size=24,num_workers=8,dataset_name='imagenet'),save_net=False,dataset_name='imagenet')

measure_flops.measure_model(net, 'imagenet', print_flop=True)
a=[0.3 for i in range(13)]
# a[8]=0.2
# a[0]=a[2]=a[4]=a[6]=a[8]=a[10]=a[12]=0.3
# a[1]=a[3]=a[5]=a[7]=a[9]=a[11]=0.3
prune_and_train.prune_inactive_neural_with_regressor(net=net,
                                                     net_name='vgg16bn_imagenet_prune',
                                                     prune_rate=0.1,
                                                     load_regressor=False,
                                                     dataset_name='imagenet',
                                                     filter_preserve_ratio=0.15,
                                                     max_filters_pruned_for_one_time=a,
                                                     # [0.11,0.11,0.11,0.11,0.11,0.11,0.08,0.11,0.11,0.11,0.2,0.2,0.2],
                                                     target_accuracy=0.91,
                                                     tar_acc_gradual_decent=True,
                                                     flop_expected=3e9,
                                                     top_acc=5,

                                                     batch_size=24,

                                                     num_epoch=20,
                                                     checkpoint_step=4000,
                                                     use_random_data=False,
                                                     round_for_train=2,
                                                     round=1,


                                                     optimizer=optim.SGD,
                                                     learning_rate=0.001,
                                                     # weight_decay=0.0006,
                                                     momentum=0.9,
                                                     learning_rate_decay=True,
                                                     learning_rate_decay_epoch=[5, 10, 15],
                                                     learning_rate_decay_factor=0.1,
                                     )