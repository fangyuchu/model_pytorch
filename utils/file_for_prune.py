import torch
from torch import nn
import torch.optim as optim
from prune import prune_and_train,prune_and_train_with_mask
from framework import evaluate,data_loader,measure_flops,train
from framework.train import name_parameters_no_grad
from network import create_net,net_with_mask,vgg,storage
from framework import config as conf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# os.environ["CUDA_VISIBLE_DEVICES"] ='1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# checkpoint=torch.load('../data/baseline/resnet56_cifar100_0.71580.tar')
# checkpoint=torch.load('/home/zengyao/fang/model_pytorch/model_saved/resnet56_cifar100_regressor3/checkpoint/flop=90655076,accuracy=0.71000.tar')
# net=storage.restore_net(checkpoint).to(device)
# net.load_state_dict(checkpoint['state_dict'])
# net.to(device)
# measure_flops.measure_model(net,'cifar100')
# prune_and_train.prune_inactive_neural_with_regressor_resnet(net=net,
#                                                             exp_name='resnet56_cifar100_regressor3',
#                                                               net_name='resnet56',
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
#                                                             evaluate_step=3000,
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
#
# checkpoint = torch.load('../data/baseline/vgg16bn_cifar100_0.72940.tar')
#
# net=storage.restore_net(checkpoint).to(device)
#
# net.load_state_dict(checkpoint['state_dict'])
# print(checkpoint['highest_accuracy'])
#
# measure_flops.measure_model(net, 'cifar100', print_flop=True)
# a=[0.05 for i in range(13)]
# a[12]=0.3
# prune_and_train.prune_inactive_neural_with_regressor(net=net,
#                                                      exp_name='tmp',
#                                                               net_name='vgg16_bn',
#                                                      prune_rate=0.15,
#                                                      load_regressor=False,
#                                                      dataset_name='cifar100',
#                                                      filter_preserve_ratio=0.15,
#                                                      max_filters_pruned_for_one_time=a,
#                                                      target_accuracy=0.7177,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=1e8,
#                                                      batch_size=128,
#                                                      num_epoch=250,
#                                                      evaluate_step=8000,
#                                                      use_random_data=False,
#                                                      round_for_train=2,
#                                                      round=1,
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
#
# checkpoint = torch.load('../data/baseline/vgg16bn_tinyimagenet_0.73150.tar')
# checkpoint=torch.load('/home/victorfang/PycharmProjects/model_pytorch/model_saved/vgg16bn_tinyimagenet_prune/checkpoint/flop=9501473860,accuracy=0.70140.tar')
#
# net=storage.restore_net(checkpoint).to(device)
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
#                                                      exp_name='vgg16bn_tinyimagenet_prune',
#                                                               net_name='vgg16_bn',
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
#                                                      evaluate_step=1000,
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
#
# checkpoint = torch.load('/home/disk_new/model_saved/vgg16bn_cifar10_realdata_regressor6_大幅度/checkpoint/flop=39915982,accuracy=0.93200.tar')
# checkpoint=torch.load('/home/victorfang/PycharmProjects/model_pytorch/model_saved/vgg16bn_cifar10_inactiveFilter_tolerance/checkpoint/flop=39045282,accuracy=0.92880.tar')
#
# net=storage.restore_net(checkpoint).to(device)
#
# net.load_state_dict(checkpoint['state_dict'])
# print(checkpoint['highest_accuracy'])
#
# measure_flops.measure_model(net, 'cifar10', print_flop=True)
#
#
# prune_and_train.prune_inactive_neural_with_regressor(net=net,
#                                                      exp_name='vgg16bn_cifar10_inactiveFilter_tolerance',
#                                                               net_name='vgg16_bn',
#                                                      dataset_name='cifar10',
#                                                      prune_rate=0.02,
#
#                                                      load_regressor=True,
#                                                      round_for_train=2,
#                                                      round=2,
#
#                                                         max_training_round=2,
#
#
#                                                      filter_preserve_ratio=0.1,
#                                                      max_filters_pruned_for_one_time=0.3,
#                                                      target_accuracy=0.931,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=1e7,
#                                                      batch_size=1600,
#                                                      num_epoch=450,
#                                                      evaluate_step=1600,
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
#
#
# net=vgg.vgg16_bn(pretrained=True).to(device)
# net=torch.nn.DataParallel(net)              #多gpu训练
#
# # import evaluate
# # evaluate.evaluate_net(net=net,data_loader=data_loader.create_validation_loader(batch_size=24,num_workers=8,dataset_name='imagenet'),save_net=False,dataset_name='imagenet')
#
# measure_flops.measure_model(net, 'imagenet', print_flop=True)
# a=[0.3 for i in range(13)]
# # a[8]=0.2
# # a[0]=a[2]=a[4]=a[6]=a[8]=a[10]=a[12]=0.3
# # a[1]=a[3]=a[5]=a[7]=a[9]=a[11]=0.3
# prune_and_train.prune_inactive_neural_with_regressor(net=net,
#                                                      exp_name='vgg16bn_imagenet_prune',
#                                                               net_name='vgg16_bn',
#                                                      prune_rate=0.1,
#                                                      load_regressor=False,
#                                                      dataset_name='imagenet',
#                                                      filter_preserve_ratio=0.15,
#                                                      max_filters_pruned_for_one_time=a,
#                                                      # [0.11,0.11,0.11,0.11,0.11,0.11,0.08,0.11,0.11,0.11,0.2,0.2,0.2],
#                                                      target_accuracy=0.91,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=3e9,
#                                                      top_acc=5,
#
#                                                      batch_size=256,
#
#                                                      num_epoch=20,
#                                                      evaluate_step=4000,
#                                                      use_random_data=False,
#                                                      round_for_train=2,
#                                                      round=1,
#
#
#                                                      optimizer=optim.SGD,
#                                                      learning_rate=0.001,
#                                                      # weight_decay=0.0006,
#                                                      momentum=0.9,
#                                                      learning_rate_decay=True,
#                                                      learning_rate_decay_epoch=[5, 10, 15],
#                                                      learning_rate_decay_factor=0.1,
#                                      )
#
#
#
# #只训练全连接层
# checkpoint=torch.load('../data/baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
# net=storage.restore_net(checkpoint).to(device)
# # 载入预训练模型参数后...
# for name, value in net.named_parameters():
#     if 'classifier' in name :
#         value.requires_grad = True
#     else:
#         value.requires_grad=False
#
# net.load_state_dict(checkpoint['state_dict'])
# print(checkpoint['highest_accuracy'])
#
# measure_flops.measure_model(net, 'cifar10', print_flop=True)
# prune_and_train.prune_inactive_neural_with_regressor(net=net,
#                                                      exp_name='test',
#                                                               net_name='vgg16_bn',
#                                                      dataset_name='cifar10',
#                                                      prune_rate=0.02,
#                                                      load_regressor=False,
#                                                      round_for_train=2,
#                                                      round=1,
#                                                      max_training_round=2,
#                                                      filter_preserve_ratio=0.1,
#                                                      max_filters_pruned_for_one_time=0.3,
#                                                      target_accuracy=0.931,
#                                                      tar_acc_gradual_decent=False,
#                                                      flop_expected=1e7,
#                                                      batch_size=512,
#                                                      num_epoch=450,
#                                                      evaluate_step=1600,
#                                                      use_random_data=False,
#                                                      # optimizer=optim.Adam,
#                                                      # learning_rate=1e-3,
#                                                      # weight_decay=0
#                                                      optimizer=optim.SGD,
#                                                      learning_rate=0.001,
#                                                      learning_rate_decay=True,
#                                                      learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
#                                                      learning_rate_decay_factor=0.5,
#                                                      no_grad=['features']
#                                                      )
#
# 用图片抽样剪，获取regressor的训练数据
#
# print(torch.cuda.is_available())
# checkpoint=torch.load('../data/baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
#
# net=storage.restore_net(checkpoint)
# net.load_state_dict(checkpoint['state_dict'])
# net=torch.nn.DataParallel(net)
# from framework import evaluate
# from framework import data_loader
# # evaluate.evaluate_net(net,data_loader=data_loader.create_validation_loader(batch_size=256,num_workers=4,dataset_name='cifar10'),save_net=False)
#
# measure_flops.measure_model(net, 'cifar10', print_flop=True)
# prune_and_train.prune_inactive_neural_with_regressor(net=net,
#                                                      exp_name='vgg16_realdata2',
#                                                      net_name='vgg16_bn',
#                                                      dataset_name='cifar10',
#                                                      prune_rate=0.02,
#                                                      load_regressor=False,
#                                                      round_for_train=10,
#                                                      round=1,
#
#                                                         max_training_round=2,
#
#
#                                                      filter_preserve_ratio=0.1,
#                                                      max_filters_pruned_for_one_time=0.3,
#                                                      target_accuracy=0.931,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=1e7,
#                                                      batch_size=512,
#                                                      num_epoch=450,
#                                                      evaluate_step=1600,
#                                                      use_random_data=False,
#                                                      # optimizer=optim.Adam,
#                                                      # learning_rate=1e-3,
#                                                      # weight_decay=0
#                                                      optimizer=optim.SGD,
#                                                      learning_rate=0.01,
#                                                      learning_rate_decay=True,
#                                                      learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
#                                                      learning_rate_decay_factor=0.5,
# )
#
#
# print(torch.cuda.is_available())
# net=storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'baseline/vgg16_bn_cifar10,accuracy=0.941.tar')))
#
# # evaluate.evaluate_net(net,data_loader=data_loader.create_validation_loader(batch_size=256,num_workers=4,dataset_name='cifar10'),save_net=False)
#
# measure_flops.measure_model(net, 'cifar10', print_flop=True)
# prune_and_train.prune_inactive_neural_with_regressor(net=net,
#                                                      exp_name='test',
#                                                      net_name='vgg16_bn',
#                                                      dataset_name='cifar10',
#                                                      prune_rate=0.1,
#                                                      load_regressor=False,
#                                                      round_for_train=2,
#                                                      round=1,
#                                                         max_training_round=2,
#                                                      filter_preserve_ratio=0.1,
#                                                      max_filters_pruned_for_one_time=0.3,
#                                                      target_accuracy=0.931,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=4e7,
#                                                      batch_size=512,
#                                                      num_epoch=450,
#                                                      evaluate_step=1600,
#                                                      use_random_data=False,
#                                                      # optimizer=optim.Adam,
#                                                      # learning_rate=1e-3,
#                                                      # weight_decay=0
#                                                      optimizer=optim.SGD,
#                                                      learning_rate=0.01,
#                                                      learning_rate_decay=True,
#                                                      learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
#                                                      learning_rate_decay_factor=0.5,)
#
# net = net_with_mask.NetWithMask(dataset_name='imagenet', net_name='vgg16_bn')
# net=nn.DataParallel(net)
# # net = nn.DataParallel(net)
#
#
#
# prune_and_train_with_mask.dynamic_prune_inactive_neural_with_feature_extractor(net=net,
#                                                                                net_name='vgg16_bn',
#                                                                                exp_name='vgg16_imagenet_mask_net_less_train_after_finetune_004_delta',
#
#                                                                                target_accuracy=0.894,
#                                                                                initial_prune_rate=0.1,
#                                                                                delta_prune_rate=0.04,
#                                                                                round_for_train=100,
#                                                                                tar_acc_gradual_decent=True,
#                                                                                flop_expected=0.2,
#                                                                                dataset_name='imagenet',
#                                                                                batch_size=512,
#                                                                                num_workers=7,
#                                                                                evaluate_step=1000,
#                                                                                num_epoch=20,
#                                                                                num_epoch_after_finetune=1,
#                                                                                filter_preserve_ratio=0.3,
#                                                                                optimizer=optim.SGD,
#                                                                                learning_rate=0.001,
#                                                                                learning_rate_decay=True,
#                                                                                learning_rate_decay_factor=0.1,
#                                                                                weight_decay=5e-4,
#                                                                                learning_rate_decay_epoch=[5,10,15],
#                                                                                max_training_round=1,
#                                                                                round=1,
#                                                                                top_acc=5,
#                                                                                max_data_to_test=1000
#                                                                                )
#
# vgg16_extractor_static_imagenet
# net=vgg.vgg16_bn(pretrained=True,dataset_name='imagenet').to(device)
# checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16_extractor_static_imagenet/checkpoint/flop=11508069616,accuracy=0.90156.tar')
# net=storage.restore_net(checkpoint,pretrained=True)
# net=nn.DataParallel(net)
# max_filters_pruned_for_one_time=[0.15 for i in range(13)]
# max_filters_pruned_for_one_time[10]=0.3
# max_filters_pruned_for_one_time[11]=0.3
# max_filters_pruned_for_one_time[12]=0.3
#
# prune_and_train.prune_inactive_neural_with_extractor(net=net,
#                                                      net_name='vgg16_bn',
#                                                      exp_name='vgg16_extractor_static_imagenet',
#                                                      target_accuracy=0.8981,
#                                                      prune_rate=0.1,
#                                                      round_for_train=2,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=0.2,
#                                                      dataset_name='imagenet',
#                                                      batch_size=256,
#                                                      num_workers=7,
#                                                      optimizer=optim.SGD,
#                                                      learning_rate=0.01,
#                                                      evaluate_step=1000,
#                                                      num_epoch=15,
#                                                      filter_preserve_ratio=0.15,
#                                                      max_filters_pruned_for_one_time=max_filters_pruned_for_one_time,
#                                                      learning_rate_decay=True,
#                                                      learning_rate_decay_factor=0.1,
#                                                      weight_decay=5e-4,
#                                                      learning_rate_decay_epoch=[3,8],
#                                                      max_training_round=1,
#                                                      round=3,
#                                                      top_acc=5,
#                                                      max_data_to_test=10000,
#                                                      extractor_epoch=700,
#                                                      extractor_feature_len=15,
#                                                      gcn_rounds=2
#                                                      )


































# # #vgg16bn_tinyimagenet_extractor_static
# checkpoint = torch.load(os.path.join(conf.root_path,'baseline/vgg16bn_tinyimagenet_0.73150.tar'))
# # checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16bn_tinyimagenet_extractor_static_dataparallel/checkpoint/flop=10503648516,accuracy=0.71200.tar')
# net=storage.restore_net(checkpoint)
# net=nn.DataParallel(net)
# a=[0.3 for i in range(13)]
# # a[0]=a[2]=a[4]=a[6]=a[8]=a[10]=a[12]=0.3
# # a[1]=a[3]=a[5]=a[7]=a[9]=a[11]=0.3
# prune_and_train.prune_inactive_neural_with_extractor(net=net,
#                                                      net_name='vgg16_bn',
#                                                      exp_name='vgg16bn_tinyimagenet_extractor_static_dataparallel',
#                                                      target_accuracy=0.70,
#                                                      prune_rate=0.1,
#                                                      round_for_train=2,
#                                                      round_to_train_freq=6,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=7e9,
#                                                      dataset_name='tiny_imagenet',
#                                                      batch_size=512,
#                                                      num_workers=8,
#                                                      optimizer=optim.SGD,
#                                                      learning_rate=0.01,
#                                                      evaluate_step=100,
#                                                      num_epoch=40,
#                                                      filter_preserve_ratio=0.2,
#                                                      max_filters_pruned_for_one_time=a,
#                                                      learning_rate_decay=True,
#
#                                                      learning_rate_decay_epoch=[7, 14, 21, 28, 35],
#                                                      learning_rate_decay_factor=0.1,
#                                                      weight_decay=5e-4,
#
#                                                      max_training_round=2,
#                                                      round=1,
#                                                      top_acc=1,
#                                                      max_data_to_test=10000,
#                                                      extractor_epoch=100,
#                                                      extractor_feature_len=15,
#                                                      gcn_rounds=2,
#
#                                      )



# # #resnet18_tinyimagenet_extractor_static
# checkpoint = torch.load(os.path.join(conf.root_path,'baseline/resnet18_tinyimagenet_v2_0.72990.tar'))
# checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet18_tinyimagenet_extractor_static/checkpoint/flop=1396513224,accuracy=0.69480.tar')
# net=storage.restore_net(checkpoint)
# net=nn.DataParallel(net)
# a=[1 for i in range(20)]
# # a[1]=a[5]=a[9]=a[13]=0.3
# # a[3]=a[7]=a[11]=a[15]=0.1
# prune_and_train.prune_inactive_neural_with_extractor(net=net,
#                                                      net_name='resnet18',
#                                                      exp_name='resnet18_tinyimagenet_extractor_static',
#                                                      target_accuracy=0.69,
#                                                      prune_rate=0.15,
#                                                      round_for_train=3,
#                                                      round_to_train_freq=6,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=12.2e8,
#                                                      dataset_name='tiny_imagenet',
#                                                      batch_size=512,
#                                                      num_workers=8,
#                                                      optimizer=optim.SGD,
#                                                      learning_rate=0.001,
#                                                      evaluate_step=100,
#                                                      num_epoch=40,
#                                                      filter_preserve_ratio=0.2,
#                                                      max_filters_pruned_for_one_time=a,
#                                                      learning_rate_decay=True,
#
#                                                      learning_rate_decay_epoch=[ 7,  14, 21,28,35],
#                                                      learning_rate_decay_factor=0.1,
#                                                      weight_decay=5e-4,
#
#                                                      max_training_round=2,
#                                                      round=2,
#                                                      top_acc=1,
#                                                      max_data_to_test=10000,
#                                                      extractor_epoch=300,
#                                                      extractor_feature_len=30,
#                                                      gcn_rounds=1,
#
#                                      )

# # # vgg16_extractor_static_cifar10
# # net=storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'baseline/vgg16_bn_cifar10,accuracy=0.941.tar')))
# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16_extractor_static_cifar10/checkpoint/flop=48525158,accuracy=0.93140.tar'),pretrained=True)
#
# filter_preserve_ratio=[0.2 for i in range(13)]
# filter_preserve_ratio=[0.1 for i in range(13)]
#
# filter_preserve_ratio[0]=filter_preserve_ratio[1]=0.25
#
# # filter_preserve_ratio[8]=filter_preserve_ratio[9]=0.14
#
# # max_filters_pruned_for_one_time=[0.3 for i in range(13)]
# # # max_filters_pruned_for_one_time[5]=0
# # max_filters_pruned_for_one_time[5]=0.15
# # max_filters_pruned_for_one_time[6]=0.15
# # max_filters_pruned_for_one_time[7]=0.15
# # max_filters_pruned_for_one_time[8]=0.15
# # max_filters_pruned_for_one_time[9]=0.15
# # max_filters_pruned_for_one_time[10]=0.2
# # # max_filters_pruned_for_one_time[11]=0.6
# # max_filters_pruned_for_one_time[12]=0.2
#
# max_filters_pruned_for_one_time=[1 for i in range(13)]
# # max_filters_pruned_for_one_time[8]=max_filters_pruned_for_one_time[9]=0.1
# # max_filters_pruned_for_one_time[2]=0.1
#
# # a=[0.05 for i in range(13)]
# # a[0]=a[2]=a[4]=a[6]=a[8]=a[10]=a[12]=0.1
# # a[1]=a[3]=a[5]=a[7]=a[9]=a[11]=0
#
# prune_and_train.prune_inactive_neural_with_extractor(net=net,
#                                                      net_name='vgg16_bn',
#                                                      exp_name='vgg16_extractor_static_cifar10',
#                                                      # exp_name='vgg16_extractor_static_cifar10_0.1prunerate',
#                                                      target_accuracy=0.931,
#                                                      prune_rate=0.05,
#                                                      round_for_train=2,
#                                                      round_to_train_freq=6,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=3.9e7,
#                                                      dataset_name='cifar10',
#                                                      batch_size=512,
#                                                      num_workers=8,
#                                                      optimizer=optim.SGD,
#
#                                                      evaluate_step=3000,
#                                                      num_epoch=450,
#                                                      filter_preserve_ratio=filter_preserve_ratio,
#                                                      max_filters_pruned_for_one_time=max_filters_pruned_for_one_time,
#                                                      learning_rate_decay=True,
#                                                      learning_rate_decay_factor=0.5,
#                                                      weight_decay=5e-4,
#                                                      learning_rate=0.01,
#                                                      learning_rate_decay_epoch=[20,50, 100, 150, 250, 300, 350, 400],
#                                                     # learning_rate=0.01,
#                                                     #  learning_rate_decay_epoch=[20,50, 100, 150, 250, 300, 350, 400],
#                                                      max_training_round=2,
#                                                      round=31,
#                                                      top_acc=1,
#                                                      max_data_to_test=10000,
#                                                      extractor_epoch=100,
#                                                      extractor_feature_len=15,
#                                                      gcn_rounds=2
#                                                      )

# # # resnet56_extractor_static_cifar10
# net=storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'baseline/resnet56_cifar10,accuracy=0.94230.tar')))
# # net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_cifar10_baseline+1/checkpoint/flop=125485706,accuracy=0.93270.tar'),pretrained=True)
# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_extractor_static_cifar10/checkpoint/flop=77783690,accuracy=0.93550.tar'),pretrained=True)
# max_filters_pruned_for_one_time=[0.1 for i in range(55)]
# # max_filters_pruned_for_one_time[53]=max_filters_pruned_for_one_time[19]=max_filters_pruned_for_one_time[37]=max_filters_pruned_for_one_time[51]=0.1
#
# prune_and_train.prune_inactive_neural_with_extractor(net=net,
#                                                      net_name='resnet56',
#                                                      exp_name='resnet56_extractor_static_cifar10',
#                                                      target_accuracy=0.9342,
#                                                      prune_rate=0.03,
#                                                      round_for_train=2,
#                                                      round_to_train_freq=6,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=5.5e7,
#                                                      dataset_name='cifar10',
#                                                      batch_size=512,
#                                                      num_workers=2,
#                                                      optimizer=optim.SGD,
#                                                      learning_rate=0.001,
#                                                      evaluate_step=3000,
#                                                      num_epoch=450,
#                                                      filter_preserve_ratio=0.2,
#                                                      max_filters_pruned_for_one_time=max_filters_pruned_for_one_time,
#                                                      learning_rate_decay=True,
#                                                      learning_rate_decay_factor=0.5,
#                                                      weight_decay=5e-4,
#                                                      learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
#                                                      max_training_round=2,
#                                                      round=6,
#                                                      top_acc=1,
#                                                      max_data_to_test=10000,
#                                                      extractor_epoch=300,
#                                                      extractor_feature_len=10,
#                                                      gcn_rounds=1,
#                                                      only_gcn=False
#                                                      )



# # vgg16_cifar100_extractor_static
# checkpoint = torch.load('/home/victorfang/model_pytorch/data/baseline/vgg16bn_cifar100_0.73020.tar')#torch.load(os.path.join(conf.root_path,'baseline/vgg16_cifar100_0.72940.tar'))
# checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16bn_cifar100_extractor_static/checkpoint/flop=191568244,accuracy=0.72060.tar')
# net=storage.restore_net(checkpoint).to(device)
#
# filter_preserve_ratio=[0.15 for i in range(13)]
# max_filters_pruned_for_one_time=[0.3 for i in range(13)]
# # max_filters_pruned_for_one_time[8]=0
# prune_and_train.prune_inactive_neural_with_extractor(net=net,
#                                                      exp_name='vgg16bn_cifar100_extractor_static',
#                                                      net_name='vgg16_bn',
#                                                      prune_rate=0.1,
#                                                      dataset_name='cifar100',
#                                                      filter_preserve_ratio=filter_preserve_ratio,
#                                                      max_filters_pruned_for_one_time=max_filters_pruned_for_one_time,
#                                                      target_accuracy=0.7189,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=1.78e8,
#                                                      batch_size=512,
#                                                      # num_epoch=250,
#                                                      evaluate_step=8000,
#                                                      round_for_train=2,
#                                                      round=3,
#
#                                                      num_epoch=200,
#                                                      learning_rate=0.001,
#                                                      learning_rate_decay_epoch=[60, 120, 160],
#                                                      learning_rate_decay_factor=0.2,
#
#                                                      optimizer=optim.SGD,
#                                                      # learning_rate=0.01,  # 标准baseline
#                                                      learning_rate_decay=True,
#                                                      # learning_rate_decay_epoch=[20,50, 120,160,200],
#                                                      # learning_rate_decay_factor=0.2,
#                                                      weight_decay=5e-3,
#                                                      top_acc=1,
#                                                      max_data_to_test=10000,
#                                                      extractor_epoch=100,
#                                                      extractor_feature_len=27,
#                                                      gcn_rounds=2,
#                                                      only_gcn=False,
#                                                     max_training_round=2,
#                                                             )


# # resnet56_extractor_static_cifar10
# net=storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'baseline/resnet56_cifar10,accuracy=0.94230.tar')))
# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_extractor_static_cifar10_only_gcn_1/checkpoint/flop=62724746,accuracy=0.93460.tar'),pretrained=True)
# max_filters_pruned_for_one_time=[0 for i in range(55)]
# no_grad=[]
#
# max_filters_pruned_for_one_time[51]=0.2
# no_grad=name_parameters_no_grad(net,['layer3.block7','fc'])
#
# prune_and_train.prune_inactive_neural_with_extractor(net=net,
#                                                      net_name='resnet56',
#                                                      exp_name='resnet56_extractor_static_cifar10_only_gcn_3',
#                                                      target_accuracy=0.9344,
#                                                      prune_rate=0.1,
#                                                      round_for_train=2,
#                                                      round_to_train_freq=6,
#                                                      tar_acc_gradual_decent=True,
#                                                      flop_expected=5.95e7,
#                                                      dataset_name='cifar10',
#                                                      batch_size=512,
#                                                      num_workers=4,
#
#                                                      optimizer=optim.SGD,
#                                                      learning_rate=0.0001,
#                                                      learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
#                                                      learning_rate_decay=True,
#                                                      learning_rate_decay_factor=0.5,
#
#                                                      no_grad=no_grad,
#
#                                                      # optimizer=optim.Adam,
#                                                      # learning_rate=0.0001,
#                                                      # learning_rate_decay=False,
#
#                                                      evaluate_step=3000,
#                                                      num_epoch=450,
#                                                      filter_preserve_ratio=0.1,
#                                                      max_filters_pruned_for_one_time=max_filters_pruned_for_one_time,
#                                                      weight_decay=5e-4,
#                                                      max_training_round=5,
#                                                      round=6,
#                                                      top_acc=1,
#                                                      max_data_to_test=10000,
#                                                      extractor_epoch=300,
#                                                      extractor_feature_len=5,
#                                                      gcn_rounds=1,
#                                                      only_gcn=False
#                                                      )


# # #
# # resnet56_extractor_static_cifar100
# net=storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'baseline/resnet56_cifar100_0.70370.tar')))
net=storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'baseline/resnet56_cifar100_0.71580.tar')))

net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_extractor_static_cifar100_2/checkpoint/flop=112773476,accuracy=0.71230.tar'),pretrained=True)
max_filters_pruned_for_one_time=[0.2 for i in range(55)]
# max_filters_pruned_for_one_time[53]=max_filters_pruned_for_one_time[19]=max_filters_pruned_for_one_time[37]=max_filters_pruned_for_one_time[51]=0.1

prune_and_train.prune_inactive_neural_with_extractor(net=net,
                                                     net_name='resnet56',
                                                     exp_name='resnet56_extractor_static_cifar100_2',
                                                     target_accuracy=0.7058,
                                                     prune_rate=0.05,
                                                     round_for_train=3,
                                                     round_to_train_freq=6,
                                                     tar_acc_gradual_decent=True,
                                                     flop_expected=6.5e7,
                                                     dataset_name='cifar100',
                                                     batch_size=512,
                                                     num_workers=4,

                                                     evaluate_step=3000,

                                                     num_epoch=200,
                                                     learning_rate=0.001,
                                                     learning_rate_decay=True,
                                                     learning_rate_decay_epoch=[60,120,160],  # , 150, 250, 300, 350, 400],
                                                     learning_rate_decay_factor=0.1,
                                                     weight_decay=1e-4,

                                                     # num_epoch=450,

                                                     # optimizer=optim.SGD,
                                                     # learning_rate=0.01,
                                                     # learning_rate_decay=True,
                                                     # learning_rate_decay_factor=0.5,
                                                     # learning_rate_decay_epoch=[20,50,100,150, 250, 300, 350, 400],
                                                     # weight_decay=5e-4,


                                                     filter_preserve_ratio=0.2,
                                                     max_filters_pruned_for_one_time=max_filters_pruned_for_one_time,
                                                     max_training_round=3,
                                                     round=2,
                                                     top_acc=1,
                                                     max_data_to_test=10000,
                                                     extractor_epoch=300,
                                                     extractor_feature_len=10,
                                                     gcn_rounds=2,
                                                     only_gcn=False
                                                     )