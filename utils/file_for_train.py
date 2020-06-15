import torch
from torch import nn
import torch.optim as optim
from prune import prune_and_train
from framework import evaluate,data_loader,measure_flops,train
from network import vgg,storage,net_with_predicted_mask,resnet_cifar,resnet_cifar
from framework import config as conf
from framework.train import set_modules_no_grad
import os,sys,logger
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#todo:考虑惩罚变化幅度大的层
#训练schedule



optimizer = optim.SGD
learning_rate = {'default': 0.1, 'extractor': 0.1}
weight_decay = {'default':5e-4,'extractor':5e-4}
momentum = {'default':0.9,'extractor':0.9}
# optimizer = optim.Adam
# learning_rate = {'default': 0.01, 'extractor': 0.01}
exp_name='resnet56_predicted_mask_shortcut_with_weight_0.5blpenalty_prune40_longTrain_1'
description=exp_name+'  '+'训练schedule放大3倍,剪40%，其余设置和“resnet56_predicted_mask_shortcut_with_weight_pruneFirstConv_wd5_sgd_blpenaltyNo_5/flop=67510922,accuracy=0.92560.tar”一样'
# exp_name='tmp'


mask_update_freq = 10
mask_update_epochs = 1
mask_training_start_epoch=10
mask_training_stop_epoch=80
# mask_update_freq = 20
# mask_update_epochs = 20
# mask_training_start_epoch=10
# mask_training_stop_epoch=32

batch_size=128
flop_expected=8.1e7
gradient_clip_value=1



learning_rate_decay_epoch = [mask_training_stop_epoch+2*i for i in [80,120]]
num_epochs=160*2+mask_training_stop_epoch

# learning_rate_decay_epoch = [mask_training_stop_epoch+2*i for i in [70,140]]
# num_epochs=220*2+mask_training_stop_epoch

net=resnet_cifar.resnet56(num_classes=10).to(device)

net = net_with_predicted_mask.predicted_mask_shortcut_with_weight_net(net,
                                                                      net_name='resnet56',
                                                                      dataset_name='cifar10',
                                                                      mask_update_epochs=mask_update_epochs,
                                                                      mask_update_freq=mask_update_freq,
                                                                      flop_expected=flop_expected,
                                                                      gcn_rounds=2,
                                                                      mask_training_start_epoch=mask_training_start_epoch,
                                                                      mask_training_stop_epoch=mask_training_stop_epoch,
                                                                      batch_size=batch_size
                                                                      )

# checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_predicted_mask_shortcut_with_weight_pruneFirstConv_wd5_sgd_blpenaltyNo_5/checkpoint/flop=67510922,accuracy=0.92560.tar')
# net.load_state_dict(checkpoint['state_dict'])
# net.current_epoch=81
#
# learning_rate = {'default': 0.001, 'extractor': 0.001}
# learning_rate_decay_epoch = [5]
# num_epochs=1000
# exp_name='resnet56_predicted_mask_shortcut_with_weight_pruneFirstConv_wd5_sgd_blpenaltyNo_5_train'


net=net.to(device)
checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
# save the output to log
print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary

print(optimizer, weight_decay, momentum, learning_rate, mask_update_freq, mask_update_epochs, flop_expected, gradient_clip_value)

# train.add_forward_hook(net,module_name='net.features.1')

train.train(net=net,
            net_name='resnet56',
            exp_name=exp_name,
            description=description,
            dataset_name='cifar10',

            optimizer=optimizer,
            weight_decay=weight_decay,
            momentum=momentum,
            learning_rate=learning_rate,

            num_epochs=num_epochs,
            batch_size=batch_size,
            evaluate_step=5000,
            load_net=False,
            test_net=False,
            num_workers=8,
            # weight_decay=5e-4,
            learning_rate_decay=True,
            learning_rate_decay_epoch=learning_rate_decay_epoch,
            learning_rate_decay_factor=0.1,
            scheduler_name='MultiStepLR',
            top_acc=1,
            data_parallel=False,
            paint_loss=True,
            save_at_each_step=False,
            gradient_clip_value=gradient_clip_value
            )


# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_extractor_static_cifar10_only_gcn_1/checkpoint/flop=62577290,accuracy=0.93400.tar'),pretrained=True)
# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_extractor_static_cifar10_only_gcn_3/checkpoint/flop=62061194,accuracy=0.93410.tar'),pretrained=True)
# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_extractor_static_cifar100_2_train/checkpoint/flop=95299940,accuracy=0.70470.tar'),pretrained=True)

# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet50_extractor_static_imagenet/checkpoint/flop=1662949668,accuracy=0.91526.tar'),pretrained=False)
# net=nn.DataParallel(net)
# success=train.train(net=net,
#
#             net_name='resnet50',
#             exp_name='resnet50_imagenet_prunednet_retrain_from_scratch',
#
#             num_epochs=200,
#             learning_rate=0.1,
#             learning_rate_decay=True,
#             learning_rate_decay_factor=0.1,
#             weight_decay=1e-4,
#             # learning_rate_decay_epoch=[10,30],
#             learning_rate_decay_epoch=[30,60,130,160],
#             dataset_name='imagenet',
#             batch_size=512,
#             evaluate_step=1000,
#             load_net=True,
#             test_net=True,
#             num_workers=8,
#
#             # target_accuracy=0.9264796074467897,
#             top_acc=5,
#                     data_parallel=True)

# checkpoint=dict()
# checkpoint['structure']=[64,26,52,64,128,128,128,256,52,52,52,52,52]
# checkpoint['dataset_name']='cifar10'
# checkpoint['net_name']='vgg16_bn'
# net=storage.restore_net(checkpoint,pretrained=False)
# measure_flops.measure_model(net,dataset_name='cifar10')
# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16_extractor_static_cifar10/checkpoint/flop=48525158,accuracy=0.93140.tar'),pretrained=False)
#
# # net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16bn_cifar10_prunednet_retrain_from_scratch/checkpoint/flop=48525158,accuracy=0.91900.tar'),pretrained=True)
# # i=0
# # while i<3:
# success = train.train(net=net,
#
#                       net_name='vgg16_bn',
#                       exp_name='vgg16bn_cifar10_afp_retrain_from_scratch',
#
#                       num_epochs=400,
#                       learning_rate=0.1,
#                       learning_rate_decay=True,
#                       learning_rate_decay_factor=0.1,
#                       weight_decay=1e-4,
#                       momentum=0.9,
#                       # learning_rate_decay_epoch=[10,30],
#                       learning_rate_decay_epoch=[2*i for i in [100,130]],
#                       dataset_name='cifar10',
#                       batch_size=128,
#                       evaluate_step=1000,
#                       load_net=True,
#                       test_net=True,
#                       num_workers=2,
#
#                       # target_accuracy=0.9264796074467897,
#                       top_acc=1,
#                       data_parallel=False)
#
#     i+=1


#
# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_extractor_static_cifar10/checkpoint/flop=71074442,accuracy=0.93330.tar'),pretrained=False)
#
# a=[7,9,9,9,11,7,7,10,10,12,16,11,12,16,9,6,6,15,13,7   ,18,21,15,18,26, 17,25]
# a=[3+i for i in a]
# p_f_s=[16,     a[0],16,a[1],16,a[2],16,a[3],16,a[4],16,a[5],16,a[6],16,a[7],16,a[8],16,
#    a[9],32,a[10],32,a[11],32,a[12],32,a[13],32,a[14],32,a[15],32,a[16],32,a[17],32,
#    a[18],64,a[19],64,a[20],64,a[21],64,a[22],64,a[23],64,a[24],64,a[25],64,a[26],64]
#
# checkpoint=dict()
# checkpoint['structure']=p_f_s
# checkpoint['net_name']='resnet56'
# checkpoint['dataset_name']='cifar10'
# net=storage.restore_net(checkpoint,pretrained=False)
#
# # net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_cifar10_prunednet_retrain_from_scratch/checkpoint/flop=71074442,accuracy=0.91930.tar'),pretrained=True)
# i=0
# # while i<3:
# success = train.train(net=net,
#
#                       net_name='resnet56',
#                       exp_name='tmp',
#
#
#                       learning_rate=0.1,
#                       learning_rate_decay=True,
#                       weight_decay=1e-4,
#                       momentum=0.9,
#
#                        num_epochs=320,
#                       learning_rate_decay_factor=0.1,
#                       # learning_rate_decay_epoch=[10,30],
#                       learning_rate_decay_epoch=[2*i for i in [80,120]],
#
#                       dataset_name='cifar10',
#                       batch_size=128,
#                       evaluate_step=1000,
#                       load_net=True,
#                       test_net=True,
#                       num_workers=4,
#
#                       # target_accuracy=0.9264796074467897,
#                       top_acc=1,
#                       data_parallel=False,
#                       paint_loss=True
#                       )
#     i+=1

# success=False
# while not success and i<1:
    # success=train.train(net=net,
    #
    #             net_name='vgg16_bn',
    #             exp_name='vgg16_extractor_static_imagenet_train',
    #
    #             num_epochs=100,
    #             learning_rate=0.001,
    #             learning_rate_decay=True,
    #             learning_rate_decay_factor=0.1,
    #             weight_decay=1e-4,
    #             # learning_rate_decay_epoch=[10,30],
    #             learning_rate_decay_epoch=[60],
    #             dataset_name='imagenet',
    #             batch_size=512,
    #             evaluate_step=1000,
    #             load_net=True,
    #             test_net=True,
    #             num_workers=8,
    #
    #             target_accuracy=0.9013827754961294,
    #             top_acc=5,)

    # success=train.train(net=net,
    #
    #             net_name='resnet50',
    #             exp_name='resnet50_extractor_static_imagenet_train2',
    #
    #             num_epochs=100,
    #             learning_rate=0.001,
    #             learning_rate_decay=True,
    #             learning_rate_decay_factor=0.1,
    #             weight_decay=1e-4,
    #             # learning_rate_decay_epoch=[10,30],
    #             learning_rate_decay_epoch=[30,60],
    #             dataset_name='imagenet',
    #             batch_size=512,
    #             evaluate_step=1000,
    #             load_net=True,
    #             test_net=True,
    #             num_workers=8,
    #
    #             target_accuracy=0.9264796074467897,
    #             top_acc=5,
    #                     data_parallel=True)
#flop=2495193892,accuracy=0.92332.tar
    # success=train.train(net=net,
    #
    #             net_name='resnet50',
    #             exp_name='resnet50_extractor_static_imagenet_train2',
    #
    #             num_epochs=60,
    #             learning_rate=0.01,
    #             learning_rate_decay=True,
    #             learning_rate_decay_factor=0.1,
    #             weight_decay=1e-4,
    #             # learning_rate_decay_epoch=[10,30],
    #             learning_rate_decay_epoch=[15,30,45],
    #             dataset_name='imagenet',
    #             batch_size=512,
    #             evaluate_step=1000,
    #             load_net=True,
    #             test_net=True,
    #             num_workers=8,
    #
    #             target_accuracy=0.9266966598185884,
    #             top_acc=5,
    #                     data_parallel=True)


    # success=train.train(net=net,
    #             net_name='resnet56',
    #             exp_name='resnet56_extractor_static_cifar100_2'+'_train',
    #
    #                     num_epochs=200,
    #                     learning_rate=0.001,
    #                     learning_rate_decay=True,
    #                     learning_rate_decay_epoch=[60, 120, 160],  # , 150, 250, 300, 350, 400],
    #                     learning_rate_decay_factor=0.1,
    #                     weight_decay=1e-4,
    #
    #
    #             dataset_name='cifar100',
    #             # num_epochs=450,
    #             batch_size=512,
    #             evaluate_step=3000,
    #             load_net=False,
    #             test_net=True,
    #             num_workers=2,
    #
    #             target_accuracy=0.7108089533597423,
    #             top_acc=1)

    # no_grad = set_modules_no_grad(net, ['layer3.block8', 'fc'])
    # success=train.train(net=net,
    #
    #             net_name='resnet56',
    #             exp_name='resnet56_extractor_static_cifar10_only_gcn_1'+'_train',
    #
    #          optimizer=optim.SGD,
    #          learning_rate=0.00001,  # 标准baseline
    #          learning_rate_decay=True,
    #          learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
    #          learning_rate_decay_factor=0.5,
    #          weight_decay=5e-4,
    #
    #             dataset_name='cifar10',
    #             num_epochs=450,
    #             batch_size=512,
    #             evaluate_step=3000,
    #             load_net=False,
    #             test_net=True,
    #             num_workers=2,
    #
    #             target_accuracy=0.9349782293095876,
    #             top_acc=1,
    #                     no_grad=no_grad)

    # success = train.train(net=net,
    #
    #                       net_name='resnet56',
    #                       exp_name='resnet56_extractor_static_cifar10_only_gcn_3' + '_train',
    #
    #                       optimizer=optim.SGD,
    #                       learning_rate=0.0001,  # 标准baseline
    #                       learning_rate_decay=True,
    #                       learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
    #                       learning_rate_decay_factor=0.5,
    #                       weight_decay=5e-4,
    #
    #                       dataset_name='cifar10',
    #                       num_epochs=450,
    #                       batch_size=512,
    #                       evaluate_step=3000,
    #                       load_net=False,
    #                       test_net=True,
    #                       num_workers=2,
    #
    #                       target_accuracy=0.9349782293095876,
    #                       top_acc=1)


    # i+=1
