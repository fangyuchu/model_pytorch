import torch
from torch import nn
import torch.optim as optim
from prune import prune_and_train,prune_and_train_with_mask
from framework import evaluate,data_loader,measure_flops,train
from network import create_net,net_with_mask,vgg,storage
from framework import config as conf
from framework.train import name_parameters_no_grad
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_extractor_static_cifar10_only_gcn_1/checkpoint/flop=62577290,accuracy=0.93400.tar'),pretrained=True)
# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_extractor_static_cifar10_only_gcn_3/checkpoint/flop=62061194,accuracy=0.93410.tar'),pretrained=True)
# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_extractor_static_cifar100_2_train/checkpoint/flop=95299940,accuracy=0.70470.tar'),pretrained=True)

net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet50_extractor_static_imagenet2/checkpoint/flop=1796559732,accuracy=0.91862.tar'),pretrained=True)

# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet18_tinyimagenet_extractor_static_train/checkpoint/flop=1289487816,accuracy=0.67650.tar'),pretrained=True)
net=torch.nn.DataParallel(net)
i=0
success=False
while not success and i<1:
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

    success=train.train(net=net,

                net_name='resnet50',
                exp_name='resnet50_extractor_static_imagenet_train2',

                num_epochs=100,
                learning_rate=0.001,
                learning_rate_decay=True,
                learning_rate_decay_factor=0.1,
                weight_decay=1e-4,
                # learning_rate_decay_epoch=[10,30],
                learning_rate_decay_epoch=[30,60],
                dataset_name='imagenet',
                batch_size=512,
                evaluate_step=1000,
                load_net=True,
                test_net=True,
                num_workers=8,

                target_accuracy=0.9264796074467897,
                top_acc=5,
                        data_parallel=True)
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

    # no_grad = name_parameters_no_grad(net, ['layer3.block8', 'fc'])
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


    i+=1
