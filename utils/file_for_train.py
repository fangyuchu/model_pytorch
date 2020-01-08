import torch
from torch import nn
import torch.optim as optim
from prune import prune_and_train,prune_and_train_with_mask
from framework import evaluate,data_loader,measure_flops,train
from network import create_net,net_with_mask,vgg,storage
from framework import config as conf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# os.environ["CUDA_VISIBLE_DEVICES"] = '3,4,5,6'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16_extractor_static_cifar10/checkpoint/flop=43303574,accuracy=0.92800.tar'),pretrained=True)
# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16bn_cifar100_extractor_static/checkpoint/flop=134625568,accuracy=0.71610.tar'),pretrained=True)
# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_extractor_static_cifar10_only_gcn/checkpoint/flop=62982794,accuracy=0.93400.tar'),pretrained=True)
net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_extractor_static_cifar100_2/checkpoint/flop=104589668,accuracy=0.70420.tar'),pretrained=True)


# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet18_tinyimagenet_extractor_static_train/checkpoint/flop=1289487816,accuracy=0.67650.tar'),pretrained=True)
# net=torch.nn.DataParallel(net)
i=0
success=False
while not success and i<4:
    success=train.train(net=net,
                net_name='resnet56',
                exp_name='resnet56_extractor_static_cifar100_2'+'_train',

                        optimizer=optim.SGD,
                        learning_rate=0.01,
                        learning_rate_decay=True,
                        learning_rate_decay_factor=0.5,
                        learning_rate_decay_epoch=[20, 50, 100, 150, 250, 300, 350, 400],
                        weight_decay=5e-4,
                        num_epochs=450,


                dataset_name='cifar100',
                # num_epochs=450,
                batch_size=512,
                evaluate_step=3000,
                load_net=False,
                test_net=True,
                num_workers=2,

                target_accuracy=0.7123446598422201,
                top_acc=1)


    # success=train.train(net=net,
    #             net_name='vgg16_bn',
    #             exp_name='vgg16bn_cifar100_extractor_static'+'_train',
    #
    #          optimizer=optim.SGD,
    #
    #                     num_epochs=200,
    #                     learning_rate=0.0001,
    #                     learning_rate_decay_epoch=[60, 120, 160],
    #                     learning_rate_decay_factor=0.2,
    #                     weight_decay=5e-3,
    #          # learning_rate=0.001,  # 标准baseline
    #          # learning_rate_decay=True,
    #          # learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
    #          # learning_rate_decay_factor=0.5,
    #          # weight_decay=5e-3,
    #
    #             dataset_name='cifar100',
    #             # num_epochs=450,
    #             batch_size=512,
    #             evaluate_step=3000,
    #             load_net=False,
    #             test_net=True,
    #             num_workers=2,
    #
    #             target_accuracy=0.719892441834323,
    #             top_acc=1)


    # success=train.train(net=net,
    #
    #             net_name='resnet56',
    #             exp_name='resnet56_extractor_static_cifar10_only_gcn'+'_train',
    #
    #          optimizer=optim.SGD,
    #          learning_rate=0.0001,  # 标准baseline
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
    #             top_acc=1)

    # success=train.train(net=net,
    #
    #             net_name='vgg16_bn',
    #             exp_name='vgg16_extractor_static_cifar10'+'_train',
    #
    #          optimizer=optim.SGD,
    #          learning_rate=0.001,  # 标准baseline
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
    #             target_accuracy=0.932,
    #             top_acc=1)


    # success = train.train(net=net,
    #                       net_name='resnet18',
    #                       exp_name='resnet18_tinyimagenet_extractor_static' + '_train',
    #
    #                       dataset_name='tiny_imagenet',
    #                       num_epochs=40,
    #                       batch_size=512,
    #                       evaluate_step=100,
    #                       load_net=False,
    #                       test_net=True,
    #                       num_workers=8,
    #                       learning_rate_decay=True,
    #                       learning_rate=0.001,
    #                       learning_rate_decay_factor=0.1,
    #                       learning_rate_decay_epoch=[7, 14, 21, 28, 35],
    #
    #                       weight_decay=1e-3,
    #                       target_accuracy=0.7035099414949139,
    #                       optimizer=optim.SGD,
    #                       top_acc=1)
    i+=1
