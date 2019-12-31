import torch
from torch import nn
import torch.optim as optim
from prune import prune_and_train,prune_and_train_with_mask
from framework import evaluate,data_loader,measure_flops,train
from network import create_net,net_with_mask,vgg,storage
from framework import config as conf
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16_extractor_static_cifar10/checkpoint/flop=81514058,accuracy=0.93410.tar'),pretrained=True)
# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_extractor_static_cifar10/checkpoint/flop=73820810,accuracy=0.93430.tar'),pretrained=True)
i=0
success=False
while not success and i<4:
    success=train.train(net=net,
                net_name='vgg16_bn',
                exp_name='vgg16_extractor_static_cifar10_train'+str(i),
                    # net_name='resnet56',
                    #     exp_name='resnet56_extractor_static_cifar10_train'+str(i),



                dataset_name='cifar10',
                num_epochs=450,
                batch_size=512,
                evaluate_step=3000,
                load_net=True,
                test_net=True,
                num_workers=8,
                learning_rate_decay=True,
                learning_rate=0.001,
                learning_rate_decay_factor=0.5,
                learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
                # learning_rate=0.01,
                # learning_rate_decay_factor=0.5,
                # learning_rate_decay_epoch=[10, 50, 100, 150, 250, 300, 350, 400],
                weight_decay=5e-4,
                target_accuracy=0.93465,
                optimizer=optim.SGD,
                top_acc=1)
    i+=1
