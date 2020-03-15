import torch
from torch import nn
from framework import data_loader, train, evaluate,measure_flops
from network import vgg_channel_weight, vgg,storage,resnet,net_with_predicted_mask,resnet_cifar
from framework import config as conf
import os,sys
from filter_characteristic import filter_feature_extractor,predict_dead_filter
import numpy as np
from torch import optim
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'baseline/resnet56_cifar10,accuracy=0.94230.tar')))
tran_net = net_with_predicted_mask.predicted_mask_net(net, net_name='resnet56', dataset_name='cifar10').to(device)
# measure_flops.measure_model(tran_net)
# tran_net = nn.DataParallel(tran_net)


train.train(net=tran_net,
            net_name='resnet56',
            exp_name='resnet56_predictionmask_baseline',
            dataset_name='cifar10',
            learning_rate=0.001,
            num_epochs=350,
            batch_size=512,
            evaluate_step=500,
            load_net=False,
            test_net=False,
            num_workers=8,
            # weight_decay=5e-4,
            optimizer=optim.Adam,
            top_acc=1,
            data_parallel=False)