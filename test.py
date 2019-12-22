import torch
from torch import nn
from framework import data_loader, train, evaluate
from network import vgg_channel_weight, vgg,storage
from framework import evaluate,data_loader
from framework import config as conf
import os,sys
#
# c=torch.load('/home/disk_new/model_saved/vgg16_bn_weighted_channel/checkpoint/flop=18923530,accuracy=0.93600.tar')
#
# network=c['network']
# network.load_state_dict(c['state_dict'])
# for mod in network.features:
#     if isinstance(mod,nn.Conv2d):
#         print()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet50_pureData/checkpoint/flop=3496274525,accuracy=0.74196.tar')
net=storage.restore_net(checkpoint,pretrained=True)
net=torch.nn.DataParallel(net)
# evaluate.evaluate_net(net=net,data_loader=data_loader.create_validation_loader(batch_size=512,num_workers=8,dataset_name='imagenet'),save_net=False)
train.train(net=net,
            net_name='resnet50',
            exp_name='resnet50_pureData_train',
            learning_rate=0.001,
            num_epochs=10,
            batch_size=512,
            test_net=True,
            num_workers=8,
            learning_rate_decay=True,
            learning_rate_decay_factor=10,
            learning_rate_decay_epoch=[5],
            weight_decay=5e-4,
            momentum=0.9,
            target_accuracy=0.743903061255542,
            optimizer=torch.optim.SGD
            )




















# checkpoint=torch.load('/home/victorfang/model_pytorch/data/baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
# net=storage.restore_net(checkpoint)
# for i in range(10):
#
#     module_list, neural_list = evaluate.check_ReLU_alive(net=net,
#                                                          data_loader=data_loader.create_validation_loader(batch_size=16,
#                                                                                                                 num_workers=8,
#                                                                                                                 dataset_name='cifar10'+'_trainset',
#                                                                                                                 shuffle=True,),
#                                                         neural_dead_times=10240,
#                                                          max_data_to_test=10000)
#     checkpoint = {'module_list': module_list,
#                   'neural_list': neural_list, 'state_dict': net.state_dict(),
#                   }
#     checkpoint.update(storage.get_net_information(net, 'cifar10', 'vgg16_bn'))
#     torch.save(checkpoint,
#                os.path.join(conf.root_path,'model_saved/'   , 'random_net/%d.tar' % i) )
#     layer=0
#     for mod in net.modules():
#         if isinstance(mod,torch.nn.Conv2d):
#             if layer==i:
#                 nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
#                 if mod.bias is not None:
#                     nn.init.constant_(mod.bias, 0)
#             layer+=1
