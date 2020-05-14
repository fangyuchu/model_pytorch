import torch
from torch import nn
from framework import data_loader, train, evaluate,measure_flops
from network import vgg_channel_weight, vgg,storage,resnet,net_with_predicted_mask,resnet_cifar,modules
from framework import config as conf
import os,sys
from filter_characteristic import filter_feature_extractor,predict_dead_filter
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import cgd
#ssh -L 16006:127.0.0.1:6006 -p 20029 victorfang@210.28.133.13
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#
# a=resnet_cifar.resnet56().to(device)
# # train.add_forward_hook(torch.nn.Conv2d,a)
# b=resnet.resnet50().to(device)
# # train.add_forward_hook(torch.nn.Conv2d,b)
# c=vgg.vgg16_bn(dataset_name='cifar10').to(device)
# # train.add_forward_hook(torch.nn.Conv2d,c)
# # train.add_forward_hook(nn.BatchNorm1d,c)
# # net=net_with_mask_shortcut.mask_and_shortcut_net(c,'vgg16_bn','cifar10').to(device)
# net=net_with_mask_shortcut.mask_and_shortcut_net(a,'resnet56','cifar10').to(device)
# # net.load_state_dict(torch.load('/home/victorfang/model_pytorch/data/model_saved/tmp/checkpoint/flop=261822474,accuracy=0.92380.tar')['state_dict'])
# net=a
# net.to(device)
# train.train(net=net,
#             net_name='resnet56',
#             exp_name='resnet56_baseline_newversion',
#             dataset_name='cifar10',
#             # optimizer=cgd.CGD,
#             optimizer=optim.SGD,
#             # weight_decay={'default':1e-6,'extractor':0},
#             # momentum={'default':0.9,'extractor':0},
#             weight_decay=1e-4,
#             momentum=0.9,
#
#             learning_rate=0.1,
#             num_epochs=160,
#             batch_size=128,
#             evaluate_step=5000,
#             load_net=False,
#             test_net=True,
#             num_workers=8,
#             # weight_decay=5e-4,
#             learning_rate_decay=True,
#             learning_rate_decay_epoch=[80,120],
#             learning_rate_decay_factor=0.1,
#             scheduler_name='MultiStepLR',
#             top_acc=1,
#             data_parallel=False,
#             paint_loss=False,
#             )
# print()



# net=vgg.vgg16_bn(dataset_name='cifar10')
# b=net.parameters()
#
# a=nn.Parameter(torch.ones(2))
# c=nn.Parameter(torch.Tensor([1,2]))
# opt=optim.SGD([a,c],lr=0.1)
#
# b=a+c
# b=torch.sum(b)
# b.backward()
# opt.step()
# print()

















mask_update_freq = 4000
mask_update_steps = 400
# net=vgg.vgg16_bn(dataset_name='cifar10')
# net = net_with_predicted_mask.predicted_mask_shortcut_with_weight_net(net,
#                                                  net_name='vgg16_bn',
#                                                  dataset_name='cifar10',
#                                                  mask_update_steps=mask_update_steps,
#                                                  mask_update_freq=mask_update_freq,
#                                                               flop_expected=12e7)


net=resnet_cifar.resnet56()
net = net_with_predicted_mask.predicted_mask_shortcut_with_weight_net(net,
                                                 net_name='resnet56',
                                                 dataset_name='cifar10',
                                                 mask_update_steps=mask_update_steps,
                                                 mask_update_freq=mask_update_freq,
                                                              flop_expected=5e7,
                                                                      gcn_rounds=1)
# checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_mask_shortcut_weight_extractorlr0.01_shortcut_in_gcn/checkpoint/flop=50133642,accuracy=0.88030.tar')
# checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_mask_shortcut_weight_extractorlr0.01_shortcut_convshortcut_in_gcn/checkpoint/flop=50078346,accuracy=0.50720.tar')

# net.load_state_dict(checkpoint['state_dict'])


# folder='resnet56_mask_shortcut_weight_bn_inextractor_tanh_gradclip_testrun'
# images=torch.load('/home/victorfang/model_pytorch/data/model_saved/'+folder+'/crash/images.pt')
# labels=torch.load('/home/victorfang/model_pytorch/data/model_saved/'+folder+'/crash/labels.pt')
# state_dict=torch.load('/home/victorfang/model_pytorch/data/model_saved/'+folder+'/crash/net.pt')
# loss_saved=torch.load('/home/victorfang/model_pytorch/data/model_saved/'+folder+'/crash/loss.pt')
# outputs_saved=torch.load('/home/victorfang/model_pytorch/data/model_saved/'+folder+'/crash/outputs.pt')
# net.load_state_dict(state_dict)
# net.to(device)
# net.train()
# net.eval()
# # net.mask_update_steps=5
# # train.add_forward_hook(net,module_name='net.layer2')
# train.add_forward_hook(net,module_name='net.layer3.block0.conv2')
#
# # train.add_backward_hook(net,module_name='net.layer2.block0.conv2.bn')
#
# outputs=net(images)
# criterion=nn.CrossEntropyLoss()
# loss = criterion(outputs, labels)
# loss.backward()
#
# l=list(net.named_modules())
# l.reverse()
# for name,mod in l:
#     if hasattr(mod,'weight'):
#         print(name, mod.weight.detach().cpu().numpy().max())
#         print(name, mod.weight.grad.detach().cpu().numpy().max())
#         if np.abs(mod.weight.detach().cpu().numpy()).max() > 1000:
#             print(name, mod.weight.detach().cpu().numpy().max())
#             raise Exception('where is this grad came from?')
#         if np.abs(mod.weight.grad.detach().cpu().numpy()).max() > 100:
#             print(name, mod.weight.grad.detach().cpu().numpy().max())
#             raise Exception('what is this grad?')
#
# for name,mod in net.named_modules():
#     if isinstance(mod,nn.BatchNorm2d):
#         print(name)
#         mod.momentum=0.5
#
# # evaluate.evaluate_net(net,data_loader=data_loader.create_validation_loader(batch_size=128,num_workers=0,dataset_name='cifar10'),save_net=False)
# print()


#
# net.detach_mask()
# net.print_mask()
# net=net.net
# c=torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_mask_shortcut_weight_basenet/checkpoint/flop=125797002,accuracy=0.91220.tar')
# net.load_state_dict(c['state_dict'])
# for name,mod in net.named_modules():
#     if hasattr(mod,'weight'):
#         print(name, mod.weight.detach().cpu().numpy().max())
net.to(device)
# print()
# measure_flops.measure_model(net,dataset_name='cifar10')

# evaluate.evaluate_net(net,data_loader=data_loader.create_train_loader(batch_size=128,num_workers=0,dataset_name='cifar10'),save_net=False)

# train.train(net=net,
#             net_name='resnet56',
#             exp_name='tmp',
#             dataset_name='cifar10',
#             # optimizer=cgd.CGD,
#             optimizer=optim.SGD,
#             weight_decay=1e-4,
#             momentum=0.9,
#             # weight_decay=0,
#             # momentum=0,
#
#             learning_rate=0.1,
#             num_epochs=160,
#             batch_size=128,
#             evaluate_step=1600,
#             load_net=False,
#             test_net=False,
#             num_workers=0,
#             # weight_decay=5e-4,
#             learning_rate_decay=True,
#             learning_rate_decay_epoch=[80,120],
#             learning_rate_decay_factor=0.1,
#             scheduler_name='MultiStepLR',
#             top_acc=1,
#             data_parallel=False,
#             paint_loss=True,
#             save_at_each_step=True
#             )

print()
# print()
#
# net=resnet.resnet50()
# net = net_with_predicted_mask.predicted_mask_shortcut_with_weight_net(net,
#                                                  net_name='resnet50',
#                                                  dataset_name='imagenet',
#                                                  mask_update_steps=mask_update_steps,
#                                                  mask_update_freq=mask_update_freq,
#                                                               flop_expected=5e7)

# print()

# net.update_mask()
# print()


# measure_flops.measure_model(net,'cifar10')


# net.eval()

# evaluate.evaluate_net(net,data_loader=data_loader.create_train_loader(batch_size=1024,num_workers=2,dataset_name='cifar10'),save_net=False)
# train_loader=data_loader.create_validation_loader(batch_size=1024,num_workers=2,dataset_name='cifar10')
# for step, data in enumerate(train_loader, 0):
#     images, labels = data
#     images, labels = images.to(device), labels.to(device)
#     net(images)
#     break


# net.train()
# #
# import cgd
#
# net=vgg.vgg16_bn(dataset_name='cifar10').to(device)
train.train(net=net,
            net_name='resnet56',
            exp_name='tmp',
            dataset_name='cifar10',
            # optimizer=cgd.CGD,
            optimizer=optim.SGD,
            weight_decay={'default':1e-4,'extractor':0},
            momentum={'default':0.9,'extractor':0},
            # weight_decay=0,
            # momentum=0,

            learning_rate={'default': 0.1, 'extractor': 0.1},
            num_epochs=350,
            batch_size=128,
            evaluate_step=1600,
            load_net=False,
            test_net=False,
            num_workers=8,
            # weight_decay=5e-4,
            learning_rate_decay=True,
            learning_rate_decay_epoch=[100,200],
            learning_rate_decay_factor=0.1,
            scheduler_name='MultiStepLR',
            top_acc=1,
            data_parallel=False,
            paint_loss=True,
            )