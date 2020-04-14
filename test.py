import torch
from torch import nn
from framework import data_loader, train, evaluate,measure_flops
from network import vgg_channel_weight, vgg,storage,resnet,net_with_predicted_mask,resnet_cifar
from framework import config as conf
import os,sys
from filter_characteristic import filter_feature_extractor,predict_dead_filter
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

# s1=[5, 5, 8, 7, 5, 9, 5, 9, 7, 16, 13, 14, 16, 17, 15, 14, 12, 15, 14, 24, 28, 21, 22, 24, 21, 19, 18]
# s2=[5, 6, 5, 5, 7, 5, 5, 7, 5, 15, 14, 13, 16, 13, 13, 15, 17, 12, 19, 26, 28, 28, 30, 27, 31, 28, 20]
# x=np.arange(len(s1))
#
# total_width, n = 0.4, 2     # 有多少个类型，只需更改n即可
# width = total_width / n
# x = x - (total_width - width) / 2
#
# plt.figure(figsize=(6, 2))
# plt.bar(x,s1,label='seed1,acc=91.04%',width=width)
# plt.bar(x+width,s2,label='seed2,acc=91.95%',width=width)
# plt.legend()
# plt.show()












# device=torch.device('cuda')
# net=resnet_cifar.resnet32(num_classes=10).to(device)
# measure_flops.measure_model(net,dataset_name='cifar10')
# print()
# i=0
# # net = storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'/home/victorfang/model_pytorch/data/model_saved/resnet56_predictionmask_baseline/checkpoint/flop=125485706,accuracy=0.93650.tar')),pretrained=True)
# # for name, mod in net.named_modules():
# #     if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
# #         print(name,mod)
# #         i+=1
# a=[7,9,9,9,11,7,7,10,10,12,16,11,12,16,9,6,6,15,13,7   ,18,21,15,18,26, 17,25]
# a=[3+i for i in a]
# b=[16,     a[0],16,a[1],16,a[2],16,a[3],16,a[4],16,a[5],16,a[6],16,a[7],16,a[8],16,
#    a[9],32,a[10],32,a[11],32,a[12],32,a[13],32,a[14],32,a[15],32,a[16],32,a[17],32,
#    a[18],64,a[19],64,a[20],64,a[21],64,a[22],64,a[23],64,a[24],64,a[25],64,a[26],64]
# checkpoint=dict()
# checkpoint['structure']=b
# checkpoint['net_name']='resnet56'
# checkpoint['dataset_name']='cifar10'
# net=storage.restore_net(checkpoint,pretrained=False)
# measure_flops.measure_model(net,dataset_name='cifar10')
# print()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net=vgg.vgg16_bn(dataset_name='cifar10').to(device)
net = storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path, 'baseline/vgg16_bn_cifar10,accuracy=0.941.tar')),pretrained=False)

net = net_with_predicted_mask.predicted_mask_net(net, net_name='vgg16_bn', dataset_name='cifar10').to(device)

# a=torch.load('/home/victorfang/model_pytorch/data/model_saved/tmp/checkpoint/flop=313733786,accuracy=0.18720.tar')
# a=torch.load('/home/victorfang/test.tar')
# net.load_state_dict(a['state_dict'])
# net.load_state_dict(a)

# net = storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path, 'baseline/vgg16_bn_cifar10,accuracy=0.941.tar')),pretrained=True)

net.to(device)

# def cal_dead_times(module, input, output):
#     print()
#     module.train()
#     out1=module.forward(input[0])
#     module.eval()
#     out2=module.forward(input[0])
#
#     size=1
#     for s in out1.shape:
#         size*=s
#     same=torch.abs((out1-out2))<1
#     b=torch.sum(same)
#
#     if b!=s:
#         print(module)
#     print()



# for name,mod in net.named_modules():
#     # if isinstance(mod,torch.nn.BatchNorm2d):
#     #     mod.register_forward_hook(cal_dead_times)
#     if name=='net.features':
#     # if name == 'features':
#
#         mod.register_forward_hook(cal_dead_times)

# evaluate.evaluate_net(net,data_loader=data_loader.create_validation_loader(batch_size=1024,num_workers=2,dataset_name='cifar10'),save_net=False)
# train_loader=data_loader.create_validation_loader(batch_size=1024,num_workers=2,dataset_name='cifar10')
# for step, data in enumerate(train_loader, 0):
#     images, labels = data
#     images, labels = images.to(device), labels.to(device)
#     net(images)
#     break


net.train()
# #
# import cgd
#
train.train(net=net,
            net_name='vgg16_bn',
            exp_name='vgg16bn_predicted_0.8_1',
            dataset_name='cifar10',
            # optimizer=cgd.CGD,
            optimizer=optim.SGD,
            weight_decay=0,
            momentum=0,

            learning_rate=0.1,
            num_epochs=350,
            batch_size=2048,
            evaluate_step=5000,
            load_net=False,
            test_net=False,
            num_workers=8,
            # weight_decay=5e-4,
            learning_rate_decay=True,
            learning_rate_decay_epoch=[100,250],
            learning_rate_decay_factor=0.1,
            scheduler_name='MultiStepLR',
            top_acc=1,
            data_parallel=False,
            paint_loss=True,
            # no_grad=no_grad,
            )