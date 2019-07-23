import torch
import train
import config as conf
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import math
import prune_and_train
import measure_flops
import evaluate
import numpy as np
import data_loader
from sklearn import svm
import vgg
import predict_dead_filter
from predict_dead_filter import fc
import prune
import generate_random_data
import resnet

def tmp(index_real,index):
    for i in range(len(index_real)):
        hit=0
        for ind in index_real[i]:
            if ind in index[i]:
                hit+=1
        print('in layer {}, number of true answer is {}. {} of total {} predictions are correct.'.format(i,len(index_real[i]),hit,len(index[i])))

checkpoint=torch.load('/home/victorfang/PycharmProjects/model_pytorch/baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
net=checkpoint['net']


df_index,_,_=evaluate.find_useless_filters_data_version(net=net,filter_dead_ratio=0.9,batch_size=800,neural_dead_times=9000,use_random_data=False)

conv_list,neural_list=evaluate.check_conv_alive_layerwise(net=net,neural_dead_times=800,batch_size=800)
df_index_random_data_conv,_,_=evaluate.find_useless_filters_data_version(net=net,filter_dead_ratio=0.9,batch_size=800,neural_dead_times=800,module_list=conv_list,neural_list=neural_list)
tmp(df_index,df_index_random_data_conv)

print('--------------------------------------------')

relu_list,neural_list=evaluate.check_ReLU_alive(net=net,neural_dead_times=800,data=generate_random_data.random_normal(num=800,dataset_name='cifar10'))
df_index_random_data_relu,_,_=evaluate.find_useless_filters_data_version(net=net,filter_dead_ratio=0.9,batch_size=800,neural_dead_times=800,module_list=relu_list,neural_list=neural_list)

tmp(df_index,df_index_random_data_relu)



num_conv = 0  # num of conv layers in the net
dead_filter_index=list()
for mod in net.features:
    if isinstance(mod, torch.nn.modules.conv.Conv2d):
        num_conv += 1
for i in range(num_conv):
    df_index,_,_=evaluate.find_useless_filters_data_version(net=net,filter_dead_ratio=0.9,batch_size=800,neural_dead_times=800,use_random_data=True)
    dead_filter_index.append(df_index[i])
    net = prune.prune_conv_layer(model=net, layer_index=i + 1,
                                 filter_index=df_index[i])  # prune the dead filter




df_val, lf_val = predict_dead_filter.read_data(balance=True,
                           path='/home/victorfang/Desktop/pytorch_model/vgg16bn_cifar10_dead_neural_normal_tar_acc_decent3/dead_neural',neural_dead_times=1200)

stat_df_val = predict_dead_filter.statistics(df_val)
stat_lf_val = predict_dead_filter.statistics(lf_val)

val_x = np.vstack((stat_df_val, stat_lf_val))
val_y = np.zeros(val_x.shape[0], dtype=np.int)
val_y[:stat_df_val.shape[0]] = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# validation data
val_x_tensor = torch.tensor(val_x, dtype=torch.float32).to(device)
val_y_tensor = torch.tensor(val_y, dtype=torch.long).to(device)

checkpoint=torch.load('/home/victorfang/Desktop/预测死亡神经元的神经网络/accuracy=0.72233.tar')
net=checkpoint['net']
net.load_state_dict(checkpoint['state_dict'])
output=net(val_x_tensor)
prediction=torch.argmax(output,1)
correct=(prediction==val_y_tensor).sum().float()
acc=correct.cpu().detach().data.numpy()/val_y_tensor.shape[0]
print()

# net = vgg.vgg16_bn(pretrained=False).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# data_loader = data_loader.create_validation_loader(dataset_name='cifar10',default_image_size=32,
#                                                    batch_size=72, mean=conf.cifar10['mean'], std=conf.cifar10['std'],
#                                                    num_workers=conf.num_workers)
# evaluate.evaluate_net(net=net, data_loader=data_loader, save_net=False)

# net = vgg.vgg16_bn(pretrained=True)
# net.classifier = nn.Sequential(
#     nn.Dropout(),
#     nn.Linear(512, 512),
#     nn.ReLU(True),
#     nn.Dropout(),
#     nn.Linear(512, 512),
#     nn.ReLU(True),
#     nn.Linear(512, 10),
# )
# for m in net.modules():
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, 0, 0.01)
#         nn.init.constant_(m.bias, 0)
# net = net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#
# # checkpoint = torch.load('/home/victorfang/Desktop/vgg16_bn_cifar10,accuracy=0.941.tar')
# checkpoint = torch.load(
#     '/home/victorfang/Desktop/vgg16_bn_cifar10,accuracy=0.941.tar')
#
# net = checkpoint['net']
# net.load_state_dict(checkpoint['state_dict'])
# print(checkpoint['highest_accuracy'])
# relu_list,neural_list=evaluate.check_ReLU_alive(net=net,
#                           data_loader=data_loader.create_validation_loader(dataset_path=conf.cifar10['validation_set_path'],
#                                                                  default_image_size=32,
#                                                                  mean=conf.cifar10['mean'],
#                                                                  std=conf.cifar10['std'],
#                                                                  batch_size=1024,
#                                                                  num_workers=4,
#                                                                  dataset_name='cifar10',
#                                                                  ),
#                           neural_dead_times=8000)










# print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# checkpoint=torch.load('/home/victorfang/Desktop/vgg19_imagenet_deadReLU.tar')
# relu_list=checkpoint['relu_list']
# neural_list=checkpoint['neural_list']
# net=checkpoint['net']
#
#
# evaluate.plot_dead_filter_statistics(net,relu_list,neural_list,40000,1)
# print()


