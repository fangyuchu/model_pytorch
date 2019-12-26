import torch
from torch import nn
from framework import data_loader, train, evaluate
from network import vgg_channel_weight, vgg,storage
from framework import evaluate,data_loader
from framework import config as conf
import os,sys
from filter_characteristic import filter_feature_extractor,predict_dead_filter
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#
# c=torch.load('/home/disk_new/model_saved/vgg16_bn_weighted_channel/checkpoint/flop=18923530,accuracy=0.93600.tar')
#
# network=c['network']
# network.load_state_dict(c['state_dict'])
# for mod in network.features:
#     if isinstance(mod,nn.Conv2d):
#         print()

net_file_name=['']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# file_list = os.listdir('/home/victorfang/model_pytorch/data/model_saved/vgg16_extractor_static_cifar10/checkpoint')
# file_list.sort()
extractor = filter_feature_extractor.load_extractor(
    '/home/victorfang/model_pytorch/data/model_saved/vgg16_extractor_static_cifar10_more_train/extractor/100.tar')
extractor.eval()
i=0

# for file_name in file_list:
# print(file_name)
# checkpoint=torch.load(os.path.join('/home/victorfang/model_pytorch/data/model_saved/vgg16_extractor_static_cifar10/checkpoint/',file_name))
checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16_extractor_static_cifar10_more_train/checkpoint/flop=160463562,accuracy=0.93700.tar')
net=storage.restore_net(checkpoint,pretrained=True)

useless_filter_index,module_list,neural_list,FIRE=evaluate.find_useless_filters_data_version(net=net,batch_size=16,percent_of_inactive_filter=0.1,dataset_name='cifar10')

# checkpoint = {'prune_rate': 0.1, 'module_list': module_list,
#               'neural_list': neural_list, 'state_dict': net.state_dict(),
#               'num_test_images': 10000}
# checkpoint.update(storage.get_net_information(net, 'cifar10', 'vgg16_bn'))
# torch.save(checkpoint,
#            os.path.join('/home/victorfang/model_pytorch/data/model_saved/vgg16_extractor_static_cifar10', 'all_Fire',file_name))
# i+=1



label = torch.Tensor(FIRE).reshape((-1, 1)).to(device)

Fire_e=extractor.forward(net,'vgg16_bn','cifar10')
criterion=torch.nn.L1Loss()
loss=criterion(label,Fire_e)
print(float(loss))

Fire_e=Fire_e.view(-1).detach().cpu().numpy()
predict_dead_filter.performance_evaluation(np.array(FIRE),Fire_e,0.1)
print('\n\n\n\n\n')




















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
