from framework import train
import os
import torch
import torch.optim as optim
import logger
import sys
from network import resnet_cifar

# network=vgg.vgg16(pretrained=True)
# print('haha')
# network.classifier=nn.Sequential(
#             nn.Linear(512, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 100),
#         )
# for m in network.modules():
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, 0, 0.01)
#         nn.init.constant_(m.bias, 0)
#
# checkpoint=torch.load('/home/zzj/fang/model_pytorch/model_saved/vgg16_on_cifar100/checkpoint/flop=332488804,accuracy=0.70970.tar')
# network.load_state_dict(checkpoint['state_dict'])
# network=network.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# train.train(network,
#             'vgg16_on_cifar100_2',
#             'cifar100',
#             load_net=True,
#             test_net=True,
#             batch_size=128,
#             checkpoint_step=8000,
#             root_path='../data/model_saved/',
#             num_workers=8,
#             num_epochs=200,
#             optimizer=optim.SGD,
#             # learning_rate=0.1,    #标准baseline
#             learning_rate=0.001,
#             learning_rate_decay=True,
#             learning_rate_decay_epoch=[60,120,160],#, 150, 250, 300, 350, 400],
#             learning_rate_decay_factor=0.2,
#             weight_decay=5e-4,
#             momentum=0.9,
#             )



if not os.path.exists('../data/model_saved/resnet56_on_cifar100'):
    os.makedirs('../data/model_saved/resnet56_on_cifar100', exist_ok=True)
sys.stdout = logger.Logger( '../data/model_saved/resnet56_on_cifar100/log.txt', sys.stdout)
sys.stderr = logger.Logger( '../data/model_saved/resnet56_on_cifar100/log.txt', sys.stderr)  # redirect std err, if necessary
net= resnet_cifar.ResNet(resnet_cifar.BasicBlock, [9, 9, 9], num_classes=100)
net=net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# checkpoint=torch.load('/home/zengyao/fang/model_pytorch/model_saved/resnet56_on_cifar1007/checkpoint/flop=125491556,accuracy=0.65020.tar')
# network.load_state_dict(checkpoint['state_dict'])

for i in range(1,15):
    train.train(net,
                'resnet56_on_cifar100' + str(i),
                'cifar100',
                load_net=True,
                test_net=True,
                batch_size=128,
                checkpoint_step=8000,
                root_path='../data/model_saved/',
                num_workers=8,
                num_epochs=200,
                optimizer=optim.SGD,
                learning_rate=0.1,
                learning_rate_decay=True,
                learning_rate_decay_epoch=[80, 120],  #, 150, 250, 300, 350, 400],
                learning_rate_decay_factor=0.1,
                weight_decay=1e-4,
                momentum=0.9,
                )