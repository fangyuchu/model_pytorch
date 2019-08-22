import train
import vgg
import torch.nn as nn
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import config as conf
import torch.optim as optim
import logger
import sys
import resnet_copied
import data_loader
# if not os.path.exists('./model_saved/vgg16_on_cifar100'):
#     os.makedirs('./model_saved/vgg16_on_cifar100', exist_ok=True)
# sys.stdout = logger.Logger( './model_saved/vgg16_on_cifar100/log.txt', sys.stdout)
# sys.stderr = logger.Logger( './model_saved/vgg16_on_cifar100/log.txt', sys.stderr)  # redirect std err, if necessary
#
#
#
net=vgg.vgg16(pretrained=False)
net.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 100),
        )
for m in net.modules():
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

net=net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# checkpoint=torch.load('/home/error/fang/model_pytorch/model_saved/vgg16_on_cifar1000/checkpoint/flop=332488804,accuracy=0.25550.tar')
# net.load_state_dict(checkpoint['state_dict'])

for i in range(3,10):
    train.train(net,
                'vgg16_on_cifar100'+str(i),
                'cifar100',
                load_net=True,
                batch_size=1600,
                checkpoint_step=8000,
                root_path='./model_saved/',
                num_workers=8,
                num_epochs=450,
                # optimizer=optim.SGD,
                # learning_rate=0.1,
                # momentum=0.9,
                # weight_decay=0.0015,
                # learning_rate_decay=True,
                learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
                learning_rate_decay_factor=0.5,

                optimizer=optim.Adam,
                learning_rate=1e-4,
                weight_decay=1e-8,
                learning_rate_decay=False,
                )
#
#
#
# train.train(net,
#             'vgg16_on_cifar100',
#             'cifar100',
#             load_net=True,
#             batch_size=1600,
#             checkpoint_step=8000,
#             root_path='./model_saved/',
#             num_workers=8,
#             weight_decay=0,
#             num_epochs=450,
#             optimizer=optim.SGD,
#             learning_rate=0.01,
#             learning_rate_decay=True,
#             learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
#             learning_rate_decay_factor=0.5,
#             )


# if not os.path.exists('./model_saved/resnet56_on_cifar100'):
#     os.makedirs('./model_saved/resnet56_on_cifar100', exist_ok=True)
# sys.stdout = logger.Logger( './model_saved/resnet56_on_cifar100/log.txt', sys.stdout)
# sys.stderr = logger.Logger( './model_saved/resnet56_on_cifar100/log.txt', sys.stderr)  # redirect std err, if necessary
# net=resnet_copied.ResNet(resnet_copied.BasicBlock, [9, 9, 9],num_classes=100)
# net=net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# train.train(net,
#             'resnet56_on_cifar100',
#             'cifar100',
#             load_net=True,
#             batch_size=1600,
#             checkpoint_step=8000,
#             root_path='./model_saved/',
#             num_workers=8,
#             weight_decay=0,
#             num_epochs=450,
#             optimizer=optim.SGD,
#             learning_rate=0.1,
#             learning_rate_decay=True,
#             learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
#             learning_rate_decay_factor=0.5,
#             )




# net=vgg.vgg16_bn(pretrained=False)
#
# net = net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# m1=nn.Linear(4096,200)
# nn.init.normal_(m1.weight,0,0.01)
# nn.init.constant_(m1.bias,0)
# net.classifier[6]=m1
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net.to(device)
# batch_size=32
# num_worker=8
# train_loader=data_loader.create_train_loader(batch_size=batch_size,
#                                              num_workers=num_worker,
#                                              dataset_name='tiny_imagenet',
#                                              default_image_size=224,
#                                              )
# validation_loader=data_loader.create_validation_loader(batch_size=batch_size,
#                                                        num_workers=num_worker,
#                                                        dataset_name='tiny_imagenet',
#                                                        default_image_size=224
#                                                     )
# for i in range(10):
#     print(i)
#     train.train(net=net,
#                 net_name='vgg16bn_tiny_imagenet' + str(i),
#                 dataset_name='tiny_imagenet',
#                 test_net=False,
#                 # optimizer=optim.SGD,
#                 # learning_rate=0.1,
#                 # learning_rate_decay=True,
#                 # learning_rate_decay_epoch=[ 30, 60, 600],
#                 # learning_rate_decay_factor=0.1,
#                 # weight_decay=0.0006,
#
#                 optimizer=optim.Adam,
#                 learning_rate=1e-3,
#                 weight_decay=1e-8,
#                 learning_rate_decay=False,
#
#                 load_net=True,
#                 batch_size=batch_size,
#                 num_epochs=1000,
#                 train_loader=train_loader,
#                 validation_loader=validation_loader,
#                 checkpoint_step=1000,
#
#                  )