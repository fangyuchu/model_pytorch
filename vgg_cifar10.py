import train
import vgg
import torch.nn as nn
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import config as conf

net=vgg.vgg16_bn(pretrained=True)
net.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
for m in net.modules():
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

net=net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


batch_size=1024
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=batch_size, shuffle=True,
    num_workers=6, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=6, pin_memory=True)

train.train(net,
            'vgg16_bn_on_cifar-10',
            'cifar10',
            train_loader=train_loader,
            validation_loader=val_loader,
            learning_rate=0.0001,
            num_epochs=1000,
            batch_size=batch_size,
            checkpoint_step=800,
            root_path='./model/',
            num_workers=8,
            )



# checkpoint_path='./model_best.pth.tar'
# if os.path.isfile(checkpoint_path):
#     print("=> loading checkpoint '{}'".format(checkpoint_path))
#     checkpoint = torch.load(checkpoint_path)
#     start_epoch = checkpoint['epoch']
#     best_prec1 = checkpoint['best_prec1']
#
#     module_name=list(checkpoint['state_dict'].keys())
#     # for i in range(len(module_name)):
#     #     module_name[i]=module_name[i].replace('.module','')
#     for module in module_name:
#         checkpoint['state_dict'][module.replace('.module','')]=checkpoint['state_dict'].pop(module)
#     net.load_state_dict(checkpoint['state_dict'])
#     print("loaded checkpoint")
# else:
#     print("=> no checkpoint found at '{}'".format(checkpoint_path))
