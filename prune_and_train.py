import config as conf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import resnet
import vgg
import os
import re
from datetime import datetime
from prune import select_and_prune_filter
import train
import evaluate


#todo:太老了，有时间改吧


def prune_and_train(
                    model_name,
                    pretrained=False,
                    dataset_name='imagenet',
                    learning_rate=conf.learning_rate,
                    num_epochs=conf.num_epochs,
                    batch_size=conf.batch_size,
                    checkpoint_step=conf.checkpoint_step,
                    checkpoint_path=None,
                    highest_accuracy_path=None,
                    sample_num_path=None,
                    default_image_size=224,
                    momentum=conf.momentum,
                    num_workers=conf.num_workers,
                    percent_of_pruning=0.3,
                    ord=2,
                  ):
    #implemented according to "Pruning Filters For Efficient ConvNets" by Hao Li
    # gpu or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ',end='')
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


    temp=re.search(r'(\d+)',model_name).span()[0]
    model=model_name[:temp]                                                     #name of the model.ex: vgg,resnet...
    del temp
    #define the model
    net=getattr(globals()[model],model_name)(pretrained=pretrained).to(device)

    num_conv=0                                                                  #num of conv layers in the net
    for mod in net.features:
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv+=1

    for i in range(1,num_conv+1):
        net = select_and_prune_filter(net, layer_index=i, percent_of_pruning=percent_of_pruning, ord=ord)  # prune the model

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=learning_rate
                          )  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    #prepare the data
    if dataset_name is 'imagenet':
        mean=conf.imagenet['mean']
        std=conf.imagenet['std']
        train_set_path=conf.imagenet['train_set_path']
        train_set_size=conf.imagenet['train_set_size']
        validation_set_path=conf.imagenet['validation_set_path']
    # Data loading code
    transform = transforms.Compose([
        transforms.RandomResizedCrop(default_image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std),
    ])
    train = datasets.ImageFolder(train_set_path, transform)
    val = datasets.ImageFolder(validation_set_path, transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if checkpoint_path is None:
        checkpoint_path=conf.root_path+model_name+','+str(percent_of_pruning)+'pruned'+'/checkpoint'
    if highest_accuracy_path is None:
        highest_accuracy_path=conf.root_path+model_name+','+str(percent_of_pruning)+'pruned'+'/highest_accuracy.txt'
    if sample_num_path is None:
        sample_num_path=conf.root_path+model_name+','+str(percent_of_pruning)+'pruned'+'/sample_num.txt'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path,exist_ok=True)

    if  os.path.exists(highest_accuracy_path):
        f = open(highest_accuracy_path, 'r')
        highest_accuracy = float(f.read())
        f.close()
        print('highest accuracy from previous training is %f' % highest_accuracy)
        del highest_accuracy

    sample_num=0
    if os.path.exists(sample_num_path):
        f = open(sample_num_path, 'r')
        sample_num = int(f.read())
        f.close()
        print('sample_num at present is %d' % sample_num)
        model_saved_at=checkpoint_path+'/sample_num='+str(sample_num)+'.pth'
        print('load model from'+model_saved_at)
        net.load_state_dict(torch.load(model_saved_at))
    else:
        print('{} test the model after pruned'.format(datetime.now()))                      #no previous checkpoint
        evaluate.evaluate_net(net,validation_loader,save_net=False)
    print("{} Start training ".format(datetime.now())+model_name+"...")
    for epoch in range(math.floor(sample_num*batch_size/train_set_size),num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        net.train()
        # one epoch for one loop
        for step, data in enumerate(train_loader, 0):
            if sample_num / math.ceil(train_set_size / batch_size)==epoch+1:               #one epoch of training finished
                evaluate.evaluate_net(net,validation_loader,
                               save_net=True,
                               checkpoint_path=checkpoint_path,
                               highest_accuracy_path=highest_accuracy_path,
                               sample_num_path=sample_num_path,
                               sample_num=sample_num)
                break

            # 准备数据
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward + backward
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sample_num += 1

            if step % checkpoint_step == 0 and step != 0:
                evaluate.evaluate_net(net,validation_loader,
                               save_net=True,
                               checkpoint_path=checkpoint_path,
                               highest_accuracy_path=highest_accuracy_path,
                               sample_num_path=sample_num_path,
                               sample_num=sample_num)
                print('{} continue training'.format(datetime.now()))


if __name__ == "__main__":
    net=train.create_net('vgg16_bn',True)

    num_conv = 0  # num of conv layers in the net
    for mod in net.features:
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv += 1

    for i in range(1, num_conv + 1):
        net = select_and_prune_filter(net, layer_index=i, percent_of_pruning=0.5,
                                      ord=2)  # prune the model
    iteration=1
    while(True):
        print('{} start iteration:{}'.format(datetime.now(),iteration))
        for i in range(1, num_conv + 1):
            net = select_and_prune_filter(net, layer_index=i, percent_of_pruning=0.1,
                                          ord=2)  # prune the model
            print('{} layer {} pruned'.format(datetime.now(),i))
            train.train(net=net,
                        net_name='vgg16_bn,gradual_pruned',
                        num_epochs=1,
                        target_accuracy=0.7,
                        learning_rate=1e-4
                        )
            iteration+=1
    # prune_and_train(model_name='vgg16_bn',
    #                 pretrained=True,
    #                 checkpoint_step=5000,
    #                 percent_of_pruning=0.9,
    #                 num_epochs=20,
    #                 learning_rate=0.005)