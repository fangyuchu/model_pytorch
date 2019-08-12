import config as conf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import resnet
import vgg
import os
from datetime import datetime
import re
import math
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA           #加载PCA算法包
import evaluate
import data_loader
from math import ceil
import logger
import sys
import measure_flops
import resnet_copied
import copy
import create_net



def exponential_decay_learning_rate(optimizer, sample_num, train_set_size,learning_rate_decay_epoch,learning_rate_decay_factor,batch_size):
    """Sets the learning rate to the initial LR decayed by learning_rate_decay_factor every decay_steps"""
    current_epoch=ceil(sample_num/train_set_size)
    if current_epoch in learning_rate_decay_epoch and sample_num-(train_set_size*(current_epoch-1))<=batch_size:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*learning_rate_decay_factor
            lr=param_group['lr']
        print('{} learning rate at present is {}'.format(datetime.now(), lr))
    # lr = learning_rate *learning_rate_decay_factor ** int(sample_num / decay_steps)
    # if sample_num%decay_steps==0:
    #     print('{} learning rate at present is {}'.format(datetime.now(),lr))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr






    


def train(
                    net,
                    net_name,
                    dataset_name='imagenet',
                    train_loader=None,
                    validation_loader=None,
                    learning_rate=conf.learning_rate,
                    num_epochs=conf.num_epochs,
                    batch_size=conf.batch_size,
                    checkpoint_step=conf.checkpoint_step,
                    load_net=True,
                    test_net=False,
                    root_path=conf.root_path,
                    checkpoint_path=None,
                    momentum=conf.momentum,
                    num_workers=conf.num_workers,
                    learning_rate_decay=False,
                    learning_rate_decay_factor=conf.learning_rate_decay_factor,
                    learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
                    weight_decay=conf.weight_decay,
                    target_accuracy=1.0,
                    optimizer=optim.Adam,
                    top_acc=1
                  ):
    '''

    :param net: net to be trained
    :param net_name: name of the net
    :param dataset_name: name of the dataset
    :param train_loader: data_loader for training. If not provided, a data_loader will be created based on dataset_name
    :param validation_loader: data_loader for validation. If not provided, a data_loader will be created based on dataset_name
    :param learning_rate: initial learning rate
    :param learning_rate_decay: boolean, if true, the learning rate will decay based on the params provided.
    :param learning_rate_decay_factor: float. learning_rate*=learning_rate_decay_factor, every time it decay.
    :param learning_rate_decay_epoch: list[int], the specific epoch that the learning rate will decay.
    :param num_epochs: max number of epochs for training
    :param batch_size:
    :param checkpoint_step: how often will the net be tested on validation set. At least one test every epoch is guaranteed
    :param load_net: boolean, whether loading net from previous checkpoint. The newest checkpoint will be selected.
    :param test_net:boolean, if true, the net will be tested before training.
    :param root_path:
    :param checkpoint_path:
    :param momentum:
    :param num_workers:
    :param weight_decay:
    :param target_accuracy:float, the training will stop once the net reached target accuracy
    :param optimizer:
    :param top_acc: can be 1 or 5
    :return:
    '''
    success=True                                                                   #if the trained net reaches target accuracy
    # gpu or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ',end='')
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print(device)

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate,momentum=momentum,#weight_decay=weight_decay
    #                       )  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    if optimizer is optim.Adam:
        optimizer = optimizer(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer is optim.SGD:
        optimizer=optimizer(net.parameters(),lr=learning_rate,weight_decay=weight_decay,momentum=momentum)

    #prepare the data
    if dataset_name is 'imagenet':
        mean=conf.imagenet['mean']
        std=conf.imagenet['std']
        train_set_path=conf.imagenet['train_set_path']
        train_set_size=conf.imagenet['train_set_size']
        validation_set_path=conf.imagenet['validation_set_path']
        default_image_size = conf.imagenet['default_image_size']
    elif dataset_name is 'cifar10':
        train_set_size=conf.cifar10['train_set_size']
        mean=conf.cifar10['mean']
        std=conf.cifar10['std']
        train_set_path=conf.cifar10['train_set_path']
        validation_set_path=conf.cifar10['validation_set_path']
        default_image_size=conf.cifar10['default_image_size']
    elif dataset_name is 'tiny_imagenet':
        train_set_size = conf.tiny_imagenet['train_set_size']
        mean = conf.tiny_imagenet['mean']
        std = conf.tiny_imagenet['std']
        train_set_path = conf.tiny_imagenet['train_set_path']
        validation_set_path = conf.tiny_imagenet['validation_set_path']
        default_image_size = conf.tiny_imagenet['default_image_size']
    if train_loader is None:
        train_loader=data_loader.create_train_loader(dataset_path=train_set_path,
                                                     default_image_size=default_image_size,
                                                     mean=mean,
                                                     std=std,
                                                     batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     dataset_name=dataset_name)
    if validation_loader is None:
        validation_loader=data_loader.create_validation_loader(dataset_path=validation_set_path,
                                                               default_image_size=default_image_size,
                                                               mean=mean,
                                                               std=std,
                                                               batch_size=batch_size,
                                                               num_workers=num_workers,
                                                               dataset_name=dataset_name)


    if checkpoint_path is None:
        checkpoint_path=root_path+net_name+'/checkpoint'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path,exist_ok=True)

    #get the latest checkpoint
    lists = os.listdir(checkpoint_path)
    file_new=checkpoint_path
    if len(lists)>0:
        lists.sort(key=lambda fn: os.path.getmtime(checkpoint_path + "/" + fn))  # 按时间排序
        file_new = os.path.join(checkpoint_path, lists[-1])  # 获取最新的文件保存到file_new

    sample_num=0
    if os.path.isfile(file_new):
        if load_net:
            checkpoint = torch.load(file_new)
            print('{} load net from previous checkpoint'.format(datetime.now()))
            net=checkpoint['net']
            net.load_state_dict(checkpoint['state_dict'])
            sample_num = checkpoint['sample_num']

    if test_net:
        print('{} test the net'.format(datetime.now()))                      #no previous checkpoint
        net_test=copy.deepcopy(net)
        accuracy=evaluate.evaluate_net(net_test,validation_loader,
                                       save_net=True,
                                       checkpoint_path=checkpoint_path,
                                       sample_num=sample_num,
                                       target_accuracy=target_accuracy,
                                       dataset_name=dataset_name,
                                       top_acc=top_acc)
        del net_test

        if accuracy >= target_accuracy:
            print('{} net reached target accuracy.'.format(datetime.now()))
            return success

    #ensure the net will be evaluated despite the inappropriate checkpoint_step
    if checkpoint_step>math.ceil(train_set_size/batch_size)-1:
        checkpoint_step=math.ceil(train_set_size/batch_size)-1

    print("{} Start training ".format(datetime.now())+net_name+"...")
    for epoch in range(math.floor(sample_num/train_set_size),num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        net.train()
        # one epoch for one loop
        for step, data in enumerate(train_loader, 0):
            if sample_num / train_set_size==epoch+1:               #one epoch of training finished
                net_test = copy.deepcopy(net)
                accuracy=evaluate.evaluate_net(net_test,validation_loader,
                                               save_net=True,
                                               checkpoint_path=checkpoint_path,
                                               sample_num=sample_num,
                                               target_accuracy=target_accuracy,
                                               dataset_name=dataset_name,
                                               top_acc=top_acc)
                del net_test
                if accuracy>=target_accuracy:
                    print('{} net reached target accuracy.'.format(datetime.now()))
                    return success
                break

            # 准备数据
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            sample_num += int(images.shape[0])

            if learning_rate_decay:
                exponential_decay_learning_rate(optimizer=optimizer,
                                                sample_num=sample_num,
                                                learning_rate_decay_factor=learning_rate_decay_factor,
                                                train_set_size=train_set_size,
                                                learning_rate_decay_epoch=learning_rate_decay_epoch,
                                                batch_size=batch_size)

            optimizer.zero_grad()
            # forward + backward
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if step%20==0:
                print('{} loss is {}'.format(datetime.now(),float(loss.data)))



            if step % checkpoint_step == 0 and step != 0:
                net_test = copy.deepcopy(net)
                accuracy=evaluate.evaluate_net(net_test,validation_loader,
                                                save_net=True,
                                                checkpoint_path=checkpoint_path,
                                                sample_num=sample_num,
                                               target_accuracy=target_accuracy,
                                               dataset_name=dataset_name,
                                               top_acc=top_acc)
                del net_test
                if accuracy>=target_accuracy:
                    print('{} net reached target accuracy.'.format(datetime.now()))
                    return success
                del accuracy
                print('{} continue training'.format(datetime.now()))

    print("{} Training finished. Saving net...".format(datetime.now()))
    flop_num=measure_flops.measure_model(net=net,dataset_name=dataset_name,print_flop=False)
    checkpoint = {'net': net,
                  'highest_accuracy': accuracy,
                  'state_dict': net.state_dict(),
                  'sample_num': sample_num,
                  'flop_num': flop_num}
    torch.save(checkpoint, '%s/flop=%d,accuracy=%.5f.tar' % (checkpoint_path, flop_num, accuracy))
    print("{} net saved at sample num = {}".format(datetime.now(), sample_num))
    return not success


def show_feature_map(
                    net,
                    data_loader,
                    layer_indexes,
                    num_image_show=64
                     ):
    '''
    show the feature converted feature maps of a cnn
    :param net: full network net
    :param data_loader: data_loader to load data
    :param layer_indexes: list of indexes of conv layer whose feature maps will be extracted and showed
    :param num_image_show: number of feature maps showed in one conv_layer. Supposed to be a square number
    :return:
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sub_net=[]
    conv_index = 0
    ind_in_features=-1
    j=0
    for mod in net.features:
        ind_in_features+=1
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            conv_index+=1
            if conv_index in layer_indexes:
                sub_net.append(nn.Sequential(*list(net.children())[0][:ind_in_features+1]))
                j+=1
    
    #sub_net = nn.Sequential(*list(net.children())[0][:conv_index+1])
    for step, data in enumerate(data_loader, 0):
        # 准备数据
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        for i in range(len(layer_indexes)):
            # forward
            sub_net[i].eval()
            outputs = sub_net[i](images)
            outputs=outputs.detach().cpu().numpy()
            outputs=outputs[0,:num_image_show,:,:]
            outputs=pixel_transform(outputs)

            #using pca to reduce the num of channels
            image_dim_reduced=np.swapaxes(np.swapaxes(outputs,0,1),1,2)
            shape=image_dim_reduced.shape
            image_dim_reduced=np.resize(image_dim_reduced,(shape[0]*shape[1],shape[2]))
            pca = PCA(n_components=40)#int(image_dim_reduced.shape[1]*0.5))  # 加载PCA算法，设置降维后主成分数目为32
            image_dim_reduced = pca.fit_transform(image_dim_reduced)  # 对样本进行降维
            image_dim_reduced=np.resize(image_dim_reduced,(shape[0],shape[1],image_dim_reduced.shape[1]))
            image_dim_reduced = np.swapaxes(np.swapaxes(image_dim_reduced, 1, 2), 0, 1)
            image_dim_reduced=pixel_transform(image_dim_reduced)
            plt.figure(figsize=[14,20],clear=True,num=layer_indexes[i]+100)
            for j in range(32):
                im=Image.fromarray(image_dim_reduced[j])
                plt.subplot(math.sqrt(num_image_show),math.sqrt(num_image_show),j+1)
                plt.imshow(im,cmap='Greys_r')


            plt.figure(figsize=[14,20],clear=True,num=layer_indexes[i])
            for j in range(num_image_show):
                im=Image.fromarray(outputs[j])
                plt.subplot(math.sqrt(num_image_show),math.sqrt(num_image_show),j+1)
                plt.imshow(im,cmap='Greys_r')
        plt.show()
        break

def pixel_transform(feature_maps):
    #把feature maps数值移至0-255区间
    mean = feature_maps.mean()
    transform = 255 / 2 - mean
    feature_maps = feature_maps + transform  # 把所有像素提至255的中点附近
    max = feature_maps.max()
    min = feature_maps.min()
    mean = feature_maps.mean()
    if max - mean > mean - min:
        ratio = (255 - mean) / (max - mean)
    else:
        ratio = mean / (mean - min)
    feature_maps = ratio * (feature_maps - mean) + mean  # 把像素划入0-255
    return feature_maps


if __name__ == "__main__":

    # save the output to log
    print('save log in:./log.txt')

    sys.stdout = logger.Logger( './log.txt', sys.stdout)
    sys.stderr = logger.Logger( './log.txt', sys.stderr)  # redirect std err, if necessary

    #net = resnet.resnet34(num_classes=10)
    net=vgg.vgg16_bn(pretrained=True)
    m1=nn.Linear(2048,4096)
    nn.init.normal_(m1.weight, 0, 0.01)
    nn.init.constant_(m1.bias, 0)
    net.classifier[0]=m1

    m3=nn.Linear(4096,200)
    nn.init.normal_(m3.weight, 0, 0.01)
    nn.init.constant_(m3.bias, 0)
    net.classifier[6]=m3
    # net.classifier = nn.Sequential(
    #     nn.Linear(512 * 7 * 7, 4096),
    #     nn.ReLU(True),
    #     nn.Dropout(),
    #     nn.Linear(4096, 4096),
    #     nn.ReLU(True),
    #     nn.Dropout(),
    #     nn.Linear(4096, 200),
    # )
    # for m in net.modules():
    #     if isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight, 0, 0.01)
    #         nn.init.constant_(m.bias, 0)
    net = net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # net=create_net.vgg_tiny_imagenet(net_name='vgg16_bn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # measure_flops.measure_model(net, dataset_name='cifar10')
    batch_size=600
    num_worker=8
    train_loader=data_loader.create_train_loader(batch_size=batch_size,
                                                 num_workers=num_worker,
                                                 dataset_name='tiny_imagenet',
                                                 default_image_size=64,
                                                 )
    validation_loader=data_loader.create_validation_loader(batch_size=batch_size,
                                                           num_workers=num_worker,
                                                           dataset_name='tiny_imagenet',
                                                           default_image_size=64
                                                        )
    for i in range(1):
        print(i)
        train(net=net,
              net_name='vgg16bn_tiny_imagenet2',
              dataset_name='tiny_imagenet',
              test_net=False,
              optimizer=optim.SGD,
              learning_rate=0.1,
              learning_rate_decay=False,
              learning_rate_decay_epoch=[ 100, 200, 300],
              learning_rate_decay_factor=0.1,
              load_net=True,
              batch_size=batch_size,
              num_epochs=450,
              weight_decay=0.0006,
              train_loader=train_loader,
              validation_loader=validation_loader,
              checkpoint_step=1000,
              )



    # checkpoint = torch.load(
    #     '/home/victorfang/Desktop/pytorch_model/vgg16bn_cifar10_dead_neural_normal_tar_acc_decent1/checkpoint/sample_num=11050000,accuracy=0.93370.tar')
    #
    # net = checkpoint['net']
    # net.load_state_dict(checkpoint['state_dict'])
    # print(checkpoint['highest_accuracy'])
    #
    # measure_flops.measure_model(net, dataset_name='cifar10')
    #
    # train(net=net,
    #       net_name='temp_train_a_net',
    #       dataset_name='cifar10',
    #       optimizer=optim.SGD,
    #       learning_rate=0.001,
    #       learning_rate_decay=True,
    #       learning_rate_decay_epoch=[50,100,150,250,300,350,400],
    #       learning_rate_decay_factor=0.5,
    #       test_net=False,
    #       load_net=False,
    #       target_accuracy=0.933958988332225,
    #       batch_size=600,
    #       num_epochs=450)





