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



def exponential_decay_learning_rate(optimizer, learning_rate, sample_num, decay_steps,learning_rate_decay_factor):
    """Sets the learning rate to the initial LR decayed by learning_rate_decay_factor every decay_steps"""
    lr = learning_rate *learning_rate_decay_factor ** int(sample_num / decay_steps)
    if sample_num%decay_steps==0:
        print('{} learning rate at present is {}'.format(datetime.now(),lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr





def create_net(net_name,pretrained):
    temp = re.search(r'(\d+)', net_name).span()[0]
    net = net_name[:temp]  # name of the net.ex: vgg,resnet...
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the net
    net = getattr(globals()[net], net_name)(pretrained=pretrained).to(device)
    print('{} Net {} created,'.format(datetime.now(),net_name),end='')
    if pretrained:
        print('using pretrained weight.')
    else:
        print('initiate weight.')
    return net
    


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
                    root_path=conf.root_path,
                    checkpoint_path=None,
                    highest_accuracy_path=None,
                    sample_num_path=None,
                    default_image_size=224,
                    momentum=conf.momentum,
                    num_workers=conf.num_workers,
                    learning_rate_decay=False,
                    learning_rate_decay_factor=conf.learning_rate_decay_factor,
                    weight_decay=conf.weight_decay,
                    target_accuracy=1,
                  ):
    #implemented according to "Pruning Filters For Efficient ConvNets" by Hao Li
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
    optimizer=optim.Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)

    #prepare the data
    if dataset_name is 'imagenet':
        mean=conf.imagenet['mean']
        std=conf.imagenet['std']
        train_set_path=conf.imagenet['train_set_path']
        train_set_size=conf.imagenet['train_set_size']
        validation_set_path=conf.imagenet['validation_set_path']
    elif dataset_name is 'cifar10':
        train_set_size=conf.cifar10['train_set_size']
    if train_loader is None:
        train_loader=data_loader.create_train_loader(train_set_path,default_image_size,mean,std,batch_size,num_workers)
    if validation_loader is None:
        validation_loader=data_loader.create_validation_loader(validation_set_path,default_image_size,mean,std,batch_size,num_workers)

    if checkpoint_path is None:
        checkpoint_path=root_path+net_name+'/checkpoint'
    if highest_accuracy_path is None:
        highest_accuracy_path=root_path+net_name+'/highest_accuracy.txt'
    if sample_num_path is None:
        sample_num_path=root_path+net_name+'/sample_num.txt'
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
        net_saved_at=checkpoint_path+'/sample_num='+str(sample_num)+'.pth'
        if load_net:
            print('load net from'+net_saved_at)
            net.load_state_dict(torch.load(net_saved_at))
    else:
        print('{} test the net'.format(datetime.now()))                      #no previous checkpoint
        accuracy=evaluate.evaluate_net(net,validation_loader,save_net=False)
        if accuracy >= target_accuracy:
            print('{} net reached target accuracy.'.format(datetime.now()))
            return

    #ensure the net will be evaluated despite the inappropriate checkpoint_step
    if checkpoint_step>int(train_set_size/batch_size):
        checkpoint_step=int(train_set_size/batch_size)

    print("{} Start training ".format(datetime.now())+net_name+"...")
    for epoch in range(math.floor(sample_num/train_set_size),num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        net.train()
        # one epoch for one loop
        for step, data in enumerate(train_loader, 0):
            if sample_num / train_set_size==epoch+1:               #one epoch of training finished
                accuracy=evaluate.evaluate_net(net,validation_loader,
                               save_net=True,
                               checkpoint_path=checkpoint_path,
                               highest_accuracy_path=highest_accuracy_path,
                               sample_num_path=sample_num_path,
                               sample_num=sample_num)
                if accuracy>=target_accuracy:
                    print('{} net reached target accuracy.'.format(datetime.now()))
                    return
                break

            # 准备数据
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            if learning_rate_decay:
                exponential_decay_learning_rate(optimizer=optimizer,
                                                learning_rate=learning_rate,
                                                sample_num=sample_num,
                                                decay_steps=sample_num/train_set_size*2,
                                                learning_rate_decay_factor=learning_rate_decay_factor)
            optimizer.zero_grad()
            # forward + backward
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sample_num += batch_size

            if step % checkpoint_step == 0 and step != 0:
                accuracy=evaluate.evaluate_net(net,validation_loader,
                               save_net=True,
                               checkpoint_path=checkpoint_path,
                               highest_accuracy_path=highest_accuracy_path,
                               sample_num_path=sample_num_path,
                               sample_num=sample_num)
                if accuracy>=target_accuracy:
                    print('{} net reached target accuracy.'.format(datetime.now()))
                    return
                print('{} continue training'.format(datetime.now()))


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
            outputs=outputs.detach().numpy()
            outputs=outputs[0,:num_image_show,:,:]
            outputs=pixel_transform(outputs)

            #using pca to reduce the num of channels
            image_dim_reduced=np.swapaxes(np.swapaxes(outputs,0,1),1,2)
            shape=image_dim_reduced.shape
            image_dim_reduced=np.resize(image_dim_reduced,(shape[0]*shape[1],shape[2]))
            pca = PCA(n_components=32)#int(image_dim_reduced.shape[1]*0.5))  # 加载PCA算法，设置降维后主成分数目为32
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

    net=vgg.vgg16_bn(pretrained=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    data_loader=data_loader.create_validation_loader('/home/victorfang/Desktop/imagenet所有数据/imagenet_validation',224,conf.imagenet['mean'],conf.imagenet['std'],batch_size=conf.batch_size,num_workers=conf.num_workers)
    evaluate.check_ReLU_alive(net,data_loader)



    # evaluate.evaluate_net(net,data_loader,False)
    # show_feature_map(net,data_loader,[2,4,8])

