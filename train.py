import config as conf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import resnet
import vgg
import os
from datetime import datetime
import re
import math
from PIL import Image
import matplotlib.pyplot as plt




def exponential_decay_learning_rate(optimizer, learning_rate, global_step, decay_steps,learning_rate_decay_factor):
    """Sets the learning rate to the initial LR decayed by learning_rate_decay_factor every decay_steps"""
    lr = learning_rate *learning_rate_decay_factor ** int(global_step / decay_steps)
    if global_step%decay_steps==0:
        print('{} learning rate at present is {}'.format(datetime.now(),lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate_model(net,
                   data_loader,
                   save_model,
                   checkpoint_path=None,
                   highest_accuracy_path=None,
                   global_step_path=None,
                   global_step=0,
                   ):
    '''
    :param net: model of NN
    :param data_loader: data loader of test set
    :param save_model: Boolean. Whether or not to save the model.
    :param checkpoint_path: 
    :param highest_accuracy_path: 
    :param global_step_path: 
    :param global_step: global step of the current trained model
    '''
    if save_model:
        if checkpoint_path is None :
            raise AttributeError('please input checkpoint path')
        if highest_accuracy_path is None :
            raise AttributeError('please input highest_accuracy path')
        if global_step_path is None :
            raise AttributeError('please input global_step path')
        if os.path.exists(highest_accuracy_path):
            f = open(highest_accuracy_path, 'r')
            highest_accuracy = float(f.read())
            f.close()
        else:
            highest_accuracy=0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("{} Start Evaluation".format(datetime.now()))
    print("{} global step = {}".format(datetime.now(), global_step))
    with torch.no_grad():
        correct = 0
        total = 0
        for val_data in data_loader:
            net.eval()
            images, labels = val_data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        correct = float(correct.cpu().numpy().tolist())
        accuracy = correct / total
        print("{} Accuracy = {:.4f}".format(datetime.now(), accuracy))
        if save_model and accuracy > highest_accuracy:
            highest_accuracy = accuracy
            # save model
            print("{} Saving model...".format(datetime.now()))
            torch.save(net.state_dict(), '%s/global_step=%d.pth' % (checkpoint_path, global_step))
            print("{} Model saved ".format(datetime.now()))
            # save highest accuracy
            f = open(highest_accuracy_path, 'w')
            f.write(str(highest_accuracy))
            f.close()
            # save global step
            f = open(global_step_path, 'w')
            f.write(str(global_step))
            print("{} model saved at global step = {}".format(datetime.now(), global_step))
            f.close()

def create_data_loader(
                    dataset_path,
                    default_image_size,
                    mean,
                    std,
                    batch_size,
                    num_workers,
                    ):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(default_image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    folder = datasets.ImageFolder(dataset_path, transform)
    data_loader = torch.utils.data.DataLoader(folder, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader

def train(
                    model_name,
                    pretrained=False,
                    dataset_name='imagenet',
                    learning_rate=conf.learning_rate,
                    num_epochs=conf.num_epochs,
                    batch_size=conf.batch_size,
                    checkpoint_step=conf.checkpoint_step,
                    checkpoint_path=None,
                    highest_accuracy_path=None,
                    global_step_path=None,
                    default_image_size=224,
                    momentum=conf.momentum,
                    num_workers=conf.num_workers,
                    learning_rate_decay_factor=conf.learning_rate_decay_factor,
                    weight_decay=conf.weight_decay
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

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,momentum=momentum,weight_decay=weight_decay
                          )  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    #prepare the data
    if dataset_name is 'imagenet':
        mean=conf.imagenet['mean']
        std=conf.imagenet['std']
        train_set_path=conf.imagenet['train_set_path']
        train_set_size=conf.imagenet['train_set_size']
        validation_set_path=conf.imagenet['validation_set_path']

    train_loader=create_data_loader(train_set_path,default_image_size,mean,std,batch_size,num_workers)
    validation_loader=create_data_loader(validation_set_path,default_image_size,mean,std,batch_size,num_workers)

    if checkpoint_path is None:
        checkpoint_path=conf.root_path+model_name+'/checkpoint'
    if highest_accuracy_path is None:
        highest_accuracy_path=conf.root_path+model_name+'/highest_accuracy.txt'
    if global_step_path is None:
        global_step_path=conf.root_path+model_name+'/global_step.txt'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path,exist_ok=True)

    if  os.path.exists(highest_accuracy_path):
        f = open(highest_accuracy_path, 'r')
        highest_accuracy = float(f.read())
        f.close()
        print('highest accuracy from previous training is %f' % highest_accuracy)
        del highest_accuracy

    global_step=0
    if os.path.exists(global_step_path):
        f = open(global_step_path, 'r')
        global_step = int(f.read())
        f.close()
        print('global_step at present is %d' % global_step)
        model_saved_at=checkpoint_path+'/global_step='+str(global_step)+'.pth'
        print('load model from'+model_saved_at)
        net.load_state_dict(torch.load(model_saved_at))
    else:
        print('{} test the model'.format(datetime.now()))                      #no previous checkpoint
        evaluate_model(net,validation_loader,save_model=False)

    step_one_epoch=math.ceil(train_set_size / batch_size)

    print("{} Start training ".format(datetime.now())+model_name+"...")
    for epoch in range(math.floor(global_step*batch_size/train_set_size),num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        net.train()
        # one epoch for one loop
        for step, data in enumerate(train_loader, 0):
            if global_step / step_one_epoch==epoch+1:               #one epoch of training finished
                evaluate_model(net,validation_loader,
                               save_model=True,
                               checkpoint_path=checkpoint_path,
                               highest_accuracy_path=highest_accuracy_path,
                               global_step_path=global_step_path,
                               global_step=global_step)
                break

            # 准备数据
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            exponential_decay_learning_rate(optimizer=optimizer,
                                            learning_rate=learning_rate,
                                            global_step=global_step,
                                            decay_steps=step_one_epoch*2,
                                            learning_rate_decay_factor=learning_rate_decay_factor)
            optimizer.zero_grad()
            # forward + backward
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            global_step += 1

            if step % checkpoint_step == 0 and step != 0:
                evaluate_model(net,validation_loader,
                               save_model=True,
                               checkpoint_path=checkpoint_path,
                               highest_accuracy_path=highest_accuracy_path,
                               global_step_path=global_step_path,
                               global_step=global_step)
                print('{} continue training'.format(datetime.now()))


def show_feature_map(
                    model,
                    data_loader,
                    layer_indexes,
                    num_image_show=36
                     ):
    '''
    show the feature converted feature maps of a cnn
    :param model: full network model
    :param data_loader: data_loader to load data
    :param layer_indexes: list of indexes of conv layer whose feature maps will be extracted and showed
    :param num_image_show: number of feature maps showed in one conv_layer. Supposed to be a square number
    :return:
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sub_model=[]
    conv_index = 0
    ind_in_features=-1
    j=0
    for mod in model.features:
        ind_in_features+=1
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            conv_index+=1
            if conv_index in layer_indexes:
                sub_model.append(nn.Sequential(*list(model.children())[0][:ind_in_features+1]))
                j+=1
    
    #sub_model = nn.Sequential(*list(model.children())[0][:conv_index+1])
    for step, data in enumerate(data_loader, 0):
        # 准备数据
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        for i in range(len(layer_indexes)):
            # forward
            sub_model[i].eval()
            outputs = sub_model[i](images)
            outputs=outputs.detach().numpy()
            outputs=outputs[0,:num_image_show,:,:]
            outputs=transform(outputs)
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
    train(model_name='vgg16_bn',pretrained=False,checkpoint_step=5000,num_epochs=40,learning_rate=0.01)

