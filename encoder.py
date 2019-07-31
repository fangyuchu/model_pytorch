import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import numpy as np
from datetime import datetime
import vgg
# from train import create_net
# import pretrainedmodels
import os
import random
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA           #加载PCA算法包
import train






# 超参数
EPOCH = 10



class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.input_dimension=3*3*512
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Sequential(
            #todo:之前激活函数是tanh，是否有必要改？
            #todo:自编码器的层数是否太多
            nn.Linear(self.input_dimension, 2304),
            nn.BatchNorm1d(2304),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(2304, 576),
            nn.BatchNorm1d(576),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(576, 144),
            nn.BatchNorm1d(144),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(144, 30),
        )

        self.decoder = nn.Sequential(
            nn.Linear(30, 144),
            nn.BatchNorm1d(144),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(144, 576),
            nn.BatchNorm1d(576),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(576, 2304),
            nn.BatchNorm1d(2304),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(2304,self.input_dimension),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def extract_feature(self,**kwargs):
        if 'net' in kwargs.keys():
            filters=get_filters(kwargs['net'])
        elif 'filters' in kwargs.keys():
            filters=kwargs['filters']
        else:
            print('you must provide net or filters')
            raise AttributeError
        filters=pad_filter(filters)
        x = torch.from_numpy(filters).to(self.device)
        features = self.encoder(x)
        decoded = self.decoder(features)
        features=features.data.cpu().numpy()
        decoded=decoded.data.cpu().numpy()
        return features, decoded



def pca(conv_weight):
    pca = PCA(n_components=27)  # int(image_dim_reduced.shape[1]*0.5))  # 加载PCA算法，设置降维后主成分数目为32
    # pca = PCA(n_components='mle')

    encoded=pca.fit_transform(conv_weight)
    return encoded


def read_from_checkpoint(path):
    if '.tar' in path:
        file_list=[path]                                #single net
    else:
        file_list=os.listdir(path)
    filters = list()
    for file_name in file_list:
        if '.tar' in file_name:
            checkpoint=torch.load(os.path.join(path,file_name))
            net=checkpoint['net']
            net.load_state_dict(checkpoint['state_dict'])
            filters+=get_filters(net=net)
    return filters

def get_filters(net):
    filters=list()
    for mod in net.modules():
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            for f in mod.weight.data.cpu().numpy():
                filters.append(f)
    return filters

def pad_filter(filters,pad_length=3*3*512):
    '''
    
    :param filters: list of 3-d ndarray
    :param pad_length: 
    :return: 2d ndarray of shape[num_filters,pad_length]. each row represents one filter
    '''
    for i in range(len(filters)):
        filters[i]=np.reshape(filters[i],-1)
        filters[i]=np.pad(filters[i],(0,pad_length-filters[i].shape[0]),'constant',constant_values=-1)
    return np.array(filters)


def train_encoder(train_dir='',val_dir=''):
    filter_train=read_from_checkpoint(train_dir)
    filter_val=read_from_checkpoint(val_dir)

    filter_train=pad_filter(filter_train)

    filter_val=pad_filter(filter_val)

    index=[j for j in range(filter_train.shape[0])]                 #used for shuffle the training data

    # use auto_encoder to encode weight. seemed not work.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auto_encoder = AutoEncoder().to(device)
    optimizer = torch.optim.SGD(auto_encoder.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4 )
    #torch.optim.Adam(auto_encoder.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()
    batch_size=1000
    num_epoch=1000
    sample_num=0
    #loss_func=nn.L1Loss()
    for epoch in range(num_epoch):
        random.shuffle(index)
        filter_train=filter_train[index]                                            #shuffle the input
        auto_encoder.train()
        batch_num=math.ceil(filter_train.shape[0]/batch_size)
        for i in range(batch_num):
            train.exponential_decay_learning_rate(optimizer=optimizer,
                                            sample_num=sample_num,
                                            learning_rate_decay_factor=0.1,
                                            train_set_size=filter_train.shape[0],
                                            learning_rate_decay_epoch=[],
                                            batch_size=batch_size)


            x=torch.from_numpy(filter_train[i*batch_size:(i+1)*batch_size]).to(device)
            sample_num+=x.shape[0]
            encoded, decoded = auto_encoder(x)
            loss = loss_func(decoded, x)#+(torch.sum(torch.abs(decoded-torch.mean(x))))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("{} epoch:{},  loss is:{}".format(datetime.now(),epoch,loss))
    checkpoint = {'auto_encoder': auto_encoder,
                  'state_dict': auto_encoder.state_dict(),
                  'sample_num': sample_num}
    torch.save(checkpoint, './auto_encoder.tar')
    print("{} net saved at sample num = {}".format(datetime.now(), sample_num))




          

model_urls = [

    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
    'vgg11_bn',
    'vgg13_bn',
    'vgg16_bn',
    'vgg19_bn'
]

root='/home/victorfang/PycharmProjects/model_pytorch/data/model_params/'

# def download_weight():
#     for net_name in model_urls:
#         net=create_net(net_name,pretrained=True)
#         for i in range(len(net.features)):
#             mod=net.features[i]
#             if isinstance(mod, torch.nn.modules.conv.Conv2d):
#                 weight=mod.weight.data.cpu().numpy()
#                 np.save(root+'weight/'+net_name+','+str(i),weight)
#                 bias=mod.bias.data.cpu().numpy()
#                 np.save(root+'bias/'+net_name+','+str(i),bias)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auto_encoder=AutoEncoder().to(device)
    checkpoint=torch.load('./auto_encoder.tar')
    auto_encoder.load_state_dict(checkpoint['state_dict'])

    checkpoint = torch.load('./baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
    net=checkpoint['net']
    net.load_state_dict(checkpoint['state_dict'])

    feature,decode=auto_encoder.extract_feature(net=net)

    print()

    # train_encoder(train_dir='./auto_encoder/train',val_dir='./auto_encoder/val')

