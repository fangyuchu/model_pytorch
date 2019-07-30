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
        self.encoder = nn.Sequential(
            #todo:之前激活函数是tanh，是否有必要改？
            #todo:自编码器的层数是否太多
            nn.Linear(self.input_dimension, 2304),
            nn.BatchNorm1d(2304),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(2304, 576),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(576, 144),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(144, 18),
        )

        self.decoder = nn.Sequential(
            # nn.Linear(18, 144),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(144, 576),
            # nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(576, 2304),
            nn.BatchNorm1d(2304),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(2304,self.input_dimension),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def pca(conv_weight):
    pca = PCA(n_components=27)  # int(image_dim_reduced.shape[1]*0.5))  # 加载PCA算法，设置降维后主成分数目为32
    # pca = PCA(n_components='mle')

    encoded=pca.fit_transform(conv_weight)
    return encoded

    # s=10
    # file_list=os.listdir(dir)
    # validation_file_list=random.sample(file_list,int(len(file_list)/s))
    # train_file_list=list(set(file_list).difference(set(validation_file_list)))
    # for file in train_file_list:
    #     x=np.load(dir+file)                 #[out_channels,in_channels,kernel_size,kernel_size]
    #     x=np.reshape(x,(x.shape[0],-1))     #each row is one kernel

def get_filters(path):
    file_list=os.listdir(path)
    filters = list()
    for file_name in file_list:
        if '.tar' in file_name:

            checkpoint=torch.load(os.path.join(path,file_name))
            net=checkpoint['net']
            net.load_state_dict(checkpoint['state_dict'])

            num_conv = 0  # num of conv layers in the net
            for mod in net.modules():
                if isinstance(mod, torch.nn.modules.conv.Conv2d):
                    num_conv += 1
                    # filter_num.append(mod.out_channels)
                    for f in mod.weight.data.cpu().numpy():
                        filters.append(f)
    return filters

            # for i in range(num_conv):
            #     for module_key in list(neural_list.keys()):
            #         if module_list[i] is module_key:                                    #find the neural_list_statistics in layer i+1
            #             dead_times=neural_list[module_key]
            #             neural_num=dead_times.shape[1]*dead_times.shape[2]  #neural num for one filter
            #             filter_weight = filters[i].weight.data.cpu().numpy()
            #
            #             if regression_or_classification is 'classification':
            #                 # judge dead filter by neural_dead_times and dead_filter_ratio
            #                 dead_times[dead_times<neural_dead_times]=0
            #                 dead_times[dead_times>=neural_dead_times]=1
            #                 dead_times=np.sum(dead_times,axis=(1,2))            #count the number of dead neural for one filter
            #                 dead_filter_index=np.where(dead_times>neural_num*filter_dead_ratio)[0].tolist()
            #                 living_filter_index=[i for i in range(filter_num[i]) if i not in dead_filter_index]
            #
            #                 for ind in dead_filter_index:
            #                     dead_filter.append(filter_weight[ind])
            #                 dead_filter_layer+=[i for j in range(len(dead_filter_index))]
            #                 for ind in living_filter_index:
            #                     living_filter.append(filter_weight[ind])
            #                 living_filter_layer += [i for j in range(len(living_filter_index))]
            #             else:
            #                 #compute sum(dead_times)/(batch_size*neural_num) as label for each filter
            #                 dead_times=np.sum(dead_times,axis=(1,2))
            #                 prediction=dead_times/(neural_num*batch_size)
            #                 for f in filter_weight:
            #                     filter.append(f)
            #                 filter_label+=prediction.tolist()
            #                 filter_layer+=[i for j in range(filter_weight.shape[0])]

def train_encoder(train_dir='',val_dir=''):
    # cross_validation,use 1/s of the data as validation set
    s=10
    filter_train=get_filters(train_dir)
    filter_val=get_filters(val_dir)

    for i in range(len(filter_train)):
        filter_train[i]=np.reshape(filter_train[i],-1)
        filter_train[i]=np.pad(filter_train[i],(0,3*3*512-filter_train[i].shape[0]),'constant',constant_values=-1)

    for i in range(len(filter_val)):
        filter_val[i] = np.reshape(filter_val[i], -1)
        filter_val[i] = np.pad(filter_val[i], (0, 3 * 3 * 512 - filter_val[i].shape[0]), 'constant',constant_values=-1)
    filter_train=np.array(filter_train)
    index=[j for j in range(filter_train.shape[0])]                 #used for shuffle the training data

    filter_val=np.array(filter_val)

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
                                            learning_rate_decay_epoch=[200,400,600],
                                            batch_size=batch_size)


            # x=np.load(dir+file)                 #[out_channels,in_channels,kernel_size,kernel_size]
            # x=np.reshape(x,(x.shape[0],-1))     #each row is one kernel
            # x=np.pad(x,((0,0),(0,auto_encoder.input_dimension-x.shape[1])),'constant')    #pad kernel with 0 at the end
            x=torch.from_numpy(filter_train[i*batch_size:(i+1)*batch_size]).to(device)
            sample_num+=x.shape[0]
            encoded, decoded = auto_encoder(x)
            loss = loss_func(decoded, x)#+(torch.sum(torch.abs(decoded-torch.mean(x))))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("{} epoch:{},  loss is:{}".format(datetime.now(),epoch,loss))
    #
    #     #plt.figure(figsize=[14, 20], clear=True, num=epoch)
    #     plt.figure(10)
    #     auto_encoder.eval()
    #     for file in validation_file_list:
    #         x = np.load(dir + file)  # [out_channels,in_channels,kernel_size,kernel_size]
    #         x = np.reshape(x, (x.shape[0], -1))  # each row is one kernel
    #         x = np.pad(x, ((0, 0), (0, auto_encoder.input_dimension - x.shape[1])),
    #                    'constant')  # pad kernel with 0 at the end
    #         x = torch.from_numpy(x)
    #         encoded, decoded = auto_encoder(x)
    #         plt.subplot(1,2,1)
    #         plt.plot(decoded[0].data.numpy()[0:1000],'.')
    #         plt.subplot(1,2,2)
    #         plt.plot(x[0].data.numpy()[0:1000],'.')
    #         plt.show()



          

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


    train_encoder(train_dir='./auto_encoder/train',val_dir='./auto_encoder/val')
    # net_name='alexnet'
    # net=pretrainedmodels.__dict__[net_name](num_classes=1000, pretrained='imagenet')
    # for i in range(len(net._features)):
    #     mod = net._features[i]
    #     if isinstance(mod, torch.nn.modules.conv.Conv2d):
    #         weight = mod.weight.data.cpu().numpy()
    #         np.save(root + 'weight/' + net_name + ',' + str(i), weight)
    #         bias = mod.bias.data.cpu().numpy()
    #         np.save(root + 'bias/' + net_name + ',' + str(i), bias)
    #download_weight()
