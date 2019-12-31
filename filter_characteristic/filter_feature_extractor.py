import torch
from filter_characteristic.graph_convolutional_network import gcn
import torch.nn as nn
import transform_conv
from network import vgg
import network.vgg as vgg
import numpy as np
from network import storage
from framework import train
from datetime import datetime
from framework import config as conf
from random import shuffle
import copy
from filter_characteristic import predict_dead_filter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


class extractor(nn.Module):
    def __init__(self,feature_len,gcn_rounds=2):
        super(extractor, self).__init__()
        self.gcn=gcn(in_features=feature_len,out_features=feature_len)
        self.feature_len=feature_len
        self.gcn_rounds=gcn_rounds
        self.network=nn.Sequential(
            nn.Linear(feature_len * 2,128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128,1,bias=False),
            # nn.Sigmoid()
        )
        self.normalization=nn.BatchNorm1d(num_features=feature_len*2)
        
    def forward(self,net,net_name,dataset_name ):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        crosslayer_features=self.gcn.forward(net=net,rounds=self.gcn_rounds,net_name=net_name,dataset_name=dataset_name)

        filter_num=[]
        singular_value_list=[]
        for name,mod in net.named_modules():                                                   #mod is a copy
            if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
                filter_num+=[mod.out_channels]
                weight= transform_conv.conv_to_matrix(copy.deepcopy(mod))
                u, s, v = torch.svd(weight)
                singular_value_list+=[s[:self.feature_len]]
        features=torch.zeros((sum(filter_num),self.feature_len*2)).to(device)
        start=0
        for i in range(len(filter_num)):
            stop = start+filter_num[i]
            features[start:stop]=torch.cat((crosslayer_features[start:stop],singular_value_list[i].repeat(filter_num[i],1)),dim=1)
            start=stop
        features=self.normalization(features)
        return self.network(features)

class weighted_MSELoss(torch.nn.MSELoss):
    def __init__(self):
        super(weighted_MSELoss,self).__init__()

    def forward(self, input, target):
        device=input.device
        ret = (input - target) ** 2
        weight=torch.zeros(target.shape).to(device)
        weight[target<0.3]=0.3
        weight[target>=0.3]=target[target>=0.3]
        ret = ret * weight
        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret




def load_extractor(path):
    '''
    load feature extractor from checkpoint in path
    :param path:
    :return:
    '''
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint=torch.load(path)
    feature_len=checkpoint['feature_len']
    gcn_rounds=checkpoint['gcn_rounds']
    net=extractor(feature_len=feature_len,gcn_rounds=gcn_rounds).to(device)
    net.load_state_dict(checkpoint['state_dict'])
    return net

def read_data(path,
              num_images=None):
    sample=[]
    file_list = os.listdir(path)
    file_list.sort()
    print(file_list)
    for file_name in file_list:
        if '.tar' in file_name:
            checkpoint=torch.load(os.path.join(path,file_name))
            net=storage.restore_net(checkpoint,pretrained=True)

            neural_list=checkpoint['neural_list']
            try:
                module_list=checkpoint['module_list']
            except KeyError:
                module_list=checkpoint['relu_list']


            num_conv = 0  # num of conv layers in the network
            filter_weight=[]
            layers=[]
            for mod in net.modules():
                if isinstance(mod, torch.nn.modules.conv.Conv2d):
                    num_conv += 1
                    conv=mod
                elif isinstance(mod,torch.nn.ReLU):                             #ensure the conv are followed by relu
                    if layers != [] and layers[-1] == num_conv - 1:             # get rid of the influence from relu in fc
                        continue
                    filter_weight.append(conv.weight.data.cpu().numpy())
                    layers.append(num_conv-1)

            filter_layer=[]
            filter_label=[]
            for i in range(len(filter_weight)):
                for module_key in list(neural_list.keys()):
                    if module_list[i] is module_key:                                    #find the neural_list_statistics in layer i+1
                        dead_times=neural_list[module_key]
                        neural_num=dead_times.shape[1]*dead_times.shape[2]              #neural num for one filter

                        #compute sum(dead_times)/(num_images*neural_num) as label for each filter
                        dead_times=np.sum(dead_times,axis=(1,2))
                        prediction=dead_times/(neural_num*num_images)
                        filter_label+=prediction.tolist()
                        filter_layer+=[layers[i] for j in range(filter_weight[i].shape[0])]
            sample.append({'net':net,'filter_label':filter_label,'filter_layer':filter_layer,
                           'net_name':checkpoint['net_name'],'dataset_name':checkpoint['dataset_name']})

    return sample

def train_extractor(train_data_dir,
                    net_name,
                    dataset_name,
                    num_images,
                    epoch=1001,
                    feature_len=27,
                    gcn_rounds=2,
                    criterion=torch.nn.MSELoss(),
                    special='',
                    checkpoint_path=None):
    print(criterion)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor_model=extractor(feature_len=feature_len,gcn_rounds=gcn_rounds).to(device)
    sample_list=read_data(path=train_data_dir, num_images=num_images)
    optimizer=train.prepare_optimizer(net=extractor_model,optimizer=torch.optim.Adam,learning_rate=1e-2,weight_decay=0)
    # optimizer=train.prepare_optimizer(net=extractor_model,optimizer=torch.optim.SGD,learning_rate=1e-3,weight_decay=0)
    if checkpoint_path is None:
        checkpoint_path = os.path.join(conf.root_path , 'filter_feature_extractor' , 'checkpoint',net_name,special+criterion.__class__.__name__)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
    order=[i for i in range(len(sample_list))]
    for i in range(epoch):
        total_loss=[0 for k in range(len(sample_list))]
        shuffle(order)

        for j in order:
            sample=sample_list[j]
            net=sample['net']

            filter_label=sample['filter_label']
            label=torch.Tensor(filter_label).reshape((-1,1)).to(device)
            optimizer.zero_grad()
            extractor_model.train()
            output=extractor_model.forward(net,net_name,dataset_name)

            loss=criterion(output,label)
            loss.backward()

            total_loss[j]=float(loss)
            optimizer.step()

        print('{}  Epoch:{}. loss is {}. Sum:'.format(datetime.now(),i, total_loss),end='')
        print(sum(total_loss))
        if i%50==0 and i!=0:


            # ########################################################################
            # sample_list_test = read_data(
            #     path='/home/victorfang/model_pytorch/data/filter_feature_extractor/model_data/vgg16_bn_cifar10/test',
            #     num_images=10000)
            #
            # criterion_test = torch.nn.L1Loss()
            # for sample_test in sample_list_test:
            #     extractor_model.eval()
            #     net_test = sample_test['net']
            #
            #     filter_label_test = sample_test['filter_label']
            #     label_test = torch.Tensor(filter_label_test).reshape((-1, 1)).to(device)
            #
            #     output_test = extractor_model.forward(net_test, net_name=sample_test['net_name'], dataset_name=sample_test['dataset_name'])
            #
            #     loss_test = criterion_test(output_test, label_test)
            #
            #     predict_dead_filter.performance_evaluation(np.array(filter_label_test),
            #                                                output_test.data.detach().cpu().numpy().reshape(-1), 0.1)
            #
            #     print(float(loss_test))
            # ########################################################################



            checkpoint={'feature_len':feature_len,
                        'gcn_rounds':gcn_rounds,
                        'state_dict':extractor_model.state_dict()}

            torch.save(checkpoint,os.path.join(checkpoint_path,str(i)+'.tar'))

    checkpoint = {'feature_len': feature_len,
                  'gcn_rounds': gcn_rounds,
                  'state_dict': extractor_model.state_dict()}

    torch.save(checkpoint, os.path.join(checkpoint_path, str(epoch) + '.tar'))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_extractor(path='/home/victorfang/model_pytorch/data/model_saved/random_net',
    #                 criterion=nn.MSELoss(),special='random_net',net_name='vgg16_bn',dataset_name='cifar10')

    # train_extractor(path='/home/victorfang/model_pytorch/data/filter_feature_extractor/model_data/resnet50/train',
    #                 criterion=nn.MSELoss(),
    #                 net_name='resnet50',
    #                 dataset_name='imagenet',
    #                 feature_len=50,
    #                 num_images=1024,
    #                 special='normalization_learningrate0.1')

    # train_extractor(
    #     train_data_dir='/home/victorfang/model_pytorch/data/filter_feature_extractor/model_data/resnet50/train',
    #     criterion=weighted_MSELoss(),
    #     net_name='resnet50',
    #     dataset_name='imagenet',
    #     feature_len=50,
    #     num_images=1024,
    #     special='normalization_learningrate0.1')
    #
    # train_extractor(path='/home/victorfang/model_pytorch/data/filter_feature_extractor/model_data/vgg16_bn_cifar10/train',
    #                 criterion=nn.MSELoss(),
    #                 net_name='vgg16_bn',
    #                 dataset_name='cifar10',
    #                 feature_len=27,
    #                 num_images=10240,
    #                 special='normalization_')

    # train_extractor(
    #     train_data_dir='/home/victorfang/model_pytorch/data/filter_feature_extractor/model_data/resnet56/train',
    #     criterion=nn.MSELoss(),
    #     net_name='resnet56',
    #     dataset_name='cifar10',
    #     feature_len=10,
    #     num_images=10000,
    #     special='resnet56_gcnround=1',
    #     epoch=700,
    #     gcn_rounds=1,
    # )


    path='/home/victorfang/model_pytorch/data/filter_feature_extractor/checkpoint/resnet56/resnet56_gcnround=1MSELoss/300.tar'
    # path='/home/victorfang/model_pytorch/data/filter_feature_extractor/checkpoint/vgg16_bn_cifar10/weighted_MSELoss/950.tar'
    extractor_model=load_extractor(path).to(device)

    extractor_model.eval()
    sample_list=read_data(path='/home/victorfang/model_pytorch/data/filter_feature_extractor/model_data/resnet56/test',num_images=10000)
    # sample_list=read_data(path='/home/victorfang/model_pytorch/data/filter_feature_extractor/model_data/vgg16_bn_cifar10/test',num_images=10240)

    criterion=torch.nn.L1Loss()
    for sample in sample_list:
        net = sample['net']

        filter_label = sample['filter_label']
        label = torch.Tensor(filter_label).reshape((-1, 1)).to(device)

        output = extractor_model.forward(net,net_name=sample['net_name'],dataset_name=sample['dataset_name'])

        loss = criterion(output, label)

        predict_dead_filter.performance_evaluation(np.array(filter_label),output.data.detach().cpu().numpy().reshape(-1),0.1)

        print(float(loss))
        print()
    print()


    # read_data(num_images=10000)
    #
    #
    # net= vgg.vgg16_bn(pretrained=True).to(device)
    # model=extractor(net=net,feature_len=15,gcn_rounds=3).to(device)
    # c=model.forward()
    # d=torch.sum(c)
    # d.backward()
    # print()