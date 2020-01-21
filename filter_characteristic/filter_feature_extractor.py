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

class extractor(nn.Module):
    def __init__(self,feature_len,gcn_rounds=2,only_gcn=False,only_inner_features=False):
        super(extractor, self).__init__()
        self.only_gcn=only_gcn
        if only_gcn:                                                                        #only use gcn for prediction
            self.gcn = gcn(in_features=feature_len, out_features=1)
        else:
            self.gcn=gcn(in_features=feature_len,out_features=feature_len)
        self.only_inner_features=only_inner_features
        self.feature_len=feature_len
        self.gcn_rounds=gcn_rounds
        if not only_inner_features:
            in_features=feature_len * 2
        else:
            in_features=feature_len
        self.network=nn.Sequential(
            nn.Linear(in_features,128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128,1,bias=False),
            # nn.Sigmoid()
        )
        self.normalization=nn.BatchNorm1d(num_features=in_features)
        
    def forward(self,net,net_name,dataset_name ):
        crosslayer_features=self.gcn.forward(net=net,rounds=self.gcn_rounds,net_name=net_name,dataset_name=dataset_name)
        if self.only_gcn:                                                                      #only use gcn for prediction
            return crosslayer_features

        innerlayer_features=self.extract_innerlayer_features(net)
        if not self.only_inner_features:
            features=torch.cat((crosslayer_features,innerlayer_features),dim=1)
        else:
            features=innerlayer_features
        features=self.normalization(features)
        return self.network(features)

    def extract_innerlayer_features(self,net):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        singular_values=[]
        mean=[]
        max=[]
        std=[]
        channel_num=[]
        filter_num=[]
        for name,mod in net.named_modules():                                                   #mod is a copy
            if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
                filter_num += [mod.out_channels]
                channel_num+=[mod.in_channels]
                weight= transform_conv.conv_to_matrix(copy.deepcopy(mod))
                u, s, v = torch.svd(weight)
                singular_values+=[s[:self.feature_len-5]]
                mean+=[torch.mean(weight,dim=1)]
                max+=[torch.max(weight,dim=1)[0]]
                std+=[torch.std(weight,dim=1)]

        innerlayer_features = torch.zeros((sum(filter_num), self.feature_len )).to(device)
        start=0
        for i in range(len(filter_num)):
            stop = start+filter_num[i]
            innerlayer_features[start:stop][:,0]=i+1                                                                            #layer
            innerlayer_features[start:stop][:,1]=channel_num[i]                                                                  #channel_num
            innerlayer_features[start:stop][:,2]=mean[i]                                                                        #mean
            innerlayer_features[start:stop][:,3]=max[i]                                                                         #max
            innerlayer_features[start:stop][:,4]=std[i]                                                                         #standard deviation
            innerlayer_features[start:stop][:,5:]=singular_values[i].repeat(filter_num[i],1)                                    #top k singuar value
            start=stop
        return innerlayer_features

# class weighted_MSELoss(torch.nn.MSELoss):
#     def __init__(self):
#         super(weighted_MSELoss,self).__init__()
#
#     def forward(self, input, target):
#         device=input.device
#         ret = (input - target) ** 2
#         weight=torch.zeros(target.shape).to(device)
#         weight[target<0.3]=0.3
#         weight[target>=0.3]=target[target>=0.3]
#         ret = ret * weight
#         if self.reduction != 'none':
#             ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
#         return ret




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
    try:
        only_gcn=checkpoint['only_gcn']
    except KeyError:
        only_gcn=False
    try:
        only_inner_features=checkpoint['only_inner_features']
    except KeyError:
        only_inner_features=False
    net=extractor(feature_len=feature_len,gcn_rounds=gcn_rounds,only_gcn=only_gcn,only_inner_features=only_inner_features).to(device)
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
            # from framework import measure_flops
            # measure_flops.measure_model(net, 'cifar10',print_flop=True)

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
                    checkpoint_path=None,
                    only_gcn=False,
                    only_inner_features=False):
    print(criterion)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor_model=extractor(feature_len=feature_len,gcn_rounds=gcn_rounds,only_gcn=only_gcn,only_inner_features=only_inner_features).to(device)
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
        if i%10==0 and i!=0:
            checkpoint={'feature_len':feature_len,
                        'gcn_rounds':gcn_rounds,
                        'state_dict':extractor_model.state_dict(),
                        'only_gcn':only_gcn,
                        'only_inner_features':only_inner_features,}
            torch.save(checkpoint,os.path.join(checkpoint_path,str(i)+'.tar'))

    checkpoint = {'feature_len': feature_len,
                  'gcn_rounds': gcn_rounds,
                  'state_dict': extractor_model.state_dict(),
                  'only_gcn':only_gcn,
                  'only_inner_features':only_inner_features,}

    torch.save(checkpoint, os.path.join(checkpoint_path, str(epoch) + '.tar'))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # train_extractor(train_data_dir=os.path.join(conf.root_path,'filter_feature_extractor/model_data/vgg16_bn_cifar10/train'),
    #                 criterion=nn.MSELoss(),
    #                 net_name='vgg16_bn',
    #                 dataset_name='cifar10',
    #                 feature_len=15,
    #                 num_images=10000,
    #                 special='combined_innerfeatures',
    #                 only_gcn=False,
    #                 epoch=150)

    # train_extractor(train_data_dir=os.path.join(conf.root_path,'filter_feature_extractor/model_data/vgg16_bn_cifar10/train'),
    #                 criterion=nn.MSELoss(),
    #                 net_name='vgg16_bn',
    #                 dataset_name='cifar10',
    #                 feature_len=15,
    #                 num_images=10000,
    #                 special='only_innerfeatures',
    #                 only_gcn=False,
    #                 epoch=150,
    #                 only_inner_features=True)

    # train_extractor(train_data_dir=os.path.join(conf.root_path,'filter_feature_extractor/model_data/vgg16_bn_cifar10/train'),
    #                 criterion=nn.MSELoss(),
    #                 net_name='vgg16_bn',
    #                 dataset_name='cifar10',
    #                 feature_len=15,
    #                 num_images=10000,
    #                 special='only_gcn',
    #                 only_gcn=True,
    #                 epoch=150,
    #                 only_inner_features=False)


    #
    # train_extractor(train_data_dir=os.path.join(conf.root_path,'filter_feature_extractor/model_data/resnet56/train'),
    #                 criterion=nn.MSELoss(),
    #                 net_name='resnet56',
    #                 dataset_name='cifar10',
    #                 feature_len=10,
    #                 num_images=10000,
    #                 special='combined_innerfeatures',
    #                 only_gcn=False,
    #                 epoch=500,
    #                 gcn_rounds=1)
    #
    # train_extractor(train_data_dir=os.path.join(conf.root_path,'filter_feature_extractor/model_data/resnet56/train'),
    #                 criterion=nn.MSELoss(),
    #                 net_name='resnet56',
    #                 dataset_name='cifar10',
    #                 feature_len=10,
    #                 num_images=10000,
    #                 special='only_innerfeatures',
    #                 only_gcn=False,
    #                 epoch=500,
    #                 only_inner_features=True,
    #                 gcn_rounds=1)
    #
    # train_extractor(train_data_dir=os.path.join(conf.root_path,'filter_feature_extractor/model_data/resnet56/train'),
    #                 criterion=nn.MSELoss(),
    #                 net_name='resnet56',
    #                 dataset_name='cifar10',
    #                 feature_len=10,
    #                 num_images=10000,
    #                 special='only_gcn',
    #                 only_gcn=True,
    #                 epoch=500,
    #                 only_inner_features=False,
    #                 gcn_rounds=1)






    sample_list=read_data(path='/home/victorfang/model_pytorch/data/model_saved/vgg16_extractor_static_imagenet/dead_neural',num_images=1000)
    # sample_list=read_data(path=os.path.join(conf.root_path,'filter_feature_extractor/model_data/vgg16_bn_cifar10/test'),num_images=10000)

    num_epoch='150'

    path_list=[]
    # path_list+=[os.path.join(conf.root_path,'filter_feature_extractor/checkpoint/vgg16_bn/combined_innerfeaturesMSELoss/'+num_epoch+'.tar')]
    # path_list+=[os.path.join(conf.root_path,'filter_feature_extractor/checkpoint/vgg16_bn/only_gcnMSELoss/'+num_epoch+'.tar')]
    # path_list+=[os.path.join(conf.root_path,'filter_feature_extractor/checkpoint/vgg16_bn/only_innerfeaturesMSELoss/'+num_epoch+'.tar')]

    # path_list+=[os.path.join(conf.root_path,'filter_feature_extractor/checkpoint/resnet56/combined_innerfeaturesMSELoss/'+num_epoch+'.tar')]
    # path_list+=[os.path.join(conf.root_path,'filter_feature_extractor/checkpoint/resnet56/only_gcnMSELoss/'+'500'+'.tar')]
    # path_list+=[os.path.join(conf.root_path,'filter_feature_extractor/checkpoint/resnet56/only_innerfeaturesMSELoss/'+'500'+'.tar')]
    path_list+=['/home/victorfang/model_pytorch/data/model_saved/vgg16_extractor_static_imagenet/extractor/10.tar']
    stat={}
    for path in path_list:
        print(path)
        data = []
        extractor_model=load_extractor(path).to(device)
        extractor_model.eval()
        criterion=torch.nn.L1Loss()
        for sample in sample_list:
            net = sample['net']

            filter_label = sample['filter_label']
            label = torch.Tensor(filter_label).reshape((-1, 1)).to(device)

            output = extractor_model.forward(net,net_name=sample['net_name'],dataset_name=sample['dataset_name'])

            loss = criterion(output, label)

            data+=[predict_dead_filter.performance_evaluation(np.array(filter_label),output.data.detach().cpu().numpy().reshape(-1),0.1)]

        data=np.array(data)
        data=np.mean(data,axis=0)
        stat[path.split('/')[-2]]=data
        print(data)
        print()
    print(stat)



    # read_data(num_images=10000)
    #
    #
    # net= vgg.vgg16_bn(pretrained=True).to(device)
    # model=extractor(net=net,feature_len=15,gcn_rounds=3).to(device)
    # c=model.forward()
    # d=torch.sum(c)
    # d.backward()
    # print()