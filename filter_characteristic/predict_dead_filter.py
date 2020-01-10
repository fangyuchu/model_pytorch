import torch
import numpy as np
import os
import torch.nn as nn
from datetime import datetime
import random
from torch import optim
from sklearn.metrics import mean_absolute_error
import copy
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
import math
from network import storage

def read_data(path='/home/victorfang/Desktop/dead_filter(normal_distribution)',
              balance=False,
              regression_or_classification='regression',
              num_images=None,
              sample_num=None):
    #note that classification function is abandoned, the code involved might be wrong
    if regression_or_classification is 'regression':
        filter=[]
        filter_layer=[]
        filter_label=[]
    elif regression_or_classification is 'classification':
        dead_filter = []
        dead_filter_layer = []
        living_filter = []
        living_filter_layer = []
    else:
        raise AttributeError

    file_list = os.listdir(path)
    for file_name in file_list:
        if '.tar' in file_name:

            checkpoint=torch.load(os.path.join(path,file_name))
            net=storage.restore_net(checkpoint)
            net.load_state_dict(checkpoint['state_dict'])
            neural_list=checkpoint['neural_list']
            try:
                module_list=checkpoint['module_list']
            except KeyError:
                module_list=checkpoint['relu_list']
            if regression_or_classification is 'classification':
                # neural_dead_times=checkpoint['neural_dead_times']
                neural_dead_times=8000
                filter_FIRE=checkpoint['filter_FIRE']


            num_conv = 0  # num of conv layers in the network
            filter_num=[]
            filters=[]
            layers=[]
            for mod in net.modules():
                if isinstance(mod, torch.nn.modules.conv.Conv2d):
                    num_conv += 1
                    conv=mod
                elif isinstance(mod,torch.nn.ReLU):                             #ensure the conv are followed by relu
                    if layers != [] and layers[-1] == num_conv - 1:             # get rid of the influence from relu in fc
                        continue
                    filter_num.append(conv.out_channels)
                    filters.append(conv)
                    layers.append(num_conv-1)



            for i in range(len(filters)):
                for module_key in list(neural_list.keys()):
                    if module_list[i] is module_key:                                    #find the neural_list_statistics in layer i+1
                        dead_times=neural_list[module_key]
                        neural_num=dead_times.shape[1]*dead_times.shape[2]  #neural num for one filter
                        filter_weight = filters[i].weight.data.cpu().numpy()

                        if regression_or_classification is 'classification':
                            # judge dead filter by neural_dead_times and dead_filter_ratio
                            dead_times[dead_times<neural_dead_times]=0
                            dead_times[dead_times>=neural_dead_times]=1
                            dead_times=np.sum(dead_times,axis=(1,2))            #count the number of dead neural for one filter
                            dead_filter_index=np.where(dead_times>neural_num*filter_FIRE)[0].tolist()
                            living_filter_index=[i for i in range(filter_num[i]) if i not in dead_filter_index]

                            for ind in dead_filter_index:
                                dead_filter.append(filter_weight[ind])
                            dead_filter_layer+=[i for j in range(len(dead_filter_index))]
                            for ind in living_filter_index:
                                living_filter.append(filter_weight[ind])
                            living_filter_layer += [i for j in range(len(living_filter_index))]
                        else:
                            #compute sum(dead_times)/(num_images*neural_num) as label for each filter
                            dead_times=np.sum(dead_times,axis=(1,2))
                            prediction=dead_times/(neural_num*num_images)
                            for f in filter_weight:
                                filter.append(f)
                            filter_label+=prediction.tolist()
                            filter_layer+=[layers[i] for j in range(filter_weight.shape[0])]

    if regression_or_classification is 'classification' and balance is True:
        living_filter=living_filter[:len(dead_filter)]
        living_filter_layer=living_filter_layer[:len(living_filter_index)]

    if regression_or_classification is 'classification':
        return dead_filter,living_filter,dead_filter_layer,living_filter_layer
    elif regression_or_classification is 'regression':
        if sample_num is not None:
            index=random.sample([i for i in range(len(filter))],sample_num)
            filter=np.array(filter)[index].tolist()
            filter_label=np.array(filter_label)[index].tolist()
            filter_layer=np.array(filter_layer)[index].tolist()
        return filter,filter_label,filter_layer

def trimmed_mean(filter,p):
    filter=filter.flatten()
    min_ind=int(filter.shape[0]*p)
    max_ind=int(filter.shape[0]*(1-p))
    filter=filter[min_ind:max_ind]
    return np.mean(filter)



def cal_F_score(prediction,label,beta=1):
    '''
    calculate f_score for binary classification
    :param prediction: 1-d array, 1 means positive.
    :param label: 1-d array, 1 means positive.
    :param beta: If beta=1, precision is as important as recall. If beta<1, precision is more important. vice versa
    :return:f_score,precision rate,recall rate
    '''
    true_positive=np.bitwise_and(prediction>0, prediction==label)
    false_positive=np.bitwise_xor(prediction>0,true_positive)
    true_negative=np.bitwise_and(prediction==0, prediction==label)
    false_negative=np.bitwise_xor(prediction==0,true_negative)
    #count
    true_positive=np.sum(true_positive)
    false_positive=np.sum(false_positive)
    #true_negative=np.sum(true_negative)
    false_negative=np.sum(false_negative)

    precision=true_positive/(true_positive+false_positive)
    recall=true_positive/(true_positive+false_negative)

    f_score=(1+beta**2)*precision*recall/((beta**2)*precision+recall)

    return f_score,precision,recall


class fc(nn.Module):

    def __init__(self, init_weights=True):
        super(fc, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 2),
        )
        if init_weights:
            self._initialize_weights()
        self.net.to(device)

    def forward(self, x_tensor):
        prediction = self.net(x_tensor)
        return prediction

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def fit(self,x,y):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dead_filter_num=int(np.sum(y))
        sample_num = int(dead_filter_num / 2)  # random choose same number of samples from lf and df

        # all training data
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        y_label=torch.tensor(y, dtype=torch.long).to(device)
        # prepare the label for training
        y_tmp = np.zeros(2 * sample_num, dtype=np.int)
        y_tmp[:sample_num] = 1  # dead filter's label is 1
        y_tensor = torch.tensor(y_tmp, dtype=torch.long).to(device)

        epoch = -1
        highest_accuracy = 0
        highest_precision=0

        LR = 0.01
        optimizer = torch.optim.SGD(self.net.parameters(), lr=LR)
        loss_func = nn.CrossEntropyLoss()

        while True:
            epoch += 1
            ind_df = random.sample([i for i in range(dead_filter_num)], sample_num)
            ind_lf = random.sample([i for i in range(dead_filter_num, x.shape[0])], sample_num)
            ind = torch.tensor(ind_df + ind_lf, dtype=torch.long).to(device)

            self.net.train()

            optimizer.zero_grad()
            output = self.net(torch.index_select(input=x_tensor, dim=0, index=ind))
            loss = loss_func(output, y_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                print("{} epoch:{},   loss is:{}".format(datetime.now(), epoch, loss))

            if epoch % 10000 == 0:
                self.net.eval()
                output = self.net(x_tensor)
                prediction = torch.argmax(output, 1)
                correct = (prediction == y_label).sum().float()
                acc = correct.cpu().detach().data.numpy() / y_label.shape[0]
                print('{} accuracy is {}'.format(datetime.now(), acc))
                f_score, precision, recall = cal_F_score(prediction.cpu().data.numpy(), y)
                print('fc:f_score:{},precision:{},recall:{}'.format(f_score, precision, recall))
                if acc > highest_accuracy and precision>highest_precision:
                    net_saved=copy.deepcopy(self.net)
                    highest_accuracy = acc
                    highest_precision=precision
                if highest_accuracy>0.7 and highest_precision>0.7 and epoch>1000:
                    break
                    # if highest_accuracy > 0.7:
                    #     print("{} Saving self.network...".format(datetime.now()))
                    #     checkpoint = {'network': self.network,
                    #                   'highest_accuracy': acc,
                    #                   'state_dict': self.network.state_dict(),
                    #                   }
                    #     torch.save(checkpoint,
                    #                '/home/victorfang/Desktop/预测死亡神经元的神经网络/accuracy=%.5f.tar' % (acc))
                    #     print("{} network saved ".format(datetime.now()))
        self.net=net_saved
        return self.net

    def predict_proba(self,x):
        #todo:incomplete
        return 1

    def predict(self,x):
        #todo:incomplete
        return 1

class predictor:
    def __init__(self, name):
        self.name=name
        self.min_max_scaler=None
        if name is 'gradient_boosting':
            #todo:容易过拟合
            self.regressor=ensemble.GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                          learning_rate=0.01, loss='ls', max_depth=17,
                          max_features='log2', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=12, min_samples_split=0.005,
                          min_weight_fraction_leaf=0.0, n_estimators=100,
                          n_iter_no_change=None, presort='auto',
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
        elif name is 'random_forest':
            self.regressor=ensemble.RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=17,
                      max_features='sqrt', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=50, min_samples_split=0.001,
                      min_weight_fraction_leaf=0.0, n_estimators=1000,
                      n_jobs=-1, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)

    def fit(self, filter,filter_layer, filter_label,):
        stat,self.min_max_scaler,_,_=statistics(filters=filter,layer=filter_layer,balance_channel=False)
        print('training the regressor')
        if self.name is 'gradient_boosting':
            self.regressor.fit(stat,filter_label)
        if self.name is 'random_forest':
            self.regressor.fit(stat, filter_label)

    # def predict_proba(self,filter):
    #     x,_=statistics(filter,min_max_scaler=self.min_max_scaler)
    #     return self.classifier.predict_proba(x)

    def predict(self,filter,filter_layer):
        # print(self.classifier.best_estimator_)
        stat, _, _, _ = statistics(filters=filter, layer=filter_layer, balance_channel=False,min_max_scaler=self.min_max_scaler)
        return self.regressor.predict(stat)

    def save(self,path):
        print('save regressor in:'+os.path.join(path,self.name+'.m'))
        joblib.dump(self.regressor,os.path.join(path,self.name+'.m'))

    def load(self,path):
        if os.path.exists(os.path.join(path,self.name+'.m')):
            print('load regressor from'+os.path.join(path,self.name+'.m'))
            self.regressor=joblib.load(os.path.join(path,self.name+'.m'))
            return True
        else:
            return False

def read_from_checkpoint(path):
    if '.tar' in path:
        file_list=[path]                                #single network
    else:
        file_list=os.listdir(path)
    filters = []
    layers=[]
    for file_name in file_list:
        if '.tar' in file_name:
            checkpoint=torch.load(os.path.join(path,file_name))
            net=storage.restore_net(checkpoint)
            net.load_state_dict(checkpoint['state_dict'])
            filters_tmp,layers_tmp=get_filters(net=net)
            filters+=filters_tmp
            layers+=layers_tmp
    return filters,layers

def get_filters(net):
    filters=[]
    layers=[]
    num_conv=0
    for mod in net.modules():
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            for f in mod.weight.data.cpu().numpy():
                filters.append(f)
            layers+=[num_conv for i in range(mod.out_channels)]
            num_conv+=1
    return filters,layers

def dead_filter_metric(label,prediction):
    true_positive=np.bitwise_and(prediction>0, prediction==label)
    false_positive=np.bitwise_xor(prediction>0,true_positive)
    true_negative=np.bitwise_and(prediction==0, prediction==label)
    false_negative=np.bitwise_xor(prediction==0,true_negative)
    #count
    true_positive=np.sum(true_positive)
    false_positive=np.sum(false_positive)
    #true_negative=np.sum(true_negative)
    false_negative=np.sum(false_negative)

    precision=true_positive/(true_positive+false_positive)
    recall=true_positive/(true_positive+false_negative)

    beta=0.5
    f_score=(1+beta**2)*precision*recall/((beta**2)*precision+recall)

    return f_score

def statistics(filters,layer,balance_channel=False,min_max_scaler=None,data_num=None,scaler=None,pca=None):
    '''

    :param filters:
    :param layer: layer index of the filter
    :param balance_channel: boolean, whether sample same number of filters in each bin of channel
    :param min_max_scaler:
    :param data_num: total number of filters used for the data returned
    :param scaler:
    :return:
    '''
    feature_num=8
    stat=np.zeros(shape=[len(filters),feature_num],dtype=np.float)
    for i in range(len(filters)):

        stat[i][0]=filters[i].shape[0]                                 #通道数1
        stat[i][1]=layer[i]                                            #哪一层1
        # stat[i][2]=np.sum(np.abs(filters[i]))                          #绝对值求和1
        stat[i][2]=np.sum(filters[i])                                  #求和1
        stat[i][3]=filters[i].max()-filters[i].min()                    #极差1
        stat[i][4]=filters[i].max()                                     #最大值1
        # stat[i][6]=stat[i][10]-stat[i][8]                                #四分位数极差1
        stat[i][6]=np.std(filters[i])                                   #标准差1
        # stat[i][8]=np.percentile(filters[i],25)                         #第一四分位数1
        stat[i][6]=filters[i].min()                                    #最小值1
        # stat[i][10]=np.percentile(filters[i],75)                         #第三四分位数1
        stat[i][7]=np.mean(filters[i])                                  #均值1
        # stat[i][12]=np.median(filters[i])                                #中位数1
        # stat[i][13]=(filters[i].max()+filters[i].min())/2                #中列数(max+min)/21
        # stat[i][14]=trimmed_mean(filters[i],0.2)                         #截断均值（p=0.2）




    # #标准化
    # if scaler is None:
    #     scaler=preprocessing.StandardScaler().fit(stat)
    #     stat=scaler.transform(stat)
    # else:
    #     stat = scaler.transform(stat)

    if min_max_scaler is None:
        #归一化
        min_max_scaler=preprocessing.MinMaxScaler().fit(stat)
        stat=min_max_scaler.transform(stat)
    else:
        stat=min_max_scaler.transform(stat)

    # stat_copy = copy.deepcopy(stat) - np.mean(stat, axis=0)
    # val, vec = np.linalg.eig(np.matmul(stat_copy.T, stat_copy))
    #
    # tmp = -np.sort(-val)
    # if pca is None:
    #     pca=PCA(n_components=15)
    #     pca.fit(stat)
    #     pca.transform(stat)
    # else:
    #     pca.transform(stat)

    if balance_channel is True:
        stat = stat[np.argsort(stat[:, 0])]
        bin = np.histogram(stat[:, 0])[0]
        channel_num_list = bin[bin > 0]

        if data_num is None:
            sample_num = min(channel_num_list)
        else:
            sample_num = int(min(data_num/channel_num_list.shape[0],min(channel_num_list)))

        stat_returned=np.zeros(shape=[sample_num*(len(channel_num_list)),feature_num],dtype=np.float)
        s=0
        for i in range(len(channel_num_list)):
            ind=random.sample([j for j in range(s,s+channel_num_list[i])], sample_num)
            s+=channel_num_list[i]
            stat_returned[i*sample_num:(i+1)*sample_num]=stat[ind]
        stat=stat_returned

    #stat=preprocessing.scale(stat)


    return stat,min_max_scaler,scaler,pca




def cal_ndcg(rel,top_p):
    #考虑dk元素为卷积核的真是死亡度，顺序为预测的死亡度进行排序
    dcg_value = 0.
    idcg_value = 0.
    topk=int(top_p*rel.shape[0])
    log_ki = []

    idcg_rel = rel[np.argsort(-rel)]

    for ki in range(0,topk):
        log_ki.append(math.log(ki+1,2) if math.log(ki+1,2) != 0. else 1.)
        dcg_value += rel[ki] / log_ki[ki]
        idcg_value += idcg_rel[ki] / log_ki[ki]


    # print( "DCG value is " + str(dcg_value))
    # print ("iDCG value is " + str(idcg_value))
    nDCG=dcg_value/idcg_value
    # print('nDCG value is '+str(nDCG))
    return nDCG

def filter_inactive_rate_ndcg(filter_label_val,prediction,ratio):
    # rel=np.array(filter_label_val)[np.argsort(-prediction)]   #以死亡程度为相关度
    arg=np.argsort(-np.array(filter_label_val)) #按死亡度从大到小排序
    rank=np.zeros(shape=len(filter_label_val),dtype=int)
    for i in range(len(filter_label_val)):
        rank[arg[i]]=i              #真实label越大排名越小（靠前）
    dk=rank[np.argsort(-prediction)]#根据预测从大到小排序，dk按序保存了真实的排名，ex[2,1,3],即预测最大，实际排第二
    rel=len(filter_label_val)-dk    #以排名为相关度，排名越小，相关性越大
    ndcg=cal_ndcg(rel,ratio)
    print('ndcg:{}'.format(ndcg))
    return ndcg

def performance_evaluation(label,prediction,ratio):
    data=[]
    #mean absolute error
    loss = mean_absolute_error(label, prediction)
    print('loss:{}'.format(loss))
    data+=[loss]
    #ndcg
    ndcg=filter_inactive_rate_ndcg(label, prediction,ratio)
    data+=[ndcg]
    #f-score,percision and recall
    # true_rank = np.argsort(-np.array(label))[:int(ratio*len(label))]
    # predicted_rank = np.argsort(-prediction)[:int(ratio*len(label))]
    # true_positive=false_positive=0
    # # for i in predicted_rank:
    # #     if i in truth:
    # #         true_positive+=1
    # true_prediction=np.intersect1d(true_rank,predicted_rank)

    #percision in p%
    true_rank = np.argsort(-np.array(label))
    predicted_rank = np.argsort(-prediction)
    i = int(true_rank.shape[0] * ratio)
    for j in [0.1, 0.2, 0.3, 0.4, 0.5]:
        truth_top = true_rank[:int(true_rank.shape[0] * j)]

        if i > truth_top.shape[0]:
            continue
        print('前' + str(j * 100) + '%的准确率:', end='')
        prediction_top = predicted_rank[:i]
        # truth_top = true_rank[:i]
        sum = 0
        for k in prediction_top:
            if k in truth_top:
                sum += 1
        print(sum / i)
        data+=[sum/i]
    print()
    return data


def pca_filter(net,feature_len):
    from transform_conv import conv_to_matrix
    from filter_characteristic.graph_convolutional_network import pca
    weight_list = []
    for name, mod in net.named_modules():
        if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
            weight_list += [conv_to_matrix(mod)]

    gcn_feature_in = []
    for i in range(len(weight_list)):
        gcn_feature_in += [
            pca(weight_list[i], dim=feature_len)]  # reduce the dimension of all filters to same value
    features = gcn_feature_in[0]
    for i in range(1, len(gcn_feature_in)):
        features = torch.cat((features, gcn_feature_in[i]), dim=0)
    features=features.detach().cpu().numpy()
    return features


if __name__ == "__main__":
    # from filter_characteristic import filter_feature_extractor
    # sample_train=filter_feature_extractor.read_data(path='/home/victorfang/model_pytorch/data/filter_feature_extractor/model_data/vgg16_bn_cifar10/train',num_images=10000)
    # sample_val=filter_feature_extractor.read_data(path='/home/victorfang/model_pytorch/data/filter_feature_extractor/model_data/vgg16_bn_cifar10/test',num_images=10000)
    #
    # stat_train=pca_filter(sample_train[0]['net'],15)
    # filter_label_train = np.array(sample_train[0]['filter_label'])
    # for i in range(1,len(sample_train)):
    #     stat_train=np.vstack((stat_train,pca_filter(sample_train[i]['net'],15)))
    #     filter_label_train=np.hstack((filter_label_train,np.array(sample_train[i]['filter_label'])))
    #
    # stat_val = pca_filter(sample_val[0]['net'], 15)
    # filter_label_val = np.array(sample_val[0]['filter_label'])
    # for i in range(1, len(sample_val)):
    #     stat_val = np.vstack((stat_val, pca_filter(sample_val[i]['net'], 15)))
    #     filter_label_val = np.hstack((filter_label_val, np.array(sample_val[i]['filter_label'])))










    ratio=0.1

    #回归#################################################################################################################################################
    filter_train,filter_label_train,filter_layer_train=read_data(num_images=10000,regression_or_classification='regression',
                                                                 path='/home/victorfang/PycharmProjects/model_pytorch/data/filter_feature_extractor/model_data/vgg16_bn_cifar10/train')
    filter_val,filter_label_val,filter_layer_val=read_data(num_images=10000,regression_or_classification='regression',
                                                           path='/home/victorfang/PycharmProjects/model_pytorch/data/filter_feature_extractor/model_data/vgg16_bn_cifar10/test')

    stat_train,min_max_scaler,_,_=statistics(filters=filter_train,layer=filter_layer_train,balance_channel=False,min_max_scaler=None)
    stat_val,_,_,_=statistics(filters=filter_val,layer=filter_layer_val,balance_channel=False,min_max_scaler=min_max_scaler)


    ##use auto_encoder to extract features################################################################################################
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto_encoder = encoder.AutoEncoder().to(device)
    # checkpoint = torch.load('../data/auto_encoder_padrepeat_576d.tar')
    # auto_encoder.load_state_dict(checkpoint['state_dict'])
    # # stat_train,_=auto_encoder.extract_feature(filters=filter_train)
    # # stat_val,_=auto_encoder.extract_feature(filters=filter_val)
    # stat_train_ae,_=auto_encoder.extract_feature(filters=filter_train,pad_mode='repeat')
    # stat_val_ae,_=auto_encoder.extract_feature(filters=filter_val,pad_mode='repeat')

    # stat_train=stat_train_ae
    # stat_val=stat_val_ae

    # stat_train=np.hstack((stat_train,stat_train_ae))                                            #把ae提取的特征和人工的合并
    # stat_val=np.hstack((stat_val,stat_val_ae))
    # #######################################################################################################################3

    ##mlp######################################################################################################
    # from sklearn.neural_network import MLPRegressor
    # print('MLP:')
    # model=MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
    #              beta_2=0.999, early_stopping=True, epsilon=1e-08,
    #              hidden_layer_sizes=120, learning_rate='constant',
    #              learning_rate_init=0.1, max_iter=1000, momentum=0.9,
    #              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
    #              random_state=None, shuffle=True, solver='sgd', tol=0.0001,
    #              validation_fraction=0.1, verbose=False, warm_start=False)
    # model=MLPRegressor(activation='relu', alpha=0, batch_size='auto', beta_1=0.9,
    #          beta_2=0.999, early_stopping=False, epsilon=1e-08,
    #          hidden_layer_sizes=170, learning_rate='constant',
    #          learning_rate_init=0.01, max_iter=1000, momentum=0.9,
    #          n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
    #          random_state=None, shuffle=True, solver='adam', tol=0.0001,
    #          validation_fraction=0.1, verbose=False, warm_start=False)
    # # model.fit(stat_train, filter_label_train)
    # param_grid = {
    #     'hidden_layer_sizes':[100,120,170,150],
    #     'activation':['identity', 'logistic', 'tanh', 'relu'],
    #     'solver':['sgd'],
    #     'alpha':[0.0001,0.001,0.01,0],
    #     'learning_rate':['constant', 'invscaling', 'adaptive'],
    #     'learning_rate_init':[0.1,0.01,0.001,0.0001]
    # }
    # model = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=8)
    # model.fit(stat_train, filter_label_train)
    # print(model.best_estimator_)
    # prediction = model.predict(stat_val)
    # performance_evaluation(filter_label_val,prediction,ratio)
    # print()





    # for i in range(13):
    #     stat_sample = stat_train[np.where(np.array(filter_layer_train) == i)[0]]
    #     sample_label = np.array(filter_label_train)[np.where(np.array(filter_layer_train) == i)[0]]
    #
    #     prediction = model.predict(stat_sample)
    #     loss = mean_absolute_error(sample_label, prediction)
    #     print('{},loss:{}'.format(i, loss))

    ####svm#####################################################################################################
    from sklearn.svm import SVR
    print('svm:')
    model=SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    param_grid = {
        # 'degree':range(1,9,2),
        # 'kernel':['linear',  'rbf', 'sigmoid'],
        # 'gamma':['scale','auto',0.1,0.2,0.3,0.4,0.01],
        # 'coef0':[0,0.01,0.001],
        # 'C':[40,50,60,70,15,20,25,30],
        # 'epsilon':[0.1,0.2,0.5,0.001]

    }
    # model = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=8)
    model.fit(stat_train, filter_label_train)
    # print(model.best_estimator_)
    prediction = model.predict(stat_val)
    performance_evaluation(filter_label_val,prediction,ratio)



    ###random forest########################################################################################
    from sklearn import ensemble
    print('随机森林')
    # for manual features
    model = ensemble.RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=17,
                      max_features='sqrt', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=50, min_samples_split=0.001,
                      min_weight_fraction_leaf=0.0, n_estimators=1000,
                      n_jobs=-1, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)
    # for auto encoder
    # model = ensemble.RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=17,
    #                   max_features='sqrt', max_leaf_nodes=None,
    #                   min_impurity_decrease=0.0, min_impurity_split=None,
    #                   min_samples_leaf=50, min_samples_split=0.001,
    #                   min_weight_fraction_leaf=0.0, n_estimators=1000,
    #                   n_jobs=-1, oob_score=False, random_state=None,
    #                   verbose=0, warm_start=False)

    # param_grid = {
    #     'n_estimators': [50,100,500,1000],#[100* i for i in range(1, 10, 2)],
    #     'min_samples_split':[0.005,0.003,0.002,0.001],
    #     'max_features':['sqrt','log2',None],
    #     'max_leaf_nodes':[None,2,4,6,8,10],
    #     'min_weight_fraction_leaf':[0.001,0],
    #
    #      'max_depth':[i*10+5 for i in range(3)],
    #     'min_samples_leaf':range(10,101,40),
    # }
    # model = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=8)
    model.fit(stat_train, filter_label_train)
    # print(model.best_estimator_)
    prediction = model.predict(stat_val)
    performance_evaluation(filter_label_val,prediction,ratio)


    # 7.GBRT回归###############################################################################################
    from sklearn import ensemble
    print('GBRT',end='')

    model = ensemble.GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                          learning_rate=0.1, loss='ls', max_depth=17,
                          max_features='log2', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=12, min_samples_split=0.005,
                          min_weight_fraction_leaf=0.0, n_estimators=1000,
                          n_iter_no_change=None, presort='auto',
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)

    # param_grid = {
    #     'n_estimators': [50,100,500,1000,1200],#[100* i for i in range(1, 10, 2)],
    #     'learning_rate': [0.05, 0.1, 0.2],
    #     'min_samples_split':[0.007,0.005,0.004,0.003,0.002],
    #     'max_features':['sqrt','log2',None],
    #     'subsample':[0.8,0.9,1],
    #     'max_depth':[11,13,15,17,19],
    #     'min_samples_leaf':[10,12,14,16,18],
    # }
    # model = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=5)
    model.fit(stat_train, filter_label_train)
    prediction = model.predict(stat_val)
    performance_evaluation(filter_label_val,prediction,ratio)


    print('瞎jb猜')
    prediction=np.random.random(size=stat_val.shape[0])
    performance_evaluation(filter_label_val,prediction,ratio)


    print()


    # from sklearn.feature_selection import RFE
    # for i in range(5,13):
    #     print(i)
    #     fs=RFE(estimator=ensemble.GradientBoostingRegressor(n_estimators=100),n_features_to_select=7)
    #     fs.fit(stat_train,filter_label_train)
    #     print(fs.support_)
    #     print(fs.ranking_)
    #     print(mean_absolute_error(filter_label_val,fs.predict(stat_val)))
    #
    #
    #

    #分类#################################################################################################################################################

    # df,lf,df_layer,lf_layer=read_data(num_images=1600,regression_or_classification='classification',balance=False,path='../data/最少样本测试/训练集')
    # df_val,lf_val,df_layer_val,lf_layer_val=read_data(num_images=1600,balance=False,path='../data/最少样本测试/测试集')
    #
    # #df,lf=read_data(balance=False,path='/home/victorfang/Desktop/dead_filter(normal_distribution)')
    #
    # #df_val,lf_val=read_data(balance=True,path='/home/victorfang/Desktop/pytorch_model/vgg16bn_cifar10_dead_neural_normal_tar_acc_decent3/dead_neural',neural_dead_times=1200)
    # _,min_max_scaler,scaler,pca=statistics(df+lf,layer=df_layer+lf_layer,balance_channel=False)
    #
    # stat_df,_,_,_=statistics(df,balance_channel=False,min_max_scaler=min_max_scaler,scaler=scaler,layer=df_layer,pca=pca)
    # stat_lf,_,_,_=statistics(lf,balance_channel=False,min_max_scaler=min_max_scaler,scaler=scaler,layer=lf_layer,pca=pca)#,data_num=stat_df.shape[0])
    # stat_df_val,_,_,_=statistics(df_val,min_max_scaler=min_max_scaler,scaler=scaler,layer=df_layer_val,pca=pca)
    # stat_lf_val,_,_,_=statistics(lf_val,min_max_scaler=min_max_scaler,scaler=scaler,layer=lf_layer_val,pca=pca)
    #
    # # test,_,_,_=statistics(df_val+lf_val,min_max_scaler=min_max_scaler,balance_channel=False,layer=df_layer_val+lf_layer_val)
    #
    # # mean_df=np.mean(stat_df,axis=0)
    # # mean_lf=np.mean(stat_lf,axis=0)
    # #
    # # norm_dead=np.linalg.norm(test-mean_df,ord=2,axis=1)
    # # norm_lived=np.linalg.norm(test-mean_lf,ord=2,axis=1)
    #
    #
    # # tmp1=stat_df.tolist()
    # # tmp1.insert(0,['均值','截断均值','中位数','极差','中列数','第一四分卫数','第三四分位数','四分位数极差','标准差','最大值','最小值','通道数','降维后的参数'])
    # # tmp2=stat_lf.tolist()
    # # tmp2.insert(0,['均值','截断均值','中位数','极差','中列数','第一四分卫数','第三四分位数','四分位数极差','标准差','最大值','最小值','通道数','降维后的参数'])
    # # operate_excel.write_excel(tmp1,'../data/test.xlsx',0,bool_row_append=False)
    # # operate_excel.write_excel(tmp2,'../data/test.xlsx',1,bool_row_append=False)
    #
    # train_x = np.vstack((stat_df, stat_lf))  # dead filters are in the front
    # train_y = np.zeros(train_x.shape[0])
    # train_y[:stat_df.shape[0]]=1
    #
    # val_x=np.vstack((stat_df_val,stat_lf_val))
    # val_y=np.zeros(val_x.shape[0],dtype=np.int)
    # val_y[:stat_df_val.shape[0]]=1
    #
    # ### GBDT(Gradient Boosting Decision Tree) Classifier
    # from sklearn.ensemble import GradientBoostingClassifier
    # print('GradientBoostingClassifier')
    #
    # from sklearn import ensemble
    #
    # print('随机森林',end='')
    # model = ensemble.RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=20,
    #                   max_features='sqrt', max_leaf_nodes=None,
    #                   min_impurity_decrease=0.0, min_impurity_split=None,
    #                   min_samples_leaf=30, min_samples_split=0.001,
    #                   min_weight_fraction_leaf=0.0, n_estimators=100,
    #                   n_jobs=-1, oob_score=False, random_state=None,
    #                   verbose=0, warm_start=False)
    # param_grid = {
    #     # 'n_estimators': [50,100,500,1000],#[100* i for i in range(1, 10, 2)],
    #     # 'min_samples_split':[0.005,0.003,0.002,0.001],
    #     # 'max_features':['sqrt','log2',None],
    #      #'max_depth':[i*10+10 for i in range(11)],
    #     # 'min_samples_leaf':range(10,101,10),
    #     # 'max_leaf_nodes':[None,2,4,6,8,10]
    #     #'min_weight_fraction_leaf':[0.001,0],
    #     # 'n_estimators': range(10, 101, 10)
    # }
    # # model = GridSearchCV(model, param_grid, scoring='f1', n_jobs=-1, cv=8)
    # model.fit(train_x, train_y)
    # # print(model.best_estimator_)
    # prediction = model.predict(val_x)
    # print(classification_report(val_y, prediction))
    #
    #
    #
    #
    # model=GradientBoostingClassifier(criterion='friedman_mse', init=None,
    #                        learning_rate=0.05, loss='deviance', max_depth=15,
    #                        max_features=None, max_leaf_nodes=None,
    #                        min_impurity_decrease=0.0, min_impurity_split=None,
    #                        min_samples_leaf=1, min_samples_split=2,
    #                        min_weight_fraction_leaf=0.0, n_estimators=500,
    #                        n_iter_no_change=None, presort='auto',
    #                        random_state=None, subsample=1.0, tol=0.0001,
    #                        validation_fraction=0.1, verbose=0,
    #                        warm_start=False)
    #
    # param_grid = {
    #     'n_estimators': [50,100,500,1000],#[100* i for i in range(1, 10, 2)],
    #     'learning_rate': [0.05, 0.1, 0.2],
    #     # 'min_samples_split':[0.007,0.005,0.004,0.003,0.002],
    #     # 'max_features':['sqrt','log2',None],
    #     # 'subsample':[0.8,0.9,1],
    #     'max_depth':[3,5,15,25,35,45,55],
    #     # 'min_samples_leaf':[10,12,14,16,18],
    # }
    #
    # model = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=8)
    # model.fit(train_x, train_y)
    # print(model.best_estimator_)
    # prediction=model.predict(val_x)
    # print(classification_report(val_y, prediction))
    #
    # ###AdaBoost Classifier
    # from sklearn.ensemble import AdaBoostClassifier
    # print('AdaBoost')
    # clf = AdaBoostClassifier()
    # clf.fit(train_x, train_y)
    # prediction=clf.predict(val_x)
    # print(classification_report(val_y, prediction))
    #
    # ### Random Forest Classifier
    # from sklearn.ensemble import RandomForestClassifier
    # print('RandomForest')
    # clf = RandomForestClassifier(n_estimators=8)
    # clf.fit(train_x, train_y)
    # prediction = clf.predict(val_x)
    # print(classification_report(val_y, prediction))
    #
    # ### KNN Classifier
    # from sklearn.neighbors import KNeighborsClassifier
    # print('KNN')
    # clf = KNeighborsClassifier()
    # clf.fit(train_x, train_y)
    # prediction = clf.predict(val_x)
    # print(classification_report(val_y, prediction))
    #
    # ### Multinomial Naive Bayes Classifier
    # from sklearn.naive_bayes import MultinomialNB
    # print('Multinomial Naive Bayes Classifier')
    # clf = MultinomialNB(alpha=0.01)
    # clf.fit(train_x, train_y)
    # prediction = clf.predict(val_x)
    # print(classification_report(val_y, prediction))
    #
    #
    # ####svm ##########################################################################################################
    # svc=svm.SVC(kernel='rbf',class_weight='balanced',C=1)
    # param_grid = {
    #     #'C': [2 ** i for i in range(-5, 15, 2)],
    #     'gamma': [2 ** i for i in range(3, -15, -2)],
    # }
    #
    # customize_score={'cm':make_scorer(dead_filter_metric,greater_is_better=True)}
    # clf = GridSearchCV(svc, param_grid, scoring='f1', n_jobs=-1, cv=5)
    #
    # clf.fit(train_x,train_y)
    # # val_x1, _ ,_= statistics(df_val+lf_val, min_max_scaler=min_max_scaler,scaler=scaler,layer=df_layer_val+lf_layer_val)
    # prediction=clf.predict(val_x)
    #
    # f_score,precision,recall=cal_F_score(prediction,val_y)
    # print('svm:f_score:{},precision:{},recall:{}'.format(f_score,precision,recall))
    #
    # print(clf.best_estimator_)
    # print(classification_report(val_y,prediction))
    #
    # ##logistic regression######################################################################################################
    # logistic_regression=LogisticRegressionCV(class_weight='balanced',max_iter=1000,cv=3,Cs=[1, 10,100,1000])
    # re=logistic_regression.fit(train_x,train_y)
    # prediction=re.predict_proba(val_x)
    # prediction=np.argmax(prediction,1)
    # #acc=np.sum(np.argmax(prediction,1)==val_y)/val_y.shape[0]
    # f_score,precision,recall=cal_F_score(prediction,val_y)
    # res=classification_report(y_true=val_y,y_pred=prediction,target_names=['lived filter','dead filter'])
    # print(res)
    # print('logistic regression:f_score:{},precision:{},recall:{}'.format(f_score,precision,recall))
    #
    #
    # ###neural network##########################################################################################################
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # LR=0.01
    # network = fc().to(device)
    # optimizer = torch.optim.SGD(network.parameters(), lr=LR)
    #
    # #optimizer=optim.Adam(network.parameters())
    # #loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([1,0.1601118217]))
    # loss_func = nn.CrossEntropyLoss()
    #
    # sample_num=int(stat_df.shape[0]/2)                                                                      #random choose same number of samples from lf and df
    #
    # #all training data
    # x_tensor = torch.tensor(train_x, dtype=torch.float32).to(device)
    #
    #
    # #validation data
    # val_x_tensor=torch.tensor(val_x,dtype=torch.float32).to(device)
    # val_y_tensor=torch.tensor(val_y,dtype=torch.long).to(device)
    #
    #
    # #prepare the label for training
    # train_y=np.zeros(2*sample_num,dtype=np.int)
    # train_y[:sample_num]=1                                                                                  #dead filter's label is 1
    # y_tensor = torch.tensor(train_y, dtype=torch.long).to(device)
    #
    # epoch=-1
    # highest_accuracy = 0
    #
    #
    # #load previous trained model(optional)
    # # checkpoint=torch.load('/home/victorfang/Desktop/预测死亡神经元的神经网络/accuracy=0.72553.tar')
    # # network.load_state_dict(checkpoint['state_dict'])
    # while True:
    #     epoch+=1
    #     ind_df=random.sample([i for i in range(stat_df.shape[0])],sample_num)
    #     ind_lf=random.sample([i for i in range(stat_df.shape[0],train_x.shape[0])],sample_num)
    #     ind=torch.tensor(ind_df+ind_lf,dtype=torch.long).to(device)
    #
    #     network.train()
    #
    #
    #     optimizer.zero_grad()
    #     output = network(torch.index_select(input=x_tensor,dim=0,index=ind))
    #     loss = loss_func(output,y_tensor)
    #     loss.backward()
    #     optimizer.step()
    #     if epoch%1000==0:
    #         print("{} epoch:{},   loss is:{}".format(datetime.now(),epoch,loss))
    #
    #     if epoch%10000==0 :
    #         network.eval()
    #         output=network(val_x_tensor)
    #         prediction=torch.argmax(output,1)
    #         correct=(prediction==val_y_tensor).sum().float()
    #         acc=correct.cpu().detach().data.numpy()/val_y_tensor.shape[0]
    #         print( '{} accuracy is {}'.format(datetime.now(),acc))
    #         f_score, precision, recall = cal_F_score(prediction.cpu().data.numpy(), val_y)
    #         print('fc:f_score:{},precision:{},recall:{}'.format(f_score, precision, recall))
    #         if acc>highest_accuracy:
    #             highest_accuracy=acc
    #             if highest_accuracy>0.7:
    #                 print("{} Saving network...".format(datetime.now()))
    #                 checkpoint = {'network': network,
    #                               'highest_accuracy': acc,
    #                               'state_dict': network.state_dict(),
    #                               }
    #                 # torch.save(checkpoint,
    #                 #            '/home/victorfang/Desktop/预测死亡神经元的神经网络/accuracy=%.5f.tar' % (acc))
    #                 print("{} network saved ".format(datetime.now()))








