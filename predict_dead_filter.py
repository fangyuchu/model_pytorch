import torch
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV
import os
import torch.nn as nn
from datetime import datetime
import random
from torch import optim
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import SCORERS
import copy
import operate_excel
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn import ensemble
import matplotlib.pyplot as plt


def read_data(path='/home/victorfang/Desktop/dead_filter(normal_distribution)',
              balance=False,
              regression_or_classification='classification',
              batch_size=None):
    if regression_or_classification is 'regression':
        filter=list()
        filter_layer=list()
        filter_label=list()
    elif regression_or_classification is 'classification':
        dead_filter = list()
        dead_filter_layer = list()
        living_filter = list()
        living_filter_layer = list()
    else:
        raise AttributeError

    file_list = os.listdir(path)
    for file_name in file_list:
        if '.tar' in file_name:

            checkpoint=torch.load(os.path.join(path,file_name))
            net=checkpoint['net']
            net.load_state_dict(checkpoint['state_dict'])
            neural_list=checkpoint['neural_list']
            module_list=checkpoint['relu_list']
            if regression_or_classification is 'classification':
                # neural_dead_times=checkpoint['neural_dead_times']
                neural_dead_times=8000
                filter_dead_ratio=checkpoint['filter_dead_ratio']
            if batch_size is None:
                batch_size=checkpoint['batch_size']

            num_conv = 0  # num of conv layers in the net
            filter_num=list()
            filters=list()
            for mod in net.features:
                if isinstance(mod, torch.nn.modules.conv.Conv2d):
                    num_conv += 1
                    filter_num.append(mod.out_channels)
                    filters.append(mod)

            for i in range(num_conv):
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
                            dead_filter_index=np.where(dead_times>neural_num*filter_dead_ratio)[0].tolist()
                            living_filter_index=[i for i in range(filter_num[i]) if i not in dead_filter_index]

                            for ind in dead_filter_index:
                                dead_filter.append(filter_weight[ind])
                            dead_filter_layer+=[i for j in range(len(dead_filter_index))]
                            for ind in living_filter_index:
                                living_filter.append(filter_weight[ind])
                            living_filter_layer += [i for j in range(len(living_filter_index))]
                        else:
                            #compute sum(dead_times)/(batch_size*neural_num) as label for each filter
                            dead_times=np.sum(dead_times,axis=(1,2))
                            prediction=dead_times/(neural_num*batch_size)
                            for f in filter_weight:
                                filter.append(f)
                            filter_label+=prediction.tolist()
                            filter_layer+=[i for j in range(filter_weight.shape[0])]

    if regression_or_classification is 'classification' and balance is True:
        living_filter=living_filter[:len(dead_filter)]
        living_filter_layer=living_filter_layer[:len(living_filter_index)]

    if regression_or_classification is 'classification':
        return dead_filter,living_filter,dead_filter_layer,living_filter_layer
    elif regression_or_classification is 'regression':
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
                    #     print("{} Saving self.net...".format(datetime.now()))
                    #     checkpoint = {'net': self.net,
                    #                   'highest_accuracy': acc,
                    #                   'state_dict': self.net.state_dict(),
                    #                   }
                    #     torch.save(checkpoint,
                    #                '/home/victorfang/Desktop/预测死亡神经元的神经网络/accuracy=%.5f.tar' % (acc))
                    #     print("{} net saved ".format(datetime.now()))
        self.net=net_saved
        return self.net

    def predict_proba(self,x):
        #todo:incomplete
        return 1

    def predict(self,x):
        #todo:incomplete
        return 1

class predictor:
    def __init__(self, name,**kargs):
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


    def fit(self, filter,filter_layer, filter_label,):
        stat,self.min_max_scaler,_,_=statistics(filters=filter,layer=filter_layer,balance_channel=False)
        if self.name is 'gradient_boosting':
            self.regressor.fit(stat,filter_label)


    # def predict_proba(self,filter):
    #     x,_=statistics(filter,min_max_scaler=self.min_max_scaler)
    #     return self.classifier.predict_proba(x)

    def predict(self,filter,filter_layer):
        # print(self.classifier.best_estimator_)
        stat, _, _, _ = statistics(filters=filter, layer=filter_layer, balance_channel=False,min_max_scaler=self.min_max_scaler)
        return self.regressor.predict(stat)


# class predictor:
#     def __init__(self, name,**kargs):
#         self.name=name
#         self.min_max_scaler=None
#         if name is 'svm':
#             self.classifier = svm.SVC(kernel=kargs['kernel'], class_weight='balanced')
#         elif name is 'logistic_regression':
#             self.classifier = LogisticRegressionCV(class_weight='balanced', max_iter=1000, cv=3, Cs=[1, 10, 100, 1000])
#         elif name is 'fc':
#             self.classifier = fc()
#
#     def fit(self, dead_filter, lived_filter):
#         _, self.min_max_scaler =statistics(dead_filter + lived_filter)
#
#         stat_df, _ = statistics(dead_filter, balance_channel=True, min_max_scaler=self.min_max_scaler)
#         stat_lf, _ = statistics(lived_filter, balance_channel=True, min_max_scaler=self.min_max_scaler)
#         train_x = np.vstack((stat_df, stat_lf))  # dead filters are in the front
#         train_y = np.zeros(train_x.shape[0])
#         train_y[:stat_df.shape[0]] = 1  # dead filters are labeled 1
#         if self.name is 'svm':
#             param_grid = {
#                 'C': [2 ** i for i in range(-5, 15, 2)],
#                 'gamma': [2 ** i for i in range(3, -15, -2)],
#             }
#             self.classifier = GridSearchCV(self.classifier, param_grid, scoring='f1', n_jobs=-1, cv=5)
#             self.classifier = self.classifier.fit(train_x, train_y)
#             print(self.classifier.best_estimator_)
#         elif self.name is 'logistic_regression':
#             self.classifier = self.classifier.fit(train_x, train_y)
#         elif self.name is 'fc':
#             self.classifier=self.classifier.fit(train_x,train_y)
#
#     def predict_proba(self,filter):
#         x,_=statistics(filter,min_max_scaler=self.min_max_scaler)
#         return self.classifier.predict_proba(x)
#
#     def predict(self,filter):
#         print(self.classifier.best_estimator_)
#         x, _ = statistics(filter, min_max_scaler=self.min_max_scaler)
#         return self.classifier.predict(x)


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
    feature_num=24
    stat=np.zeros(shape=[len(filters),feature_num],dtype=np.float)
    for i in range(len(filters)):
        # stat[i][0]=np.mean(filters[i])                                  #均值
        # stat[i][1]=trimmed_mean(filters[i],0.2)                         #截断均值（p=0.2）
        # stat[i][2]=np.median(filters[i])                                #中位数
        # stat[i][3]=filters[i].max()-filters[i].min()                    #极差
        # stat[i][4]=(filters[i].max()+filters[i].min())/2                #中列数(max+min)/2
        # stat[i][5]=np.percentile(filters[i],25)                         #第一四分位数
        # stat[i][6]=np.percentile(filters[i],75)                         #第三四分位数
        # stat[i][7]=stat[i][6]-stat[i][5]                                #四分位数极差
        # stat[i][8]=np.std(filters[i])                                   #标准差
        # stat[i][9]=filters[i].max()                                     #最大值
        # stat[i][10]=filters[i].min()                                    #最小值
        # stat[i][11]=filters[i].shape[0]                                 #通道数
        # stat[i][12]=layer[i]                                            #哪一层
        # stat[i][13:22]=np.mean(filters[i],axis=0).flatten()               #降维后的卷积核参数
        # stat[i][22]=np.sum(filters[i])                                  #求和
        # stat[i][23]=np.sum(np.abs(filters[i]))                          #绝对值求和

        stat[i][0]=filters[i].shape[0]                                 #通道数
        stat[i][1]=layer[i]                                            #哪一层
        stat[i][2]=np.sum(np.abs(filters[i]))                          #绝对值求和
        stat[i][3]=np.sum(filters[i])                                  #求和
        stat[i][4]=filters[i].max()-filters[i].min()                    #极差
        stat[i][5]=filters[i].max()                                     #最大值
        stat[i][6]=stat[i][6]-stat[i][5]                                #四分位数极差
        stat[i][7]=np.std(filters[i])                                   #标准差
        stat[i][8]=np.percentile(filters[i],25)                         #第一四分位数
        stat[i][9]=filters[i].min()                                    #最小值
        stat[i][10]=np.percentile(filters[i],75)                         #第三四分位数
        stat[i][11]=np.mean(filters[i])                                  #均值
        stat[i][12]=np.median(filters[i])                                #中位数
        stat[i][13]=(filters[i].max()+filters[i].min())/2                #中列数(max+min)/2
        stat[i][14:23]=np.mean(filters[i],axis=0).flatten()               #降维后的卷积核参数
        stat[i][23]=trimmed_mean(filters[i],0.2)                         #截断均值（p=0.2）





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

if __name__ == "__main__":
    # import sklearn
    # print(sorted(sklearn.metrics.SCORERS.keys()))

    #回归#################################################################################################################################################
    filter_train,filter_label_train,filter_layer_train=read_data(batch_size=10000,regression_or_classification='regression',path='./最少样本测试/训练集')
    filter_val,filter_label_val,filter_layer_val=read_data(batch_size=10000,regression_or_classification='regression',path='./最少样本测试/测试集')




    #汇至cdf和pdf##################################################################################################
    # for i in range(13):
    #     filter_label_train_tmp=np.array(filter_label_train)[np.where(np.array(filter_layer_train)==i)]
    #     title='randomdata_cdf:layer'+str(i)+'round:1'
    #     label=1-np.array(filter_label_train_tmp)
    #     plt.figure()
    #     plt.title(title)
    #     plt.hist(label,cumulative=True,histtype='step',bins=100) #cumulative=False为pdf，true为cdf
    #     plt.xlabel('filter activation ratio')
    #     plt.ylabel('number of filters')
    #     plt.legend()
    #     # plt.show()
    #     plt.savefig(title+'.png', format='png')



    # _, min_max_scaler, scaler, pca = statistics(filter_train, layer=filter_layer_train, balance_channel=False)
    stat_train,min_max_scaler,_,_=statistics(filters=filter_train,layer=filter_layer_train,balance_channel=False,min_max_scaler=None)
    stat_val,_,_,_=statistics(filters=filter_val,layer=filter_layer_val,balance_channel=False,min_max_scaler=min_max_scaler)

    # from sklearn.feature_selection import VarianceThreshold
    #
    # # 方差选择法，返回值为特征选择后的数据
    # # 参数threshold为方差的阈值
    # vt=VarianceThreshold(threshold=0.0005)
    # vt.fit_transform(stat_train)
    # print(vt._get_support_mask())
    #
    # var=np.std(stat_train,axis=0)
    # np.argsort(-var)



    #
    #
    # from sklearn import ensemble
    # print('bagging',end='')
    # model=ensemble.BaggingRegressor()
    # model.fit(stat_train,filter_label_train)
    # prediction=model.predict(stat_val)
    # c=mean_absolute_error(filter_label_val,prediction)
    # print(c)
    #
    # from sklearn import ensemble
    # print('adaboost',end='')
    # model = ensemble.AdaBoostRegressor(n_estimators=100)  # 这里使用50个决策树
    # model.fit(stat_train,filter_label_train)
    # prediction=model.predict(stat_val)
    # c=mean_absolute_error(filter_label_val,prediction)
    # print(c)

    from sklearn import ensemble
    print('???')
    print('随机森林')

    model = ensemble.RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=17,
                      max_features='sqrt', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=50, min_samples_split=0.001,
                      min_weight_fraction_leaf=0.0, n_estimators=1000,
                      n_jobs=-1, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)
    param_grid = {
        # 'n_estimators': [50,100,500,1000],#[100* i for i in range(1, 10, 2)],
        # 'min_samples_split':[0.005,0.003,0.002,0.001],
        # 'max_features':['sqrt','log2',None],
         #'max_depth':[i*10+10 for i in range(11)],
        'min_samples_leaf':range(10,101,40),
        # 'max_leaf_nodes':[None,2,4,6,8,10]
        #'min_weight_fraction_leaf':[0.001,0],
        # 'n_estimators': range(10, 101, 10)
    }
    # model = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=8)



    # test_arg=random.sample([i for i in range(stat_train.shape[0])],1000)
    # train_arg=[i for i in range(stat_train.shape[0]) if i not in test_arg]
    #
    # filter_label_train=np.array(filter_label_train)
    #
    # model.fit(stat_train[train_arg], filter_label_train[train_arg])
    # print(model.best_estimator_)
    #
    #
    # prediction = model.predict(stat_train[test_arg])
    # loss = mean_absolute_error(filter_label_train[test_arg], prediction)
    # print('loss:{}'.format( loss))
    # print(mean_absolute_error(filter_label_train[train_arg], model.predict(stat_train[train_arg])))

    model.fit(stat_train, filter_label_train)

    # print(model.best_estimator_)
    # print(model.oob_score_)

    prediction = model.predict(stat_val)
    loss = mean_absolute_error(filter_label_val, prediction)



    print('loss:{}'.format(loss))
    truth = np.argsort(-np.array(filter_label_val))
    prediction_argsort = np.argsort(-prediction)
    i = int(truth.shape[0] * 0.1)
    print(i)
    for j in [0.2, 0.3, 0.4, 0.5]:
        print('j:' + str(j))
        truth_top1000 = truth[:int(truth.shape[0] * j)]

        if i >= truth_top1000.shape[0]:
            continue
        prediction_top = prediction_argsort[:i]
        # truth_top1000 = truth[:i]
        sum = 0
        for k in prediction_top:
            if k in truth_top1000:
                sum += 1
        print(sum / i)

    # prediction = model.predict(stat_train[test_arg])
    # loss = mean_absolute_error(filter_label_train[test_arg], prediction)
    # print('loss:{}'.format( loss))
    # print(mean_absolute_error(filter_label_train[train_arg], model.predict(stat_train[train_arg])))




    for i in range(13):
        stat_sample=stat_train[np.where(np.array(filter_layer_train)==i)[0]]
        sample_label=np.array(filter_label_train)[np.where(np.array(filter_layer_train)==i)[0]]

        prediction = model.predict(stat_sample)
        loss = mean_absolute_error(sample_label, prediction)
        print('{},loss:{}'.format(i,loss))


    prediction = model.predict(stat_val)
    loss = mean_absolute_error(filter_label_val, prediction)
    print('loss:{}'.format(loss))
    # print(model.best_estimator_)




    # 7.GBRT回归
    from sklearn import ensemble
    print('GBRT',end='')

    # model = ensemble.GradientBoostingRegressor(n_estimators=1000,learning_rate=0.1)  # 这里使用100个决策树
    # mt=predictor(name='gradient_boosting')
    # mt.fit(stat_train,filter_label_train)
    # prediction=mt.predict(stat_val)
    # c = mean_absolute_error(filter_label_val, prediction)
    # print(c)


    # model = ensemble.GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
    #                       learning_rate=0.1, loss='ls', max_depth=13,
    #                       max_features='sqrt', max_leaf_nodes=None,
    #                       min_impurity_decrease=0.0, min_impurity_split=None,
    #                       min_samples_leaf=1, min_samples_split=0.005,
    #                       min_weight_fraction_leaf=0.0, n_estimators=1000,
    #                       n_iter_no_change=None, presort='auto',
    #                       random_state=None, subsample=1.0, tol=0.0001,
    #                       validation_fraction=0.1, verbose=0, warm_start=False)

    model = ensemble.GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                          learning_rate=0.1, loss='ls', max_depth=17,
                          max_features='log2', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=12, min_samples_split=0.005,
                          min_weight_fraction_leaf=0.0, n_estimators=1000,
                          n_iter_no_change=None, presort='auto',
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)

    model.fit(stat_train, filter_label_train)

    prediction = model.predict(stat_val)
    loss = mean_absolute_error(filter_label_val, prediction)
    print('loss:{}'.format(loss))
    # param_grid = {
    #     'n_estimators': [50,100,500,1000,1200],#[100* i for i in range(1, 10, 2)],
    #     'learning_rate': [0.05, 0.1, 0.2],
    #     'min_samples_split':[0.007,0.005,0.004,0.003,0.002],
    #     'max_features':['sqrt','log2',None],
    #     'subsample':[0.8,0.9,1],
    #     'max_depth':[11,13,15,17,19],
    #     'min_samples_leaf':[10,12,14,16,18],
    # }


    n_estimators= [50,100,500,1000,1200]#[100* i for i in range(1, 10, 2)],
    learning_rate= [0.05, 0.1, 0.2]
    min_samples_split=[0.007,0.005,0.004,0.003,0.002]
    max_features=['sqrt','log2',None]
    subsample=[0.8,0.9,1]
    max_depth=[11,13,15,17,19]
    min_samples_leaf=[10,12,14,16,18]

    min_comb=list()
    small_comb=list()
    min_loss=100
    for n_e in n_estimators:
        for lr in learning_rate:
            for mss in min_samples_split:
                for mf in max_features:
                    for subsam in subsample:
                        for md in max_depth:
                            for msl in min_samples_leaf:
                                model = ensemble.GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse',
                                                                           init=None,
                                                                           learning_rate=lr, loss='ls', max_depth=md,
                                                                           max_features=mf, max_leaf_nodes=None,
                                                                           min_impurity_decrease=0.0,
                                                                           min_impurity_split=None,
                                                                           min_samples_leaf=msl, min_samples_split=mss,
                                                                           min_weight_fraction_leaf=0.0,
                                                                           n_estimators=n_e,
                                                                           n_iter_no_change=None, presort='auto',
                                                                           random_state=None, subsample=subsam, tol=0.0001,
                                                                           validation_fraction=0.1, verbose=0,
                                                                           warm_start=False)

                                print(n_e,lr,mss,mf,subsam,md,msl)

                                # model = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=5)
                                model.fit(stat_train,filter_label_train)

                                prediction=model.predict(stat_val)
                                loss=mean_absolute_error(filter_label_val,prediction)
                                print('loss:{}'.format(loss))
                                if loss<0.2:
                                    small_comb.append([n_e,lr,mss,mf,subsam,md,msl])
                                if loss<min_loss:
                                    min_comb=[n_e,lr,mss,mf,subsam,md,msl]
                                    min_loss=loss

                                truth = np.argsort(-np.array(filter_label_val))
                                prediction_argsort=np.argsort(-prediction)
                                i = int(truth.shape[0] * 0.1)
                                print(i)
                                for j in [0.2,0.3,0.4,0.5]:
                                    print('j:'+str(j))
                                    truth_top1000 = truth[:int(truth.shape[0] *j)]

                                    if i>=truth_top1000.shape[0]:
                                        continue
                                    prediction_top=prediction_argsort[:i]
                                    # truth_top1000 = truth[:i]
                                    sum=0
                                    for k in prediction_top:
                                        if k in truth_top1000:
                                            sum+=1
                                    print(sum/i)
                                print('-----------------------------------------------------')


    print('min_comb:{},loss:{}'.format(min_comb,min_loss))
    print('small_comb:{}'.format(small_comb))








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
    # print('瞎jb猜',end='')
    # prediction=np.random.random(size=stat_val.shape[0])
    # c=mean_absolute_error(filter_label_val,prediction)
    # print(c)
    #分类#################################################################################################################################################

    # df,lf,df_layer,lf_layer=read_data(batch_size=1600,regression_or_classification='classification',balance=False,path='./最少样本测试/训练集')
    # df_val,lf_val,df_layer_val,lf_layer_val=read_data(batch_size=1600,balance=False,path='./最少样本测试/测试集')
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
    # # operate_excel.write_excel(tmp1,'./test.xlsx',0,bool_row_append=False)
    # # operate_excel.write_excel(tmp2,'./test.xlsx',1,bool_row_append=False)
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
    # net = fc().to(device)
    # optimizer = torch.optim.SGD(net.parameters(), lr=LR)
    #
    # #optimizer=optim.Adam(net.parameters())
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
    # # net.load_state_dict(checkpoint['state_dict'])
    # while True:
    #     epoch+=1
    #     ind_df=random.sample([i for i in range(stat_df.shape[0])],sample_num)
    #     ind_lf=random.sample([i for i in range(stat_df.shape[0],train_x.shape[0])],sample_num)
    #     ind=torch.tensor(ind_df+ind_lf,dtype=torch.long).to(device)
    #
    #     net.train()
    #
    #
    #     optimizer.zero_grad()
    #     output = net(torch.index_select(input=x_tensor,dim=0,index=ind))
    #     loss = loss_func(output,y_tensor)
    #     loss.backward()
    #     optimizer.step()
    #     if epoch%1000==0:
    #         print("{} epoch:{},   loss is:{}".format(datetime.now(),epoch,loss))
    #
    #     if epoch%10000==0 :
    #         net.eval()
    #         output=net(val_x_tensor)
    #         prediction=torch.argmax(output,1)
    #         correct=(prediction==val_y_tensor).sum().float()
    #         acc=correct.cpu().detach().data.numpy()/val_y_tensor.shape[0]
    #         print( '{} accuracy is {}'.format(datetime.now(),acc))
    #         f_score, precision, recall = cal_F_score(prediction.cpu().data.numpy(), val_y)
    #         print('fc:f_score:{},precision:{},recall:{}'.format(f_score, precision, recall))
    #         if acc>highest_accuracy:
    #             highest_accuracy=acc
    #             if highest_accuracy>0.7:
    #                 print("{} Saving net...".format(datetime.now()))
    #                 checkpoint = {'net': net,
    #                               'highest_accuracy': acc,
    #                               'state_dict': net.state_dict(),
    #                               }
    #                 # torch.save(checkpoint,
    #                 #            '/home/victorfang/Desktop/预测死亡神经元的神经网络/accuracy=%.5f.tar' % (acc))
    #                 print("{} net saved ".format(datetime.now()))








