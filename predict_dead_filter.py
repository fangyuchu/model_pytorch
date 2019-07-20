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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import SCORERS
import copy
import operate_excel
from sklearn import preprocessing






def read_data(path='/home/victorfang/Desktop/dead_filter(normal_distribution)',
              balance=False,
              neural_dead_times=999,
              filter_dead_ratio=0.85):
    dead_filter =list()
    dead_filter_layer=list()
    living_filter = list()
    living_filter_layer=list()
    file_list = os.listdir(path)
    for file_name in file_list:
        if '.tar' in file_name:

            checkpoint=torch.load(os.path.join(path,file_name))
            net=checkpoint['net']
            net.load_state_dict(checkpoint['state_dict'])
            neural_list=checkpoint['neural_list']
            relu_list=checkpoint['relu_list']

            num_conv = 0  # num of conv layers in the net
            filter_num=list()
            filters=list()
            for mod in net.features:
                if isinstance(mod, torch.nn.modules.conv.Conv2d):
                    num_conv += 1
                    filter_num.append(mod.out_channels)
                    filters.append(mod)

            for i in range(num_conv):
                for relu_key in list(neural_list.keys()):
                    if relu_list[i] is relu_key:                                    #find the neural_list_statistics in layer i+1
                        dead_relu_list=neural_list[relu_key]
                        neural_num=dead_relu_list.shape[1]*dead_relu_list.shape[2]  #neural num for one filter

                        # judge dead filter by neural_dead_times and dead_filter_ratio
                        dead_relu_list[dead_relu_list<neural_dead_times]=0
                        dead_relu_list[dead_relu_list>=neural_dead_times]=1
                        dead_relu_list=np.sum(dead_relu_list,axis=(1,2))            #count the number of dead neural for one filter
                        dead_filter_index=np.where(dead_relu_list>neural_num*filter_dead_ratio)[0].tolist()
                        living_filter_index=[i for i in range(filter_num[i]) if i not in dead_filter_index]

                        filter_weight=filters[i].weight.data.cpu().numpy()
                        for ind in dead_filter_index:
                            dead_filter.append(filter_weight[ind])
                        dead_filter_layer+=[i for j in range(len(dead_filter_index))]
                        for ind in living_filter_index:
                            living_filter.append(filter_weight[ind])
                        living_filter_layer += [i for j in range(len(living_filter_index))]

    if balance is True:
        living_filter=living_filter[:len(dead_filter)]
        living_filter_layer=living_filter_layer[:len(living_filter_index)]
    return dead_filter,living_filter,dead_filter_layer,living_filter_layer

def trimmed_mean(filter,p):
    filter=filter.flatten()
    min_ind=int(filter.shape[0]*p)
    max_ind=int(filter.shape[0]*(1-p))
    filter=filter[min_ind:max_ind]
    return np.mean(filter)

def statistics(filters,layer,balance_channel=False,min_max_scaler=None,data_num=None,scaler=None):
    '''

    :param filters:
    :param layer: layer index of the filter
    :param balance_channel: boolean, whether sample same number of filters in each bin of channel
    :param min_max_scaler:
    :param data_num: total number of filters used for the data returned
    :param scaler:
    :return:
    '''
    feature_num=22
    stat=np.zeros(shape=[len(filters),feature_num],dtype=np.float)
    for i in range(len(filters)):
        stat[i][0]=np.mean(filters[i])                                  #均值
        stat[i][1]=trimmed_mean(filters[i],0.2)                         #截断均值（p=0.2）
        stat[i][2]=np.median(filters[i])                                #中位数
        stat[i][3]=filters[i].max()-filters[i].min()                    #极差
        stat[i][4]=(filters[i].max()+filters[i].min())/2                #中列数(max+min)/2
        stat[i][5]=np.percentile(filters[i],25)                         #第一四分位数
        stat[i][6]=np.percentile(filters[i],75)                         #第三四分位数
        stat[i][7]=stat[i][6]-stat[i][5]                                #四分位数极差
        stat[i][8]=np.std(filters[i])                                   #标准差
        stat[i][9]=filters[i].max()                                     #最大值
        stat[i][10]=filters[i].min()                                    #最小值
        stat[i][11]=filters[i].shape[0]                                 #通道数
        stat[i][12]=layer[i]                                            #哪一层
        stat[i][13:]=np.mean(filters[i],axis=0).flatten()               #降维后的卷积核参数

    if min_max_scaler is None:
        #归一化
        min_max_scaler=preprocessing.MinMaxScaler().fit(stat)
        stat=min_max_scaler.transform(stat)
    else:
        stat=min_max_scaler.transform(stat)

    #标准化
    if scaler is None:
        scaler=preprocessing.StandardScaler().fit(stat)
        stat=scaler.transform(stat)
    else:
        stat = scaler.transform(stat)


    if balance_channel is True:
        stat = stat[np.argsort(stat[:, 11])]

        # bin=np.bincount(stat[:,11].astype(int),minlength=20)
        bin = np.histogram(stat[:, 11])[0]

        channel_num_list = bin[bin > 0]
        # channel=np.argwhere(bin>0).flatten()

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

    return stat,min_max_scaler,scaler

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
            nn.Linear(11, 128),
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
        if name is 'svm':
            self.classifier = svm.SVC(kernel=kargs['kernel'], class_weight='balanced')
        elif name is 'logistic_regression':
            self.classifier = LogisticRegressionCV(class_weight='balanced', max_iter=1000, cv=3, Cs=[1, 10, 100, 1000])
        elif name is 'fc':
            self.classifier = fc()

    def fit(self, dead_filter, lived_filter):
        _, self.min_max_scaler =statistics(dead_filter + lived_filter)

        stat_df, _ = statistics(dead_filter, balance_channel=True, min_max_scaler=self.min_max_scaler)
        stat_lf, _ = statistics(lived_filter, balance_channel=True, min_max_scaler=self.min_max_scaler)
        train_x = np.vstack((stat_df, stat_lf))  # dead filters are in the front
        train_y = np.zeros(train_x.shape[0])
        train_y[:stat_df.shape[0]] = 1  # dead filters are labeled 1
        if self.name is 'svm':
            param_grid = {
                'C': [2 ** i for i in range(-5, 15, 2)],
                'gamma': [2 ** i for i in range(3, -15, -2)],
            }
            self.classifier = GridSearchCV(self.classifier, param_grid, scoring='f1', n_jobs=-1, cv=5)
            self.classifier = self.classifier.fit(train_x, train_y)
            print(self.classifier.best_estimator_)
        elif self.name is 'logistic_regression':
            self.classifier = self.classifier.fit(train_x, train_y)
        elif self.name is 'fc':
            self.classifier=self.classifier.fit(train_x,train_y)

    def predict_proba(self,filter):
        x,_=statistics(filter,min_max_scaler=self.min_max_scaler)
        return self.classifier.predict_proba(x)

    def predict(self,filter):
        print(self.classifier.best_estimator_)
        x, _ = statistics(filter, min_max_scaler=self.min_max_scaler)
        return self.classifier.predict(x)

if __name__ == "__main__":

    #test=predictor(name='svm',kernel='rbf')


    df,lf,df_layer,lf_layer=read_data(balance=False,path='/home/victorfang/Desktop/dead_filter(normal_distribution)/最少样本测试/训练集',neural_dead_times=1600)
    df_val,lf_val,df_layer_val,lf_layer_val=read_data(balance=False,path='/home/victorfang/Desktop/dead_filter(normal_distribution)/最少样本测试/测试集',neural_dead_times=1600)

    #df,lf=read_data(balance=False,path='/home/victorfang/Desktop/dead_filter(normal_distribution)')

    #df_val,lf_val=read_data(balance=True,path='/home/victorfang/Desktop/pytorch_model/vgg16bn_cifar10_dead_neural_normal_tar_acc_decent3/dead_neural',neural_dead_times=1200)
    _,min_max_scaler,scaler=statistics(df+lf,layer=df_layer+lf_layer,balance_channel=True)

    stat_df,_,_=statistics(df,balance_channel=True,min_max_scaler=min_max_scaler,scaler=scaler,layer=df_layer)
    stat_lf,_,_=statistics(lf,balance_channel=True,min_max_scaler=min_max_scaler,scaler=scaler,layer=lf_layer)#,data_num=stat_df.shape[0])
    stat_df_val,_,_=statistics(df_val,min_max_scaler=min_max_scaler,scaler=scaler,layer=df_layer_val)
    stat_lf_val,_,_=statistics(lf_val,min_max_scaler=min_max_scaler,scaler=scaler,layer=lf_layer_val)

    # tmp1=stat_df.tolist()
    # tmp1.insert(0,['均值','截断均值','中位数','极差','中列数','第一四分卫数','第三四分位数','四分位数极差','标准差','最大值','最小值','通道数','降维后的参数'])
    # tmp2=stat_lf.tolist()
    # tmp2.insert(0,['均值','截断均值','中位数','极差','中列数','第一四分卫数','第三四分位数','四分位数极差','标准差','最大值','最小值','通道数','降维后的参数'])
    # operate_excel.write_excel(tmp1,'./test.xlsx',0,bool_row_append=False)
    # operate_excel.write_excel(tmp2,'./test.xlsx',1,bool_row_append=False)

    train_x = np.vstack((stat_df, stat_lf))  # dead filters are in the front
    train_y = np.zeros(train_x.shape[0])
    train_y[:stat_df.shape[0]]=1

    val_x=np.vstack((stat_df_val,stat_lf_val))
    val_y=np.zeros(val_x.shape[0],dtype=np.int)
    val_y[:stat_df_val.shape[0]]=1


    ####svm ##########################################################################################################
    svc=svm.SVC(kernel='rbf',class_weight='balanced')
    param_grid = {
        'C': [2 ** i for i in range(-5, 15, 2)],
        'gamma': [2 ** i for i in range(3, -15, -2)],
    }
    clf = GridSearchCV(svc, param_grid, scoring='f1', n_jobs=-1, cv=5)

    clf.fit(train_x,train_y)
    val_x1, _ ,_= statistics(df_val+lf_val, min_max_scaler=min_max_scaler,scaler=scaler,layer=df_layer_val+lf_layer_val)
    prediction=clf.predict(val_x)

    f_score,precision,recall=cal_F_score(prediction,val_y)
    print('svm:f_score:{},precision:{},recall:{}'.format(f_score,precision,recall))

    print(clf.best_estimator_)
    print(classification_report(val_y,prediction))


    pr=predictor(name='svm',min_max_scaler=min_max_scaler,kernel='rbf')
    pr.fit(dead_filter=df,lived_filter=lf)
    prediction=pr.predict(df_val+lf_val)

    print(classification_report(val_y, prediction))

    f_score, precision, recall = cal_F_score(prediction, val_y)
    print('predictor:f_score:{},precision:{},recall:{}'.format(f_score, precision, recall))


    ##logistic regression######################################################################################################
    logistic_regression=LogisticRegressionCV(class_weight='balanced',max_iter=1000,cv=3,Cs=[1, 10,100,1000])
    re=logistic_regression.fit(train_x,train_y)
    prediction=re.predict_proba(val_x)
    prediction=np.argmax(prediction,1)
    #acc=np.sum(np.argmax(prediction,1)==val_y)/val_y.shape[0]
    f_score,precision,recall=cal_F_score(prediction,val_y)
    res=classification_report(y_true=val_y,y_pred=prediction,target_names=['lived filter','dead filter'])
    print(res)
    print('logistic regression:f_score:{},precision:{},recall:{}'.format(f_score,precision,recall))


    ###neural network##########################################################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LR=0.01
    net = fc().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)

    #optimizer=optim.Adam(net.parameters())
    #loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([1,0.1601118217]))
    loss_func = nn.CrossEntropyLoss()

    sample_num=int(stat_df.shape[0]/2)                                                                      #random choose same number of samples from lf and df

    #all training data
    x_tensor = torch.tensor(train_x, dtype=torch.float32).to(device)


    #validation data
    val_x_tensor=torch.tensor(val_x,dtype=torch.float32).to(device)
    val_y_tensor=torch.tensor(val_y,dtype=torch.long).to(device)


    #prepare the label for training
    train_y=np.zeros(2*sample_num,dtype=np.int)
    train_y[:sample_num]=1                                                                                  #dead filter's label is 1
    y_tensor = torch.tensor(train_y, dtype=torch.long).to(device)

    epoch=-1
    highest_accuracy = 0


    #load previous trained model(optional)
    # checkpoint=torch.load('/home/victorfang/Desktop/预测死亡神经元的神经网络/accuracy=0.72553.tar')
    # net.load_state_dict(checkpoint['state_dict'])
    while True:
        epoch+=1
        ind_df=random.sample([i for i in range(stat_df.shape[0])],sample_num)
        ind_lf=random.sample([i for i in range(stat_df.shape[0],train_x.shape[0])],sample_num)
        ind=torch.tensor(ind_df+ind_lf,dtype=torch.long).to(device)

        net.train()


        optimizer.zero_grad()
        output = net(torch.index_select(input=x_tensor,dim=0,index=ind))
        loss = loss_func(output,y_tensor)
        loss.backward()
        optimizer.step()
        if epoch%1000==0:
            print("{} epoch:{},   loss is:{}".format(datetime.now(),epoch,loss))

        if epoch%10000==0 :
            net.eval()
            output=net(val_x_tensor)
            prediction=torch.argmax(output,1)
            correct=(prediction==val_y_tensor).sum().float()
            acc=correct.cpu().detach().data.numpy()/val_y_tensor.shape[0]
            print( '{} accuracy is {}'.format(datetime.now(),acc))
            f_score, precision, recall = cal_F_score(prediction.cpu().data.numpy(), val_y)
            print('fc:f_score:{},precision:{},recall:{}'.format(f_score, precision, recall))
            if acc>highest_accuracy:
                highest_accuracy=acc
                if highest_accuracy>0.7:
                    print("{} Saving net...".format(datetime.now()))
                    checkpoint = {'net': net,
                                  'highest_accuracy': acc,
                                  'state_dict': net.state_dict(),
                                  }
                    torch.save(checkpoint,
                               '/home/victorfang/Desktop/预测死亡神经元的神经网络/accuracy=%.5f.tar' % (acc))
                    print("{} net saved ".format(datetime.now()))


            





