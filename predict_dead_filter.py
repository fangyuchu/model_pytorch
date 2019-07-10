import torch
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV
import os
import torch.nn as nn
from datetime import datetime
import random
from torch import optim







def read_data(path='/home/victorfang/Desktop/dead_filter(normal_distribution)',
              balance=False,
              neural_dead_times=999,
              filter_dead_ratio=0.85):
    dead_filter =list()
    living_filter = list()
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
                        for ind in living_filter_index:
                            living_filter.append(filter_weight[ind])

    if balance is True:
        living_filter=living_filter[:len(dead_filter)]
    return dead_filter,living_filter

def trimmed_mean(filter,p):
    filter=filter.flatten()
    min_ind=int(filter.shape[0]*p)
    max_ind=int(filter.shape[0]*(1-p))
    filter=filter[min_ind:max_ind]
    return np.mean(filter)

def statistics(filters):
    '''
    均值，截断均值（20%），中位数，极差，中列数，四分位数（第一第三个四分位数），四分位数极差，标准差，最大值，最小值
    :param filters:
    :return:
    '''
    stat=np.zeros(shape=[len(filters),11],dtype=np.float)
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

    return stat

class fc(nn.Module):

    def __init__(self, init_weights=True):
        super(fc, self).__init__()
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

if __name__ == "__main__":
    df,lf=read_data(balance=False,
                              path='/home/victorfang/Desktop/dead_filter(normal_distribution)')

    df_val,lf_val=read_data(balance=True,path='/home/victorfang/Desktop/pytorch_model/vgg16bn_cifar10_dead_neural_normal_tar_acc_decent2_2/dead_neural')

    stat_df=statistics(df)
    stat_lf=statistics(lf)
    stat_df_val=statistics(df_val)
    stat_lf_val=statistics(lf_val)

    train_x = np.vstack((stat_df, stat_lf))  # dead filters are in the front
    train_y = np.zeros(train_x.shape[0])
    train_y[:stat_df.shape[0]]=1

    val_x=np.vstack((stat_df_val,stat_lf_val))
    val_y=np.zeros(val_x.shape[0],dtype=np.int)
    val_y[:stat_df_val.shape[0]]=1


    ##logistic regression######################################################################################################
    logistc_regression=LogisticRegressionCV(class_weight='balanced',max_iter=1000)
    re=logistc_regression.fit(train_x,train_y)
    prediction=re.predict_proba(val_x)
    acc=np.sum(np.argmax(prediction,1)==val_y)/val_y.shape[0]
    print()


    ###neural network##########################################################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LR=0.1
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


            

    ####svm ##########################################################################################################
    # svc=svm.SVC(kernel='rbf')
    # svc.fit(train_x,train_y)
    # predict_y=svc.predict(val_x)


    print()



