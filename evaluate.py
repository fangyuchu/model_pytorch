import torch
import torch.nn as nn
import time
import os
from datetime import datetime
import numpy as np
import vgg
import data_loader
import config as conf
import prune_and_train
import torch.optim as optim
import train
import measure_flops
import logger
import math
import generate_random_data
import matplotlib.pyplot as plt
import predict_dead_filter

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def validate(val_loader, model, criterion):
def validate(val_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        batch_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            #print('{} {}'.format(datetime.now(),i))
            target = target.to(device)
            input = input.to(device)

            # compute output
            output = model(input)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            #losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('{} Acc@1 {top1.avg:} Acc@5 {top5.avg:}'
              .format(datetime.now(),top1=top1, top5=top5))

        return top1.avg, top5.avg


def same_two_nets(net1,net2,print_diffrent_module=False):
    '''
    Compare net1 and net2. If their structure and weights are identical, return True.
    :param net1:
    :param net2:
    :param print_diffrent_module:
    :return:
    '''
    param1 = net1.state_dict()
    param2=net2.state_dict()
    for key in list(param1.keys()):
        o=param1[key].data.cpu().numpy()
        n=param2[key].data.cpu().numpy()
        if not (o==n).all():
            if print_diffrent_module :
                print(key)
            return False
    return True

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))                  #each item is one k_accuracy
    return res


def evaluate_net(  net,
                   data_loader,
                   save_net,
                   checkpoint_path=None,
                   sample_num=0,
                   target_accuracy=1
                   ):
    '''
    :param net: net of NN
    :param data_loader: data loader of test set
    :param save_net: Boolean. Whether or not to save the net.
    :param checkpoint_path:
    :param highest_accuracy_path:
    :param sample_num_path:
    :param sample_num: sample num of the current trained net
    :param target_accuracy: save the net if its accuracy surpasses the target_accuracy
    '''
    if save_net:
        if checkpoint_path is None :
            raise AttributeError('please input checkpoint path')

        lists=os.listdir(checkpoint_path)
        file_new = checkpoint_path
        if len(lists) > 0:
            lists.sort(key=lambda fn: os.path.getmtime(checkpoint_path + "/" + fn))  # 按时间排序
            file_new = os.path.join(checkpoint_path, lists[-1])  # 获取最新的文件保存到file_new


        if os.path.isfile(file_new):
            checkpoint=torch.load(file_new)
            highest_accuracy = checkpoint['highest_accuracy']
        else:
            highest_accuracy=0


    print("{} Start Evaluation".format(datetime.now()))
    print("{} sample num = {}".format(datetime.now(), sample_num))

    accuracy,_=validate(data_loader,net,)

    if save_net and (accuracy > highest_accuracy or accuracy>target_accuracy):
        # save net
        print("{} Saving net...".format(datetime.now()))
        checkpoint={'net':net,
                    'highest_accuracy':accuracy,
                    'state_dict':net.state_dict(),
                    'sample_num':sample_num}
        torch.save(checkpoint,'%s/sample_num=%d,accuracy=%.5f.tar' % (checkpoint_path, sample_num,accuracy))
        print("{} net saved at sample num = {}".format(datetime.now(), sample_num))

    return accuracy

def predict_dead_filters_classifier_version(net,
                                         predictor,
                                         min_ratio_dead_filters=0,
                                         max_ratio_dead_filters=1,
                                         filter_num_lower_bound=None
                                         ):
    '''
    use trained predictor to predict dead filters in net
    :param net:
    :param predictor:
    :param min_ratio_dead_filters: float, ensure the function will return at least (min_ratio_dead_filters)*100% of the filters.

    :param max_ratio_dead_filters: float, ensure the function will return at most (max_ratio_dead_filters)*100% of the filters.
                                          ensure the number of filters pruned will not be too large for one time.
    :param filter_num_lower_bound: int, ensure the number of filters alive will be larger than filter_num_lower_bound
                                          ensure the lower bound of filter number

    :return:
    '''

    dead_filter_index_data,_,_=find_dead_filters_data_version(net=net,filter_dead_ratio=0.9,batch_size=1200,neural_dead_times=1200)

    dead_filter_index=list()
    df_num_max=list()                           #max number of filter num
    df_num_min=list()                           #min number of filter num
    filter_num=list()                           #filter num in each layer
    weight=list()
    num_conv=0
    for mod in net.features:
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            weight+=list(mod.weight.data.cpu().numpy())
            filter_num.append(mod.weight.data.cpu().numpy().shape[0])
            df_num_min.append(math.ceil(min_ratio_dead_filters*filter_num[num_conv]))                                            #lower bound of dead_filter's num
            if filter_num_lower_bound is not None:                                                              #upper bound of dead_filter's num
                df_num_max.append(min(int(max_ratio_dead_filters*len(weight)),filter_num[num_conv]-filter_num_lower_bound[num_conv]))
            else:
                df_num_max.append(int(max_ratio_dead_filters*len(weight)))

            if df_num_max[num_conv]<df_num_min[num_conv]:
                print('Number of filters in layer{} is {}. At most {} filters will be predicted to be dead.'.format(num_conv,filter_num[num_conv],df_num_max[num_conv]))
            num_conv+=1

#todo:全部的卷积核一起标准化更合理
    stat_filters,_=predict_dead_filter.statistics(weight,min_max_scaler=predictor.min_max_scaler)

    s=0
    for i in range(num_conv):
        output = predictor.predict_proba(stat_filters[s:s+filter_num[i]])
        dead_filter_proba_sorted=np.argsort(-output[:,1])                   #filter indices sorted by the probability of which to be dead
        dead_filter_predicted=np.where(np.argmax(output,1))[0]             #filter indices of which are predicted to be dead
        if dead_filter_predicted.shape[0]<df_num_min[i]:
            print(i,'死亡卷积核太少')
            dead_filter_predicted=dead_filter_proba_sorted[:df_num_min[i]]
        if dead_filter_predicted.shape[0]>df_num_max[i]:
            print(i,'死亡卷积核太多')
            dead_filter_predicted=dead_filter_proba_sorted[:df_num_max[i]]
        dead_filter_index.append(dead_filter_predicted)
        s += filter_num[i]
    return dead_filter_index


def find_dead_filters_data_version(net,
                                   filter_dead_ratio,
                                   batch_size,
                                   neural_dead_times,
                                   dataset_name='cifar10',
                                   use_random_data=True,
                                   relu_list=None,
                                   neural_list=None,
                      ):
    '''
    use validation set or random generated data to find dead filters in net
    :param net:
    :param filter_dead_ratio:
    :param dataset_name:
    :param use_random_data:
    :param relu_list:
    :param neural_list:
    :param batch_size:
    :param neural_dead_times:
    :return:
    '''
    if relu_list is None or neural_list is None:
        #calculate dead neural
        if use_random_data is True:
            random_data=generate_random_data.random_normal(num=batch_size,dataset_name=dataset_name)
            print('{} generate random data.'.format(datetime.now()))
            relu_list, neural_list = check_ReLU_alive(net=net, neural_dead_times=neural_dead_times, data=random_data)
            del random_data
        else:
            if dataset_name is 'imagenet':
                mean = conf.imagenet['mean']
                std = conf.imagenet['std']
                validation_set_path = conf.imagenet['validation_set_path']
                default_image_size = conf.imagenet['default_image_size']
            elif dataset_name is 'cifar10':
                mean = conf.cifar10['mean']
                std = conf.cifar10['std']
                validation_set_path = conf.cifar10['validation_set_path']
                default_image_size = conf.cifar10['default_image_size']
            validation_loader = data_loader.create_validation_loader(dataset_path=validation_set_path,
                                                                     default_image_size=default_image_size,
                                                                     mean=mean,
                                                                     std=std,
                                                                     batch_size=batch_size,
                                                                     num_workers=2,
                                                                     dataset_name=dataset_name,
                                                                     )
            relu_list,neural_list=check_ReLU_alive(net=net,data_loader=validation_loader,neural_dead_times=neural_dead_times)

    num_conv = 0  # num of conv layers in the net
    filter_num = list()
    for mod in net.features:
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv += 1
            filter_num.append(mod.out_channels)
    dead_filter_index=list()
    for i in range(num_conv):
        for relu_key in list(neural_list.keys()):
            if relu_list[i] is relu_key:  # find the neural_list_statistics in layer i+1
                dead_relu_list = neural_list[relu_key]
                neural_num = dead_relu_list.shape[1] * dead_relu_list.shape[2]  # neural num for one filter

                # judge dead filter by neural_dead_times and dead_filter_ratio
                dead_relu_list[dead_relu_list < neural_dead_times] = 0
                dead_relu_list[dead_relu_list >= neural_dead_times] = 1
                dead_relu_list = np.sum(dead_relu_list, axis=(1, 2))  # count the number of dead neural for one filter

                df_num=np.where(dead_relu_list >= neural_num * filter_dead_ratio)[0].shape[0]                    #number of dead filters
                df_index=np.argsort(-dead_relu_list)[:df_num].tolist()                                           #dead filters' indices. sorted by the times that they died.
                dead_filter_index.append(df_index)

    return dead_filter_index,relu_list,neural_list

def check_ReLU_alive(net,neural_dead_times,data=None,data_loader=None):
    handle = list()
    global relu_list                                                        #list containing relu module
    global neural_list
    relu_list=list()
    neural_list=dict()

    #register a hook for ReLU
    for mod in net.modules():
        if isinstance(mod, torch.nn.ReLU):
            handle.append(mod.register_forward_hook(compute_dead_times))

    if data_loader is not None:
        evaluate_net(net, data_loader, False)
        cal_dead_neural_rate(neural_dead_times)
    elif data is not None:
        net.eval()
        net(data)
        cal_dead_neural_rate(neural_dead_times)
    else:
        raise BaseException("Please provide input data.")

    #close the hook
    for h in handle:
        h.remove()

    neural_list_temp=neural_list
    relu_list_temp=relu_list
    del relu_list,neural_list
    return relu_list_temp,neural_list_temp



def plot_dead_filter_statistics(net,relu_list,neural_list,neural_dead_times,filter_dead_ratio):
    dead_filter_num=list()                                                                      #num of dead filters in each layer
    filter_num=list()                                                                           #num of filters in each layer

    num_conv = 0  # num of conv layers in the net
    for mod in net.features:
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv += 1

    for i in range(num_conv):
        for relu_key in list(neural_list.keys()):
            if relu_list[i] is relu_key:  # find the neural_list_statistics in layer i+1
                dead_relu_list = neural_list[relu_key]
                neural_num = dead_relu_list.shape[1] * dead_relu_list.shape[2]  # neural num for one filter

                # judge dead filter by neural_dead_times and dead_filter_ratio
                dead_relu_list[dead_relu_list < neural_dead_times] = 0
                dead_relu_list[dead_relu_list >= neural_dead_times] = 1
                dead_relu_list = np.sum(dead_relu_list, axis=(1, 2))  # count the number of dead neural for one filter
                dead_filter_index = np.where(dead_relu_list >= neural_num * filter_dead_ratio)[0].tolist()
                dead_filter_num.append(len(dead_filter_index))
                filter_num.append(len(neural_list[relu_key]))
    plt.figure()
    plt.title('statistics of dead filter\nneural_dead_time={},filter_dead_ratio={}'.format(neural_dead_times,filter_dead_ratio))
    plt.bar(range(len(filter_num)),filter_num,label='filter')
    plt.bar(range(len(dead_filter_num)),dead_filter_num,label='dead filter')
    plt.xlabel('layer')
    plt.ylabel('number of filters')
    plt.legend()
    plt.show()
    print()



def compute_dead_times(module, input, output):
    if module not in relu_list:
        relu_list.append(module)
        neural_list[module]=np.zeros(output.shape[1:],dtype=np.int)
    output=output.detach()                                              #set requires_grad to False
    zero_matrix=np.zeros(output.shape,dtype=np.int)
    zero_matrix[output.cpu().numpy()==0]=1
    zero_matrix=np.sum(zero_matrix,axis=0)

    neural_list[module]=neural_list[module]+zero_matrix

def cal_dead_neural_rate(neural_dead_times,neural_list_temp=None):
    dead_num=0
    neural_num=0
    if neural_list_temp is None:
        neural_list_temp=neural_list
    for (k,v) in neural_list_temp.items():
        dead_num+=np.sum(v>=neural_dead_times)                                   #neural unactivated for more than 40000 times
        neural_num+=v.size
    dead_neural_rate=100*float(dead_num)/neural_num
    print("{} {:.3f}% of nodes are dead".format(datetime.now(),dead_neural_rate))
    return dead_neural_rate


if __name__ == "__main__":

    # net = vgg.vgg16_bn(pretrained=True)
    # net.classifier = nn.Sequential(
    #     nn.Dropout(),
    #     nn.Linear(512, 512),
    #     nn.ReLU(True),
    #     nn.Dropout(),
    #     nn.Linear(512, 512),
    #     nn.ReLU(True),
    #     nn.Linear(512, 10),
    # )
    # for m in net.modules():
    #     if isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight, 0, 0.01)
    #         nn.init.constant_(m.bias, 0)
    # net = net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    checkpoint = torch.load('./baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
    #checkpoint = torch.load('./vgg16_bn,baseline.tar')

    net=checkpoint['net']
    net.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['highest_accuracy'])


    #measure_flops.measure_model(net,dataset_name='cifar10')

    prune_and_train.prune_dead_neural(net=net,
                                      net_name='tmp',
                                      dataset_name='cifar10',
                                      neural_dead_times=9000,
                                      filter_dead_ratio=0.9,
                                      neural_dead_times_decay=0.95,
                                      filter_dead_ratio_decay=0.98,
                                      filter_preserve_ratio=0.1,
                                      max_filters_pruned_for_one_time=0.3,
                                      target_accuracy=0.933,
                                      tar_acc_gradual_decent=True,
                                      flop_expected=5e7,
                                      batch_size=1600,
                                      num_epoch=300,
                                      checkpoint_step=1600,

                                      # optimizer=optim.Adam,
                                      # learning_rate=1e-3,
                                      # weight_decay=0
                                      optimizer=optim.SGD,
                                      learning_rate=0.01,
                                      learning_rate_decay=True,
                                      learning_rate_decay_epoch=[50,100,150,250,300,350,400],
                                      learning_rate_decay_factor=0.5,
                                      )


    # prune_and_train.prune_dead_neural_with_predictor(net=net,
    #                                   net_name='tmp',
    #                                      # predictor_name='logistic_regression',
    #                                    predictor_name='svm',
    #                                                  kernel='rbf',
    #                                    round_for_train=2,
    #                                   dataset_name='cifar10',
    #                                   use_random_data=True,
    #                                   neural_dead_times=1600,
    #                                   filter_dead_ratio=0.9,
    #                                   neural_dead_times_decay=0.99,
    #                                   filter_dead_ratio_decay=0.98,
    #                                   filter_preserve_ratio=0.01,
    #                                   max_filters_pruned_for_one_time=0.2,
    #                                   target_accuracy=0.933,
    #                                   batch_size=1600,
    #                                   num_epoch=450,
    #                                   checkpoint_step=1600,
    #
    #                                   tar_acc_gradual_decent=True,
    #                                   flop_expected=5e7,
    #
    #                                   # optimizer=optim.Adam,
    #                                   # learning_rate=1e-3,
    #                                   # weight_decay=0
    #                                   optimizer=optim.SGD,
    #                                   learning_rate=0.01,
    #                                   learning_rate_decay=True,
    #                                   learning_rate_decay_epoch=[50,100,150,250,300,350,400],
    #                                   learning_rate_decay_factor=0.5,
    #
    #                                   )







