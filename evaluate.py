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
from PIL import Image
import matplotlib.pyplot as plt

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


def check_ReLU_alive(net,neural_dead_times,data=None,data_loader=None):
    handle = list()
    global relu_list                                                        #list containing relu module
    global neural_list
    relu_list=list()
    neural_list=dict()

    #register a hook for ReLU
    for mod in net.modules():
        if isinstance(mod, torch.nn.ReLU):
            handle.append(mod.register_forward_hook(check_if_dead))

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



def check_if_dead(module, input, output):
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
    print("{} {:.3f}% of nodes are dead".format(datetime.now(),100*float(dead_num)/neural_num))


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


    checkpoint = torch.load('/home/victorfang/Desktop/vgg16_bn_cifar10,accuracy=0.941.tar')
    #checkpoint = torch.load('/home/victorfang/Desktop/pytorch_model/vgg16bn_cifar10_dead_neural_normal_tar_acc_decent1/checkpoint/sample_num=11050000,accuracy=0.93370.tar')

    net=checkpoint['net']
    net.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['highest_accuracy'])


    #measure_flops.measure_model(net,dataset_name='cifar10')

    # prune_and_train.prune_dead_neural(net=net,
    #                                   net_name='vgg16_bn_cifar10_dead_neural_pruned100',
    #                                   dataset_name='cifar10',
    #                                   neural_dead_times=8000,
    #                                   filter_dead_ratio=0.9,
    #                                   neural_dead_times_decay=0.95,
    #                                   filter_dead_ratio_decay=0.98,
    #                                   filter_preserve_ratio=0.1,
    #                                   max_filters_pruned_for_one_time=0.3,
    #                                   target_accuracy=0.931,
    #                                   batch_size=300,
    #                                   num_epoch=300,
    #                                   checkpoint_step=1600,
    #
    #                                   # optimizer=optim.Adam,
    #                                   # learning_rate=1e-3,
    #                                   # weight_decay=0
    #                                   optimizer=optim.SGD,
    #                                   learning_rate=0.01,
    #                                   learning_rate_decay=True,
    #                                   learning_rate_decay_epoch=[50,100,150,250,300,350,400],
    #                                   learning_rate_decay_factor=0.5,
    #                                   )


    prune_and_train.prune_dead_neural(net=net,
                                      net_name='vgg16bn_cifar10_dead_neural_normal_tar_acc_decent2',
                                      dataset_name='cifar10',
                                      use_random_data=True,
                                      neural_dead_times=1200,
                                      filter_dead_ratio=0.9,
                                      neural_dead_times_decay=0.99,
                                      filter_dead_ratio_decay=0.98,
                                      filter_preserve_ratio=0.1,
                                      max_filters_pruned_for_one_time=0.2,
                                      target_accuracy=0.9325,
                                      batch_size=1200,
                                      num_epoch=450,
                                      checkpoint_step=1600,

                                      tar_acc_gradual_decent=True,
                                      flop_expected=6e7,

                                      # optimizer=optim.Adam,
                                      # learning_rate=1e-3,
                                      # weight_decay=0
                                      optimizer=optim.SGD,
                                      learning_rate=0.01,
                                      learning_rate_decay=True,
                                      learning_rate_decay_epoch=[50,100,150,250,300,350,400],
                                      learning_rate_decay_factor=0.5,
                                      )







