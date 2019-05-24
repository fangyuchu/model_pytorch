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
        torch.save(checkpoint,'%s/sample_num=%d,accuracy=%.3f.tar' % (checkpoint_path, sample_num,accuracy))
        print("{} net saved at sample num = {}".format(datetime.now(), sample_num))

    return accuracy


def check_ReLU_alive(net, data_loader,neural_dead_times):
    handle = list()
    global relu_list
    global neural_list
    relu_list=list()
    neural_list=dict()

    #register a hook for ReLU
    for mod in net.modules():
        if isinstance(mod, torch.nn.ReLU):
            handle.append(mod.register_forward_hook(check_if_dead))

    evaluate_net(net, data_loader, False)
    check_dead_rate(neural_dead_times)

    #close the hook
    for h in handle:
        h.remove()

    neural_list_temp=neural_list
    relu_list_temp=relu_list
    del relu_list,neural_list
    return relu_list_temp,neural_list_temp


def check_if_dead(module, input, output):
    if module not in relu_list:
        relu_list.append(module)
        neural_list[module]=np.zeros(output.shape[1:],dtype=np.int)

    zero_matrix=np.zeros(output.shape,dtype=np.int)
    zero_matrix[output.cpu().numpy()==0]=1
    zero_matrix=np.sum(zero_matrix,axis=0)

    neural_list[module]=neural_list[module]+zero_matrix

def check_dead_rate(neural_dead_times):
    dead_num=0
    neural_num=0
    for (k,v) in neural_list.items():
        dead_num+=np.sum(v>=neural_dead_times)                                   #neural unactivated for more than 40000 times
        neural_num+=v.size
    print("{} {:.3f}% of nodes are dead".format(datetime.now(),100*float(dead_num)/neural_num))


if __name__ == "__main__":

    net = vgg.vgg16_bn(pretrained=True)
    net.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Linear(512, 10),
    )
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    net = net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    checkpoint = torch.load('/home/victorfang/Desktop/sample_num=12460032.tar')
    #checkpoint = torch.load('/home/victorfang/Desktop/pytorch_model/vgg16_bn_cifar10_dead_neural_pruned_3/checkpoint/sample_num=2200000,accuracy=0.911.tar')

    net=checkpoint['net']
    net.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['highest_accuracy'])

    # net = test.vgg19_bn()
    # net.features = torch.nn.DataParallel(net.features)
    # #net.cpu()
    # net.cuda()
    # checkpoint=torch.load('/home/victorfang/Desktop/vgg19_bn.zip')
    # net.load_state_dict(checkpoint['state_dict'])
    # net.features=net.features.module
    # measure_flops.measure_model(net,'cifar10')



    prune_and_train.prune_dead_neural(net=net,net_name='vgg16_bn_cifar10_dead_neural_pruned',
                                      neural_dead_times=8000,
                                      dataset_name='cifar10',
                                      filter_dead_ratio=0.8,
                                      target_accuracy=0.91,
                                      optimizer=optim.SGD,
                                      learning_rate=0.01,
                                      checkpoint_step=800,
                                      batch_size=1024,
                                      filter_preserve_ratio=0.3)


    '''
        prune_and_train.prune_dead_neural(net=net,net_name='new_vgg16_bn_cifar10_dead_neural_pruned',
                                      neural_dead_times=7000,
                                      dataset_name='cifar10',
                                      filter_dead_ratio=0.95,
                                      target_accuracy=0.905,
                                      optimizer=optim.SGD,
                                      learning_rate=0.01,
                                      checkpoint_step=800,
                                      batch_size=1024)
        flops=210182114,Acc@1 0.4446 Acc@5 0.8751000005722046,Epoch number: 6,Acc@1 0.9050999990463257 Acc@5 0.9913000002861023
        
        
        prune_and_train.prune_dead_neural(net=net,net_name='new_vgg16_bn_cifar10_dead_neural_pruned',
                                  neural_dead_times=6500,
                                  dataset_name='cifar10',
                                  filter_dead_ratio=0.95,
                                  target_accuracy=0.905,
                                  optimizer=optim.SGD,
                                  learning_rate=0.01,
                                  checkpoint_step=800,
                                  batch_size=1024)
        flops=183511802,51.797% of nodes are dead,Acc@1 0.18589999964237214 Acc@5 0.5201000001430511
        Epoch number: 34,Acc@1 0.9056999988555908 Acc@5 0.9897000001907349
        
    '''








    #check_ReLU_alive(net,val_loader,8000)









    # for mod in relu_list:
    #     if module==mod.module:
    #         mod.update(dead_num)
    #
    # dead_rate.update(dead_num/output.numel(),output.numel())

