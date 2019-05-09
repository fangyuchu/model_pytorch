import config as conf
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import resnet
import vgg
import os
from datetime import datetime
import re
import math
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA           #加载PCA算法包

# def validate(val_loader, model, criterion):
def validate(val_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        batch_time = AverageMeter()
        #losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input = input.to(device)

            # compute output
            output = model(input)
            #loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            #losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            #         i, len(val_loader), batch_time=batch_time, loss=losses,
            #         top1=top1, top5=top5))

        print(' {} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
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
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def evaluate_net(  net,
                   data_loader,
                   save_net,
                   checkpoint_path=None,
                   highest_accuracy_path=None,
                   global_step_path=None,
                   global_step=0,
                   ):
    '''
    :param net: net of NN
    :param data_loader: data loader of test set
    :param save_net: Boolean. Whether or not to save the net.
    :param checkpoint_path: 
    :param highest_accuracy_path: 
    :param global_step_path: 
    :param global_step: global step of the current trained net
    '''
    if save_net:
        if checkpoint_path is None :
            raise AttributeError('please input checkpoint path')
        if highest_accuracy_path is None :
            raise AttributeError('please input highest_accuracy path')
        if global_step_path is None :
            raise AttributeError('please input global_step path')
        if os.path.exists(highest_accuracy_path):
            f = open(highest_accuracy_path, 'r')
            highest_accuracy = float(f.read())
            f.close()
        else:
            highest_accuracy=0


    print("{} Start Evaluation".format(datetime.now()))
    print("{} global step = {}".format(datetime.now(), global_step))
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for val_data in data_loader:
    #         net.eval()
    #         images, labels = val_data
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = net(images)
    #         # 取得分最高的那个类 (outputs.data的索引号)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum()
    #     correct = float(correct.cpu().numpy().tolist())
    #     accuracy = correct / total
    #     print("{} Accuracy = {:.4f}".format(datetime.now(), accuracy))
    accuracy,_=validate(data_loader,net,)
    if save_net and accuracy > highest_accuracy:
        highest_accuracy = accuracy
        # save net
        print("{} Saving net...".format(datetime.now()))
        torch.save(net.state_dict(), '%s/global_step=%d.pth' % (checkpoint_path, global_step))
        print("{} net saved ".format(datetime.now()))
        # save highest accuracy
        f = open(highest_accuracy_path, 'w')
        f.write(str(highest_accuracy))
        f.close()
        # save global step
        f = open(global_step_path, 'w')
        f.write(str(global_step))
        print("{} net saved at global step = {}".format(datetime.now(), global_step))
        f.close()

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