import torch
import time
import os
from datetime import datetime
import numpy as np



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

        print(' {} Acc@1 {top1.avg:} Acc@5 {top5.avg:}'
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
                   highest_accuracy_path=None,
                   sample_num_path=None,
                   sample_num=0,
                   ):
    '''
    :param net: net of NN
    :param data_loader: data loader of test set
    :param save_net: Boolean. Whether or not to save the net.
    :param checkpoint_path: 
    :param highest_accuracy_path: 
    :param sample_num_path:
    :param sample_num: sample num of the current trained net
    '''
    if save_net:
        if checkpoint_path is None :
            raise AttributeError('please input checkpoint path')
        if highest_accuracy_path is None :
            raise AttributeError('please input highest_accuracy path')
        if sample_num_path is None :
            raise AttributeError('please input sample_num path')
        if os.path.exists(highest_accuracy_path):
            f = open(highest_accuracy_path, 'r')
            highest_accuracy = float(f.read())
            f.close()
        else:
            highest_accuracy=0


    print("{} Start Evaluation".format(datetime.now()))
    print("{} sample num = {}".format(datetime.now(), sample_num))
    accuracy,_=validate(data_loader,net,)
    if save_net and accuracy > highest_accuracy:
        highest_accuracy = accuracy
        # save net
        print("{} Saving net...".format(datetime.now()))
        torch.save(net.state_dict(), '%s/sample_num=%d.pth' % (checkpoint_path, sample_num))
        print("{} net saved ".format(datetime.now()))
        # save highest accuracy
        f = open(highest_accuracy_path, 'w')
        f.write(str(highest_accuracy))
        f.close()
        # save sample num
        f = open(sample_num_path, 'w')
        f.write(str(sample_num))
        print("{} net saved at sample num = {}".format(datetime.now(), sample_num))
        f.close()
    return accuracy



def check_ReLU_alive(net, data_loader):
    handle = list()
    global relu_list
    global neural_list
    relu_list=set()
    neural_list=dict()

    #register a hook for ReLU
    for mod in net.modules():
        if isinstance(mod, torch.nn.ReLU):
            handle.append(mod.register_forward_hook(check_if_dead))

    evaluate_net(net, data_loader, False)
    check_dead_rate()

    #close the hook
    for h in handle:
        h.remove()

    del relu_list,neural_list


def check_if_dead(module, input, output):
    if module not in relu_list:
        relu_list.add(module)
        neural_list[module]=np.zeros(output.shape[1:],dtype=np.int)

    # zero_matrix=output.cpu().numpy()

    zero_matrix=np.zeros(output.shape,dtype=np.int)
    zero_matrix[output.cpu().numpy()==0]=1
    zero_matrix=np.sum(zero_matrix,axis=0)

    neural_list[module]=neural_list[module]+zero_matrix

def check_dead_rate():
    dead_num=0
    neural_num=0
    for (k,v) in neural_list.items():
        dead_num+=np.sum(v>40000)                                   #neural unactivated for more than 40000 times
        neural_num+=v.size
    print("{} {:.3f}% of nodes are dead".format(datetime.now(),100*float(dead_num)/neural_num))



    # for mod in relu_list:
    #     if module==mod.module:
    #         mod.update(dead_num)
    #
    # dead_rate.update(dead_num/output.numel(),output.numel())

