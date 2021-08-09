import torch
import torch.nn as nn
import time
import os
from datetime import datetime
import numpy as np
from framework import data_loader, measure_flops, config as conf
# from prune import prune_and_train
import torch.optim as optim
import math
import generate_random_data
from filter_characteristic import predict_dead_filter
import copy
from network import storage
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
def validate(val_loader, model,max_data_to_test=99999999,device=None):

    with torch.no_grad():
        batch_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        model.eval()
        end = time.time()
        s_n=0
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            s_n+=input.shape[0]

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

            if s_n>=max_data_to_test:
                break
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
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))                  #each item is one k_accuracy
    return res


def evaluate_net(  net,
                   data_loader,
                   save_net,
                   net_name=None,
                   exp_name='',
                   checkpoint_path=None,
                   sample_num=0,
                   target_accuracy=1,
                   dataset_name='cifar10',
                   max_data_to_test=99999999,
                   top_acc=1,
                   device=None,
                   optimizer=None,
                   ):
    '''
    :param net: net of NN
    :param data_loader: data loader of test set
    :param save_net: Boolean. Whether or not to save the net.
    :param net_name: name of the network. eg:vgg16_bn
    :param checkpoint_path:
    :param highest_accuracy_path:
    :param sample_num_path:
    :param sample_num: sample num of the current trained net
    :param target_accuracy: save the net if its accuracy surpasses the target_accuracy
    :param max_data_to_test: use at most max_data_to_test images to evaluate the net
    :param top_acc: top 1 or top 5 accuracy
    '''
    net.eval()
    if save_net:
        flop_num = measure_flops.measure_model(net=net, dataset_name=dataset_name, print_flop=False)
        if checkpoint_path is None :
            raise AttributeError('please input checkpoint path')

        lists=os.listdir(checkpoint_path)
        file_new = checkpoint_path
        if len(lists) > 0:
            lists.sort(key=lambda fn: os.path.getmtime(checkpoint_path + "/" + fn))  # 按时间排序
            file_name = lists[-1] if 'last_model' not in lists[-1] else lists[-2]
            file_new = os.path.join(checkpoint_path, file_name)  # 获取最新的文件保存到file_new

        if os.path.isfile(file_new):
            checkpoint=torch.load(file_new)
            highest_accuracy = checkpoint['highest_accuracy']
            flop_num_old=checkpoint['flop_num']
            if flop_num!=flop_num_old:
                highest_accuracy=0
        else:
            highest_accuracy=0

    print("{} Start Evaluation".format(datetime.now()))
    print("{} sample num = {}".format(datetime.now(), sample_num))

    top1_accuracy, top5_accuracy = validate(data_loader, net, max_data_to_test, device)
    if top_acc==1:
        accuracy=top1_accuracy
    elif top_acc==5:
        accuracy=top5_accuracy

    if save_net and (accuracy > highest_accuracy or accuracy>target_accuracy):
        # save net
        print("{} Saving net...".format(datetime.now()))
        checkpoint={'highest_accuracy':accuracy,
                    'sample_num':sample_num,
                    'flop_num':flop_num,
                    'exp_name':exp_name}

        if optimizer is not None:
            checkpoint['optimizer']=optimizer
            checkpoint['optimizer_state_dict']=optimizer.state_dict()

        checkpoint.update(storage.get_net_information(net,dataset_name,net_name))
        try:
            torch.save(checkpoint,'%s/flop=%d,accuracy=%.5f.pth' % (checkpoint_path, flop_num,accuracy))
        except AttributeError:
            checkpoint.pop('net')  # AttributeError: Can't pickle local object 'new_forward.<locals>.lambda_forward'
            torch.save(checkpoint, '%s/flop=%d,accuracy=%.5f.pth' % (checkpoint_path, flop_num, accuracy))
        print("{} net saved at sample num = {}".format(datetime.now(), sample_num))

    return accuracy

#

def find_useless_filters_regressor_version(net,
                                           predictor,
                                           percent_of_inactive_filter,
                                           max_filters_pruned_for_one_time
                                           ):
    num_conv = 0
    for mod in net.modules():
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv += 1
    max_num_filters_pruned_layerwise=[0 for i in range(num_conv)]                                             #upperbound of number of filters to prune
    filter_layer=[]                                                                 #list containing layer of the filters corresponding to filter_index
    filter = []
    filter_index=[]                                                                 #index of filters in their own conv
    num_conv = 0
    for mod in net.modules():
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            conv_weight = mod.weight.cpu().detach().numpy()
            num_conv+=1
        if isinstance(mod,torch.nn.ReLU):                                               #ensure the conv are followed by relu
            if filter_layer !=[] and filter_layer[-1]==num_conv-1:                                            #get rid of the influence from relu in fc
                continue
            for weight in conv_weight:
                filter.append(weight)
            filter_layer += [num_conv-1 for j in range(conv_weight.shape[0])]
            filter_index += [j for j in range(conv_weight.shape[0])]
            if type(max_filters_pruned_for_one_time) is list:
                max_num_filters_pruned_layerwise[num_conv-1]=int(conv_weight.shape[0]*max_filters_pruned_for_one_time[num_conv-1])
            else:
                max_num_filters_pruned_layerwise[num_conv-1]=int(conv_weight.shape[0]*max_filters_pruned_for_one_time)

    FIRE=predictor.predict(filter=filter,filter_layer=filter_layer)

    inactive_rank = np.argsort(-np.array(FIRE))                         # arg for most inactive filters
    inactive_filter_index = np.array(filter_index)[inactive_rank]
    inactive_filter_layer = np.array(filter_layer)[inactive_rank]
    useless_filter_index=[[] for i in range(num_conv)]
    num_selected_filters=0
    num_filters_to_select=percent_of_inactive_filter*len(FIRE)

    for i in range(inactive_filter_index.shape[0]) :
        layer=inactive_filter_layer[i]
        if len(useless_filter_index[layer])>=max_num_filters_pruned_layerwise[layer]:
            continue                                                         #eunsure that num of filters pruned for each layer will not be too large
        useless_filter_index[layer].append(inactive_filter_index[i])
        num_selected_filters+=1
        if num_selected_filters>num_filters_to_select:
            break

    return useless_filter_index



def find_useless_filters_data_version(net,
                                      batch_size,
                                      percent_of_inactive_filter,
                                      dataset_name='cifar10',
                                      use_random_data=False,
                                      module_list=None,
                                      neural_list=None,
                                      dead_or_inactive='inactive',
                                      neural_dead_times=None,
                                      filter_FIRE=None,
                                      # max_data_to_test=10000,
                                      num_filters_to_prune_at_most=None,
                                      max_data_to_test=10000,
                      ):
    '''
    use test set or random generated data to find useless filters in net
    :param net:
    :param batch_size:
    :param dataset_name:
    :param use_random_data:
    :param module_list:
    :param neural_list:
    :param dead_or_inactive:
    param for dead filter
    :param neural_dead_times:
    :param filter_FIRE:
    :param percent_of_inactive_filter:
    :param max_data_to_test: use at most (max_data_to_test) images to calculate the inactive rate
    :param num_filters_to_prune_at_most: list containing the minimum number of filters in each layer
    :return:
    '''
    if dead_or_inactive == 'dead':
        if neural_dead_times is None or filter_FIRE is None:
            print('neural_dead_times and filter_FIRE are required to find dead filters.')
            raise AttributeError
    elif dead_or_inactive == 'inactive':
        if percent_of_inactive_filter is None:
            print('percent_of_inactive_filter is required to find dead filters.')
            raise AttributeError
    else:
        print('unknown type of dead_or_inactive')
        raise AttributeError
    if module_list is None or neural_list is None:
        #calculate dead neural
        if use_random_data is True:
            random_data=generate_random_data.random_normal(num=batch_size,dataset_name=dataset_name)
            num_test_images=batch_size
            print('{} generate random data.'.format(datetime.now()))
            # module_list, neural_list = check_conv_alive_layerwise(net=net,neural_dead_times=batch_size,batch_size=batch_size)
            module_list, neural_list = check_ReLU_alive(net=net, neural_dead_times=batch_size, data=random_data)
            del random_data
        else:
            if dataset_name == 'imagenet':
                train_set_size = conf.imagenet['train_set_size']
            elif dataset_name == 'cifar10':
                train_set_size=conf.cifar10['train_set_size']
            elif dataset_name == 'cifar100':
                train_set_size=conf.cifar100['train_set_size']
            elif dataset_name == 'tiny_imagenet':
                train_set_size=conf.tiny_imagenet['train_set_size']
            train_loader = data_loader.create_test_loader(
                                                                batch_size=batch_size,
                                                                num_workers=8,
                                                                dataset_name=dataset_name+'_trainset',
                                                                shuffle=True,
                                                                     )

            num_test_images = min(train_set_size, math.ceil(max_data_to_test / batch_size) * batch_size)
            if neural_dead_times is None and dead_or_inactive == 'inactive':
                neural_dead_times=0.8*num_test_images

            if isinstance(net,torch.nn.DataParallel):
                net_test=copy.deepcopy(net._modules['module'])                      #use one gpu
            else:
                net_test=copy.deepcopy(net)
            module_list,neural_list=check_ReLU_alive(net=net_test,data_loader=train_loader,
                                                     neural_dead_times=neural_dead_times,max_data_to_test=max_data_to_test)
            del net_test
            del train_loader
    num_conv = 0  # num of conv layers in the net
    filter_num = []
    for name,mod in net.named_modules():
        if isinstance(mod, torch.nn.Conv2d) and 'downsample' not in name:
            num_conv += 1
            filter_num.append(mod.out_channels)


    useless_filter_index=[[] for i in range(num_conv)]
    if dead_or_inactive == 'inactive':
        # filter_index=[]                                 #index of the filter in its layer
        # filter_layer=[]                                 #which layer the filter is in
        FIRE=[]
    for i in range(len(module_list)):#the number of relu after conv  range(len(module_list)):
        for module_key in list(neural_list.keys()):
            if module_list[i] is module_key:  # find the neural_list_statistics in layer i+1
                dead_times = copy.deepcopy(neural_list[module_key])
                neural_num = dead_times.shape[1] * dead_times.shape[2]  # neural num for one filter

                if dead_or_inactive == 'dead':
                    print('warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! may be wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    # judge dead filter by neural_dead_times and dead_filter_ratio
                    dead_times[dead_times < neural_dead_times] = 0
                    dead_times[dead_times >= neural_dead_times] = 1
                    dead_times = np.sum(dead_times, axis=(1, 2))  # count the number of dead neural for one filter

                    df_num=np.where(dead_times >= neural_num * filter_FIRE)[0].shape[0]                    #number of dead filters
                    df_index=np.argsort(-dead_times)[:df_num].tolist()                                           #dead filters' indices. sorted by the times that they died.
                    useless_filter_index.append(df_index)
                elif dead_or_inactive == 'inactive':
                    # compute sum(dead_times)/(batch_size*neural_num) as label for each filter
                    dead_times = np.sum(dead_times, axis=(1, 2))
                    # if use_random_data is True:
                    #     FIRE += (dead_times / (neural_num * batch_size)).tolist()
                    # else:
                    FIRE += (dead_times / (neural_num * num_test_images)).tolist()
                    # filter_layer += [i for j in range(dead_times.shape[0])]
                    # filter_index+=[j for j in range(dead_times.shape[0])]

    if dead_or_inactive == 'dead':
        return useless_filter_index, module_list, neural_list
    elif dead_or_inactive == 'inactive':
        useless_filter_index=sort_inactive_filters(net,percent_of_inactive_filter,FIRE,num_filters_to_prune_at_most)
        # cutoff_rank_increase=-1
        # delta=0
        # while cutoff_rank_increase!=delta:
        #     cutoff_rank_increase = delta
        #     delta = 0
        #     cutoff_rank=int(percent_of_inactive_filter*len(FIRE))+cutoff_rank_increase
        #     inactive_rank=np.argsort(-np.array(FIRE))[:cutoff_rank]                #arg for top (percent_of_inactive_filter)*100% of inactive filters
        #     inactive_filter_index=np.array(filter_index)[inactive_rank]                     #index of top (percent_of_inactive_filter)*100% inactive filters
        #     inactive_filter_layer=np.array(filter_layer)[inactive_rank]
        #     for i in range(num_conv):
        #         index = inactive_filter_index[np.where(inactive_filter_layer == i)]     #index of inactive filters in layer i
        #         if num_filters_to_prune_at_most is not None:
        #             if len(index)>num_filters_to_prune_at_most[i]:
        #                 delta+=len(index)-num_filters_to_prune_at_most[i]           #number of inactive filters excluded because of the restriction
        #                 index=index[:num_filters_to_prune_at_most[i]]
        #         useless_filter_index[i]=index

        return useless_filter_index,module_list,neural_list,FIRE

def sort_inactive_filters(net,
                          percent_of_inactive_filter,
                          FIRE,
                          num_filters_to_prune_at_most
                          ):
    num_conv=0
    filter_index=[]                                 #index of the filter in its layer
    filter_layer=[]                                 #which layer the filter is in
    for name,mod in net.named_modules():
        if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
            filter_layer +=[num_conv for i in range(mod.out_channels)]
            filter_index+=[i for i in range(mod.out_channels)]
            num_conv+=1

    useless_filter_index=[[] for i in range(num_conv)]
    cutoff_rank_increase = -1
    delta = 0
    while cutoff_rank_increase != delta:
        cutoff_rank_increase = delta
        delta = 0
        cutoff_rank = int(percent_of_inactive_filter * len(FIRE)) + cutoff_rank_increase
        inactive_rank = np.argsort(-np.array(FIRE))[:cutoff_rank]  # arg for top (percent_of_inactive_filter)*100% of inactive filters
        inactive_filter_index = np.array(filter_index)[inactive_rank]  # index of top (percent_of_inactive_filter)*100% inactive filters
        inactive_filter_layer = np.array(filter_layer)[inactive_rank]
        for i in range(num_conv):
            index = inactive_filter_index[np.where(inactive_filter_layer == i)]  # index of inactive filters in layer i
            if num_filters_to_prune_at_most is not None:
                if len(index) > num_filters_to_prune_at_most[i]:
                    if num_filters_to_prune_at_most[i]<0:
                        num_filters_to_prune_at_most[i]=0
                    delta += len(index) - num_filters_to_prune_at_most[i]  # number of inactive filters excluded because of the restriction
                    index = index[:num_filters_to_prune_at_most[i]]
            useless_filter_index[i] = index
    return useless_filter_index


def check_ReLU_alive(net,neural_dead_times,data=None,data_loader=None,max_data_to_test=10000):
    handle = []
    global relu_list                                                        #list containing relu module
    global neural_list
    relu_list=[]
    neural_list=dict()


    #register a hook for ReLU
    for mod in net.modules():
        if isinstance(mod, torch.nn.ReLU):
            handle.append(mod.register_forward_hook(cal_dead_times))

    if data_loader is not None:
        evaluate_net(net, data_loader, False,max_data_to_test=max_data_to_test)
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

def cal_dead_times(module, input, output):
    if len(output.shape) !=4:                                           #ReLU for fc
        return
    if module not in relu_list:
        relu_list.append(module)
    if module not in neural_list.keys():
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

# def check_conv_alive_layerwise(net,neural_dead_times,batch_size):
#     '''
#     generate random data as input for each conv layer to test the dead neural
#     :param net:
#     :param neural_dead_times:
#     :param data:
#     :param data_loader:
#     :param use_random_data:
#     :return:
#     '''
#     handle = []
#     global conv_list                                                        #list containing relu module
#     global neural_list
#     global input_shape_list,output_shape_list
#     global mean_list,std_list
#     conv_list=[]
#     neural_list=dict()
#     input_shape_list=[]
#     output_shape_list=[]
#     mean_list=[[0.485, 0.456, 0.406]]                                       #mean list initiated with mean value in layer one
#     std_list=[[0.229, 0.224, 0.225]]
#
#     module_block_list=[]                                                      #2-d list
#
#     #register a hook for ReLU
#     num_conv=-1
#     new_block=False
#     for mod in net.modules():
#         if new_block is True:
#             module_block_list[num_conv].append(mod)
#         if isinstance(mod, torch.nn.Conv2d):
#             num_conv+=1
#             module_block_list.append([mod])
#             conv_list.append(mod)
#             new_block=True
#             handle.append(mod.register_forward_hook(record_input_output_size))
#         if isinstance(mod,torch.nn.ReLU):
#             new_block=False
#         if isinstance(mod,torch.nn.BatchNorm2d):
#             handle.append(mod.register_forward_hook(record_mean_std_layerwise))
#     num_conv+=1
#
#     data=generate_random_data.random_normal(num=1,dataset_name='cifar10')
#     net.eval()
#     net(data)
#
#     #close the hook
#     for h in handle:
#         h.remove()
#
#     for i in range(num_conv):
#         if i == 0:
#             data_foward=generate_random_data.random_normal(num=batch_size,size=input_shape_list[i],mean=mean_list[i],std=std_list[i],is_image=True)
#         else:
#             data_foward=generate_random_data.random_normal(num=batch_size,size=input_shape_list[i],mean=mean_list[i],std=std_list[i],is_image=False)
#
#         for mod in module_block_list[i]:
#             data_foward=mod(data_foward)
#         cal_dead_times(module=conv_list[i],input=None,output=data_foward)
#
#     cal_dead_neural_rate(neural_dead_times)
#     neural_list_temp = neural_list
#     conv_list_temp = conv_list
#     del conv_list, neural_list
#     return conv_list_temp, neural_list_temp

# def record_input_output_size(module, input, output):
#     input_shape_list.append(input[0].shape[1:])
#     output_shape_list.append(output.shape[1:])
#
# def record_mean_std_layerwise(module,input,output):
#     mean_list.append(module.running_mean.cpu().numpy())
#     std_list.append(np.sqrt(module.running_var.cpu().numpy()))


#def predict_dead_filters_classifier_version(net,
#                                          predictor,
#                                          min_ratio_dead_filters=0,
#                                          max_ratio_dead_filters=1,
#                                          filter_num_lower_bound=None
#                                          ):
#     '''
#     use trained predictor to predict dead filters in net
#     :param net:
#     :param predictor:
#     :param min_ratio_dead_filters: float, ensure the function will return at least (min_ratio_dead_filters)*100% of the filters.
#
#     :param max_ratio_dead_filters: float, ensure the function will return at most (max_ratio_dead_filters)*100% of the filters.
#                                           ensure the number of filters pruned will not be too large for one time.
#     :param filter_num_lower_bound: int, ensure the number of filters alive will be larger than filter_num_lower_bound
#                                           ensure the lower bound of filter number
#
#     :return:
#     '''
#
#     # dead_filter_index_data,_,_=find_useless_filters_data_version(net=net,filter_FIRE=0.9,batch_size=1200,neural_dead_times=1200)
#
#     dead_filter_index=[]
#     df_num_max=[]                           #max number of filter num
#     df_num_min=[]                           #min number of filter num
#     filter_num=[]                           #filter num in each layer
#     weight=[]
#     num_conv=0
#     for mod in net.features:
#         if isinstance(mod, torch.nn.modules.conv.Conv2d):
#             weight+=list(mod.weight.data.cpu().numpy())
#             filter_num.append(mod.weight.data.cpu().numpy().shape[0])
#             df_num_min.append(math.ceil(min_ratio_dead_filters*filter_num[num_conv]))                                            #lower bound of dead_filter's num
#             if filter_num_lower_bound is not None:                                                              #upper bound of dead_filter's num
#                 df_num_max.append(min(int(max_ratio_dead_filters*len(weight)),filter_num[num_conv]-filter_num_lower_bound[num_conv]))
#             else:
#                 df_num_max.append(int(max_ratio_dead_filters*len(weight)))
#
#             if df_num_max[num_conv]<df_num_min[num_conv]:
#                 print('Number of filters in layer{} is {}. At most {} filters will be predicted to be dead.'.format(num_conv,filter_num[num_conv],df_num_max[num_conv]))
#             num_conv+=1
#
# #todo:全部的卷积核一起标准化更合理
#     stat_filters,_= predict_dead_filter.statistics(weight, min_max_scaler=predictor.min_max_scaler)
#
#     s=0
#     for i in range(num_conv):
#         output = predictor.predict_proba(stat_filters[s:s+filter_num[i]])
#         dead_filter_proba_sorted=np.argsort(-output[:,1])                   #filter indices sorted by the probability of which to be dead
#         dead_filter_predicted=np.where(np.argmax(output,1))[0]             #filter indices of which are predicted to be dead
#         if dead_filter_predicted.shape[0]<df_num_min[i]:
#             print(i,'死亡卷积核太少')
#             dead_filter_predicted=dead_filter_proba_sorted[:df_num_min[i]]
#         if dead_filter_predicted.shape[0]>df_num_max[i]:
#             print(i,'死亡卷积核太多')
#             dead_filter_predicted=dead_filter_proba_sorted[:df_num_max[i]]
#         dead_filter_index.append(dead_filter_predicted)
#         s += filter_num[i]
#     return dead_filter_index
if __name__ == "__main__":
    print()







