import torch
import numpy as np
from network import vgg
import torch.nn as nn
def replace_layers(module,old_mod,new_mod):
    for i in range(len(old_mod)):
        if module is old_mod[i]:
            return new_mod[i]
    return module

def create_module_list(module,key='',prefix=''):
    module_dict=getattr(module,'_modules')
    if not module_dict:                                 #module_dict is empty, which means this module is the last node.
        if prefix =='':
            return [key]
        else:
            return [prefix+'.'+key]
    modules_list=list()
    if key != '':                                               #module is not network
        if prefix =='':
            prefix=key
        else:
            prefix=prefix+'.'+key
    for k in module_dict:
        modules_list+=create_module_list(module_dict[k],k,prefix)
    return modules_list

def string_to_module(net, string):
    """
    将字符串表示的module转换成module类型
    """
    tmp_list = string.split(".")
    if (len(tmp_list) == 1):
        replace_module = getattr(net, string)
    else:
        tmp = getattr(net, tmp_list[0])
        tmp = getattr(tmp, tmp_list[1])
        replace_module = getattr(tmp, tmp_list[2])
    return replace_module

def get_module(model,name):
    '''

    :param model:
    :param name:
    :return: the second last module in name
    '''
    attr=name.split('.')
    mod=model
    for i in range(len(attr)-1):
        mod=getattr(mod,attr[i])

    return getattr(mod,'_modules')

def prune_conv_layer(model, layer_index, filter_index):
    ''' layer_index:要删的卷基层的索引
        filter_index:要删layer_index层中的哪个filter
    '''
    if len(filter_index)==0:  #no filter need to be pruned
        return model
    conv=None                                                               #获取要删filter的那层conv
    batch_norm=None                                                         #如果有的话：获取要删的conv后的batch normalization层
    next_conv=None                                                          #如果有的话：获取要删的那层后一层的conv，用于删除对应通道
    i=0

    for name, mod in model.named_modules():
        if conv is not None:
            if isinstance(mod, torch.nn.modules.conv.Conv2d):            #要删的filter后一层的conv
                next_conv = mod
                next_conv_name=name
                break
            elif isinstance(mod,torch.nn.modules.BatchNorm2d):             #要删的filter后一层的batch normalization
                batch_norm=mod
                batch_norm_name=name
            else:
                continue
        if isinstance(mod,torch.nn.modules.conv.Conv2d):
            if i==layer_index:                                              #要删filter的conv
                conv=mod
                conv_name=name
            i += 1


    new_conv = torch.nn.Conv2d(                                             #创建新的conv替代要删filter的conv
                                in_channels=conv.in_channels,
                                out_channels=conv.out_channels - len(filter_index),
                                kernel_size=conv.kernel_size,
                                stride=conv.stride,
                                padding=conv.padding,
                                dilation=conv.dilation,
                                groups=conv.groups,
                                bias=(conv.bias is not None))

    #复制其他filter
    old_weights = conv.weight.data.cpu().numpy()
    new_weights=new_conv.weight.data.cpu().numpy()
    new_weights[:]=old_weights[[i for i in range(old_weights.shape[0]) if i not in filter_index]]  #复制剩余的filters的weight

    if conv.bias is not None:
        old_bias = conv.bias.data.cpu().numpy()
        new_bias = new_conv.bias.data.cpu().numpy()
        new_bias[:] = old_bias[[i for i in range(old_bias.shape[0]) if i not in filter_index]]  # 复制剩余的filters的bias
    if torch.cuda.is_available():
        new_conv.cuda()
    
    _modules=get_module(model=model,name=conv_name)
    _modules[conv_name.split('.')[-1]]=new_conv

    if batch_norm is not None:
        new_batch_norm=torch.nn.BatchNorm2d(new_conv.out_channels)
        new_batch_norm.num_batches_tracked=batch_norm.num_batches_tracked

        old_weights = batch_norm.weight.data.cpu().numpy()                                      #删除weight
        new_weights = new_batch_norm.weight.data.cpu().numpy()
        new_weights[:] = old_weights[[i for i in range(old_weights.shape[0]) if i not in filter_index]]

        old_bias=batch_norm.bias.data.cpu().numpy()                                             #删除bias
        new_bias=new_batch_norm.bias.data.cpu().numpy()
        new_bias[:] = old_bias[[i for i in range(old_bias.shape[0]) if i not in filter_index]]

        old_running_mean=batch_norm.running_mean.cpu().numpy()
        new_running_mean=new_batch_norm.running_mean.cpu().numpy()
        new_running_mean[:] = old_running_mean[[i for i in range(old_running_mean.shape[0]) if i not in filter_index]]

        old_running_var=batch_norm.running_var.cpu().numpy()
        new_running_var=new_batch_norm.running_var.cpu().numpy()
        new_running_var[:] = old_running_var[[i for i in range(old_running_var.shape[0]) if i not in filter_index]]

        if torch.cuda.is_available():
            new_batch_norm.cuda()

        
        _modules = get_module(model=model, name=batch_norm_name)
        _modules[batch_norm_name.split('.')[-1]] = new_batch_norm
        # model.features = torch.nn.Sequential(
        #     *(replace_layers(mod, [batch_norm], [new_batch_norm]) for mod in model.features))
        

    if next_conv is not None:                                                       #next_conv中需要把对应的通道也删了
        new_next_conv = \
            torch.nn.Conv2d(in_channels=next_conv.in_channels - len(filter_index),
                            out_channels=next_conv.out_channels,
                            kernel_size=next_conv.kernel_size,
                            stride=next_conv.stride,
                            padding=next_conv.padding,
                            dilation=next_conv.dilation,
                            groups=next_conv.groups,
                            bias=(next_conv.bias is not None))

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = new_next_conv.weight.data.cpu().numpy()
        new_weights[:] = old_weights[:,[i for i in range(old_weights.shape[1]) if i not in filter_index],:,:]  # 复制剩余的filters的weight

        if next_conv.bias is not None:
            new_next_conv.bias.data = next_conv.bias.data
        if torch.cuda.is_available():
            new_next_conv.cuda()

        
        _modules = get_module(model=model, name=next_conv_name)
        _modules[next_conv_name.split('.')[-1]] = new_next_conv

    else:
        # Prunning the last conv layer. This affects the first linear layer of the classifier.
        old_linear_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                linear_layer_name=name
                break

        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")
        params_per_input_channel = int(old_linear_layer.in_features / conv.out_channels)

        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - len(filter_index)*params_per_input_channel,
                            old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        node_index=[]
        for f in filter_index:
            node_index.extend([i for i in range(f*params_per_input_channel,(f+1)*params_per_input_channel)])

        new_weights[:] = old_weights[:,[i for i in range(old_weights.shape[1]) if i not in node_index]]  # 复制剩余的filters的weight

        #
        # new_weights[:, : filter_index * params_per_input_channel] = \
        #     old_weights[:, : filter_index * params_per_input_channel]
        # new_weights[:, filter_index * params_per_input_channel:] = \
        #     old_weights[:, (filter_index + 1) * params_per_input_channel:]

        new_linear_layer.bias.data = old_linear_layer.bias.data

        if torch.cuda.is_available():
            new_linear_layer.cuda()
        
        _modules = get_module(model=model, name=linear_layer_name)
        _modules[linear_layer_name.split('.')[-1]] = new_linear_layer

    return model

def prune_conv_layer_resnet(net, layer_index, filter_index, modules_list):
    """
    :param net:
    :param layer_index: 要删的卷基层的索引,从0开始
    :param filter_index: 要删layer_index层中的哪个filter
    :return:
    """
    if len(filter_index)==0:  #no filter need to be pruned
        return net
    conv_to_prune = None  # 获取要删filter的那层conv
    batch_norm = None  # 如果有的话：获取要删的conv后的batch normalization层
    next_conv = None  # 如果有的话：获取要删的那层后一层的conv，用于删除对应通道
    i = 0

    conv_name = ""
    next_conv_name = ""
    batch_norm_name = ""
    for string in modules_list:
        if conv_name != "":
            if "conv" in string:
                next_conv_name = string
                break
            elif "bn" in string:
                batch_norm_name = string
            else:
                continue
        if "conv" in string:
            if i == layer_index:
                conv_name = string
            i += 1

    i = 0
    for name,mod in net.named_modules():
        if conv_to_prune is not None:
            if isinstance(mod, torch.nn.Conv2d):  # 要删的filter后一层的conv
                if 'downsample' in name:
                    raise Exception('Pruning last conv in block is not implemented yet.')
                next_conv = mod
                break
            elif isinstance(mod, torch.nn.BatchNorm2d):  # 要删的filter后一层的batch normalization
                batch_norm = mod
            else:
                continue
        if isinstance(mod, torch.nn.Conv2d) and 'downsample' not in name:
            if i == layer_index:  # 找到要删filter的conv
                conv_to_prune = mod
            i += 1

    new_conv = torch.nn.Conv2d(  # 创建新的conv替代要删filter的conv
        in_channels=conv_to_prune.in_channels,
        out_channels=conv_to_prune.out_channels - len(filter_index),
        kernel_size=conv_to_prune.kernel_size,
        stride=conv_to_prune.stride,
        padding=conv_to_prune.padding,
        dilation=conv_to_prune.dilation,
        groups=conv_to_prune.groups,
        bias=(conv_to_prune.bias is not None))

    # 复制其他filter
    old_weights = conv_to_prune.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()
    new_weights[:] = old_weights[
        [i for i in range(old_weights.shape[0]) if i not in filter_index]]  # 复制剩余的filters的weight

    if conv_to_prune.bias is not None:
        old_bias = conv_to_prune.bias.data.cpu().numpy()
        new_bias = new_conv.bias.data.cpu().numpy()
        new_bias[:] = old_bias[[i for i in range(old_bias.shape[0]) if i not in filter_index]]  # 复制剩余的filters的bias
    if torch.cuda.is_available():
        new_conv.cuda()

    # 替换
    _modules=get_module(model=net,name=conv_name)
    _modules[conv_name.split('.')[-1]]=new_conv

    if batch_norm is not None:
        new_batch_norm = torch.nn.BatchNorm2d(new_conv.out_channels)
        new_batch_norm.num_batches_tracked = batch_norm.num_batches_tracked

        old_weights = batch_norm.weight.data.cpu().numpy()  # 删除weight
        new_weights = new_batch_norm.weight.data.cpu().numpy()
        new_weights[:] = old_weights[[i for i in range(old_weights.shape[0]) if i not in filter_index]]

        old_bias = batch_norm.bias.data.cpu().numpy()  # 删除bias
        new_bias = new_batch_norm.bias.data.cpu().numpy()
        new_bias[:] = old_bias[[i for i in range(old_bias.shape[0]) if i not in filter_index]]

        old_running_mean = batch_norm.running_mean.cpu().numpy()
        new_running_mean = new_batch_norm.running_mean.cpu().numpy()
        new_running_mean[:] = old_running_mean[[i for i in range(old_running_mean.shape[0]) if i not in filter_index]]

        old_running_var = batch_norm.running_var.cpu().numpy()
        new_running_var = new_batch_norm.running_var.cpu().numpy()
        new_running_var[:] = old_running_var[[i for i in range(old_running_var.shape[0]) if i not in filter_index]]

        if torch.cuda.is_available():
            new_batch_norm.cuda()
        # 替换
        _modules = get_module(model=net, name=batch_norm_name)
        _modules[batch_norm_name.split('.')[-1]] = new_batch_norm


    if next_conv is not None:  # next_conv中需要把对应的通道也删了
        new_next_conv = \
            torch.nn.Conv2d(in_channels=next_conv.in_channels - len(filter_index),
                            out_channels=next_conv.out_channels,
                            kernel_size=next_conv.kernel_size,
                            stride=next_conv.stride,
                            padding=next_conv.padding,
                            dilation=next_conv.dilation,
                            groups=next_conv.groups,
                            bias=(next_conv.bias is not None))

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = new_next_conv.weight.data.cpu().numpy()
        new_weights[:] = old_weights[:, [i for i in range(old_weights.shape[1]) if i not in filter_index], :,
                         :]  # 复制剩余的filters的weight

        if next_conv.bias is not None:
            new_next_conv.bias.data = next_conv.bias.data
        if torch.cuda.is_available():
            new_next_conv.cuda()
        # 替换
        _modules = get_module(model=net, name=next_conv_name)
        _modules[next_conv_name.split('.')[-1]] = new_next_conv

    else:
        # Prunning the last conv layer. This affects the first linear layer of the classifier.
        old_linear_layer = None
        for _, module in net.named_modules():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break

        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")
        params_per_input_channel = int(old_linear_layer.in_features / conv_to_prune.out_channels)

        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - len(filter_index) * params_per_input_channel,
                            old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        node_index = []
        for f in filter_index:
            node_index.extend([i for i in range(f * params_per_input_channel, (f + 1) * params_per_input_channel)])

        new_weights[:] = old_weights[:,
                         [i for i in range(old_weights.shape[1]) if i not in node_index]]  # 复制剩余的filters的weight



        new_linear_layer.bias.data = old_linear_layer.bias.data

        if torch.cuda.is_available():
            new_linear_layer.cuda()

        net.fc = new_linear_layer

    return net


# def select_and_prune_filter(model,ord,layer_index=0,num_to_prune=0,percent_of_pruning=0):
#     '''
#
#     :param model: network model
#     :param ord: which norm to compute as the standard. Support l1 and l2 norm
#     :param layer_index: layer in which the filters being pruned. If being set to 0, all conv layers will be pruned.
#     :param num_to_prune: number of filters to prune. Disabled if percent_of_pruning is not 0
#     :param percent percent_of_pruning: percent of filters to prune for one conv
#     :return: filter indexes in the [layer_index] layer
#     '''
#     if ord!=1 and ord !=2:
#         raise TypeError('unsupported type of norm')
#
#     i = 0
#     conv_index=-1                                                       #index of the conv in model.features
#     for mod in model.features:
#         conv_index+=1
#         if isinstance(mod, torch.nn.modules.conv.Conv2d):
#             i += 1
#             if i == layer_index:                                        # hit the conv to be pruned
#                 conv=mod
#                 break
#     if percent_of_pruning is not 0:
#         if num_to_prune is not 0:
#             print('Warning: Param: num_to_prune disabled!')
#         num_to_prune=int(conv.out_channels*percent_of_pruning)
#     weights = model.features[conv_index].weight.data.cpu().numpy()  # get weight of all filters
#
#     filter_norm=np.linalg.norm(weights,ord=ord,axis=(2,3))          #compute filters' norm
#     if ord==1:
#         filter_norm=np.sum(filter_norm,axis=1)
#     elif ord==2:
#         filter_norm=np.square(filter_norm)
#         filter_norm=np.sum(filter_norm,axis=1)
#     filter_min_norm_index=np.argsort(filter_norm)
#     model=prune_conv_layer(model,layer_index,filter_min_norm_index[:num_to_prune])
#
#     return model



if __name__ == "__main__":
    model= vgg.vgg16_bn(pretrained=True)
    # select_and_prune_filter(model,layer_index=3,num_to_prune=2,ord=2)
    # prune_conv_layer(model,layer_index=3,filter_index=1)