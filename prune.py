import torch
import numpy as np
import vgg


def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]



def prune_conv_layer(model, layer_index, filter_index):
    ''' layer_index:要删的是哪一层中的filter，这个layer_index指向的必须要是conv
        filter_index:要删layer_index层中的哪个filter
    '''
    _, conv = list(model.features._modules.items())[layer_index]            #获得要删filter的那层conv
    next_conv = None
    offset = 1

    while layer_index + offset < len(model.features._modules.items()):
        res = list(model.features._modules.items())[layer_index + offset]
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):                #获得要删的下一层的conv
            next_name, next_conv = res
            break
        offset = offset + 1

    new_conv = torch.nn.Conv2d(                                             #创建新的conv替代要删filter的conv
                                in_channels=conv.in_channels,
                                out_channels=conv.out_channels - 1,
                                kernel_size=conv.kernel_size,
                                stride=conv.stride,
                                padding=conv.padding,
                                dilation=conv.dilation,
                                groups=conv.groups,
                                bias=(conv.bias is not None))

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]     #将索引前的filer复制到新conv中
    new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]   #将索引后一个直至最后的所有filter复制到新conv中
    new_conv.weight.data = torch.from_numpy(new_weights)
    if torch.cuda.is_available():
        new_conv.weight.data=new_conv.weight.data.cuda()

    if conv.bias is not None:
        bias_numpy = conv.bias.data.cpu().numpy()

        bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index:] = bias_numpy[filter_index + 1:]
        new_conv.bias.data = torch.from_numpy(bias)
        if torch.cuda.is_available():
            new_conv.bias.data=new_conv.bias.data.cuda()
    if not next_conv is None:                                                       #next_conv中需要把对应的通道也删了
        next_new_conv = \
            torch.nn.Conv2d(in_channels=next_conv.in_channels - 1,
                            out_channels=next_conv.out_channels,
                            kernel_size=next_conv.kernel_size,
                            stride=next_conv.stride,
                            padding=next_conv.padding,
                            dilation=next_conv.dilation,
                            groups=next_conv.groups,
                            bias=(next_conv.bias is not None))

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights).cpu()
        if torch.cuda.is_available():
            next_new_conv.weight.data=next_new_conv.weight.data.cuda()
        if next_conv.bias is not None:
            next_new_conv.bias.data = next_conv.bias.data

    if not next_conv is None:                                                                     #根据新定义好的module重新定义sequential
        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index, layer_index + offset], [new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
        del model.features
        del conv

        model.features = features
        #todo：为什么要把relu改成batchnorm？
        model.features[layer_index + 1] = torch.nn.BatchNorm2d(model.features[layer_index].out_channels)

    else:
        # Prunning the last conv layer. This affects the first linear layer of the classifier.
        model.features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index], [new_conv]) for i, _ in enumerate(model.features)))
        model.features[layer_index + 1] = torch.nn.BatchNorm2d(model.features[layer_index].out_channels)

        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index + 1

        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")
        params_per_input_channel = int(old_linear_layer.in_features / conv.out_channels)

        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel,
                            old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        new_weights[:, : filter_index * params_per_input_channel] = \
            old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel:] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel:]

        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], [new_linear_layer]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

    return model

def select_and_prune_filter(model,layer_index,num_to_prune,ord):
    '''

    :param model: net model
    :param layer_index: layer in which the filters being pruned
    :param num_to_prune: number of filters to prune
    :param ord: which norm to compute as the standard. Support l1 and l2 norm
    :return: filter indexes in the [layer_index] layer
    '''
    if ord!=1 and ord !=2:
        raise TypeError('unsupported type of norm')
    _, conv = list(model.features._modules.items())[layer_index]            #get conv module
    weights = conv.weight.data.cpu().numpy()                                #get weight of all filters

    filter_norm=np.linalg.norm(weights,ord=ord,axis=(2,3))                            #compute filters' norm
    if ord==1:
        filter_norm=np.sum(filter_norm,axis=1)
    elif ord==2:
        filter_norm=np.square(filter_norm)
        filter_norm=np.sum(filter_norm,axis=1)

    filter_norm_sorted_index=np.argsort(filter_norm)

    model=prune_conv_layer(model,layer_index,filter_norm_sorted_index[0:num_to_prune])
    return model



if __name__ == "__main__":
    model= vgg.vgg11(False)
    select_and_prune_filter(model,layer_index=3,num_to_prune=2,ord=2)
    prune_conv_layer(model,layer_index=3,filter_index=1)