import config as conf
import torch
import torch.nn as nn
import tensorly as tl

from tensorly.decomposition import parafac


def cp_decomposition_net(net):
    tl.set_backend('pytorch')
    N = len(net.features._modules.keys())
    for i, key in enumerate(net.features._modules.keys()):
        if i >= N - 2:
            break
        if isinstance(net.features._modules[key], torch.nn.modules.conv.Conv2d):
            conv_layer = net.features._modules[key]
            rank = max(conv_layer.weight.data.cpu().numpy().shape) // 3
            decomposed = cp_decomposition_conv_layer(conv_layer, rank)  # CP分解

            net.features._modules[key] = decomposed

    for param in net.parameters():
        param.requires_grad = True
    if torch.cuda.is_available():
        net.cuda()
    return net

def cp_decomposition_conv_layer(layer, rank):
    """ Gets a conv layer and a target rank,
        returns a nn.Sequential object with the decomposition """

    # Perform CP decomposition on the layer weight tensorly.
    last, first, vertical, horizontal = parafac(layer.weight.data, rank=rank, init='svd')

    # 构造新的结构
    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                                             out_channels=first.shape[1],
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             dilation=layer.dilation,
                                             bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1],
                                               out_channels=vertical.shape[1],
                                               kernel_size=(vertical.shape[0], 1),
                                               stride=1,
                                               padding=(layer.padding[0], 0),
                                               dilation=layer.dilation,
                                               groups=vertical.shape[1],
                                               bias=False)

    depthwise_horizontal_layer = torch.nn.Conv2d(in_channels=horizontal.shape[1],
                                                 out_channels=horizontal.shape[1],
                                                 kernel_size=(1, horizontal.shape[0]), stride=layer.stride,
                                                 padding=(0, layer.padding[0]),
                                                 dilation=layer.dilation,
                                                 groups=horizontal.shape[1],
                                                 bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                             out_channels=last.shape[0],
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             dilation=layer.dilation,
                                             bias=True)
    # 给最后一层添加bias，即复制原来的bias
    pointwise_r_to_t_layer.bias.data = layer.bias.data

    # 往新的结构中赋值
    depthwise_horizontal_layer.weight.data = torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_s_to_r_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

    # 整合4个卷积层
    new_layers = [pointwise_s_to_r_layer,
                  depthwise_vertical_layer,
                  depthwise_horizontal_layer,
                  pointwise_r_to_t_layer]
    return nn.Sequential(*new_layers)  # 返回必须以打包的方式，仅改变被分解的层