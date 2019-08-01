import torch
import numpy as np
import train
import vgg
import torch.nn as nn
import copy
import logger

count_ops = 0

def measure_layer(layer, x, multi_add=1):
    type_name = str(layer)[:str(layer).find('(')].strip()
    #print(type_name)
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) //
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) //
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w // layer.groups * multi_add

    # ### ops_nonlinearity
    # elif type_name in ['ReLU']:
    #     delta_ops = x.numel()
    #
    # ### ops_pooling
    # elif type_name in ['AvgPool2d']:
    #     in_w = x.size()[2]
    #     kernel_ops = layer.kernel_size * layer.kernel_size
    #     out_w = int((in_w + 2 * layer.padding - layer.kernel_size) // layer.stride + 1)
    #     out_h = int((in_w + 2 * layer.padding - layer.kernel_size) // layer.stride + 1)
    #     delta_ops = x.size()[1] * out_w * out_h * kernel_ops
    #
    # elif type_name in ['AdaptiveAvgPool2d']:
    #     delta_ops = x.numel()

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = weight_ops + bias_ops

    # elif type_name in ['BatchNorm2d']:
    #     normalize_ops = x.numel()
    #     scale_shift = normalize_ops
    #     delta_ops = normalize_ops + scale_shift
    #
    # ### ops_nothing
    # elif type_name in ['Dropout2d', 'DropChannel', 'Dropout']:
    #     delta_ops = 0

    ### unknown layer type
    else:
        delta_ops=0
        #raise TypeError('unknown layer type: %s' % type_name)

    global count_ops
    count_ops += delta_ops
    return

def is_leaf(module):
    return sum(1 for x in module.children()) == 0

# 判断是否为需要计算flops的结点模块
def should_measure(module):
    # 代码中的残差结构可能定义了空内容的Sequential
    if str(module).startswith('Sequential'):
        return False
    if is_leaf(module):
        return True
    return False


def measure_model_mine(net,dataset_name='imagenet'):
    if dataset_name is 'imagenet':
        shape=(1,3,224,224)
    elif dataset_name is 'cifar10':
        shape=(1,3,32,32)
    global count_ops
    data = torch.zeros(shape)
    if torch.cuda.is_available():
        data=data.cuda()
    for mod in net.modules():
        measure_layer(mod,data,)
    print('flop_num:{}'.format(count_ops))
    count_ops_temp=count_ops
    count_ops=0
    return count_ops_temp



def measure_model(net, dataset_name='imagenet',print_flop=True):
    model=copy.deepcopy(net)                    #防止把原模型做了改变

    if dataset_name is 'imagenet':
        shape=(1,3,224,224)
    elif dataset_name is 'cifar10':
        shape=(1,3,32,32)
    global count_ops
    data = torch.zeros(shape)
    if torch.cuda.is_available():
        data=data.cuda()

    # 将计算flops的操作集成到forward函数
    def new_forward(m):
        def lambda_forward(x):
            measure_layer(m, x)
            return m.old_forward(x)
        return lambda_forward

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                # 新增一个old_forward属性保存默认的forward函数
                # 便于计算flops结束后forward函数的恢复
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # 对修改后的forward函数进行恢复
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    # forward过程中对全局的变量count_ops进行更新
    model.forward(data)
    restore_forward(model)
    print('flop_num:{}'.format(count_ops))
    count_ops_temp=count_ops
    count_ops=0
    return count_ops_temp

if __name__ == '__main__':
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

    # checkpoint=torch.load('/home/victorfang/Desktop/sample_num=256032.tar',map_location='cpu')
    # net=checkpoint['net']

    print(measure_model(net,dataset_name='cifar10'))

    #print(measure_model_mine(net,dataset_name='cifar10'))
    # ≈1.8G，和原文一致
