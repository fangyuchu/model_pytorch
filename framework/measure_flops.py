import torch
from network import vgg
import torch.nn as nn
import copy
import numpy as np
from network.modules import conv2d_with_mask,conv2d_with_mask_shortcut
count_ops = 0

def measure_layer(name,layer, x, multi_add=1):
    type_name = str(layer)[:str(layer).find('(')].strip()
    if isinstance(layer,nn.Conv2d):
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) //
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) //
                    layer.stride[1] + 1)

        in_channels=layer.in_channels
        out_channels=layer.out_channels
        if isinstance(layer,conv2d_with_mask):
            out_channels=out_channels-torch.sum(layer.mask == 0).detach().cpu().numpy()
        delta_ops = in_channels * out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w // layer.groups * multi_add

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
# def should_measure(module):
#     # 代码中的残差结构可能定义了空内容的Sequential
#     if str(module).startswith('Sequential'):
#         return False
#     if is_leaf(module):
#         return True
#     return False

def should_measure(mod):
    if isinstance(mod,nn.Conv2d):
        return True
    elif isinstance(mod,nn.Linear):
        return True
    else:
        return False


def measure_model(net, dataset_name='imagenet', print_flop=True):

    if dataset_name == 'imagenet'or dataset_name == 'tiny_imagenet':
        shape=(2,3,224,224)
    elif dataset_name == 'cifar10' or dataset_name == 'cifar100':
        shape=(2,3,32,32)


    global count_ops
    data = torch.zeros(shape)
    if torch.cuda.is_available():
        data=data.cuda()

    # 将计算flops的操作集成到forward函数
    def new_forward(name,m):
        def lambda_forward(x):
            measure_layer(name,m, x)
            return m.old_forward(x)
        return lambda_forward

    # def modify_forward(model):
    #     for child in model.children():
    #         if should_measure(child):
    #             # 新增一个old_forward属性保存默认的forward函数
    #             # 便于计算flops结束后forward函数的恢复
    #             child.old_forward = child.forward
    #             child.forward = new_forward(child)
    #         else:
    #             modify_forward(child)

    # def restore_forward(model):
    #     for child in model.children():
    #         # 对修改后的forward函数进行恢复
    #         if is_leaf(child) and hasattr(child, 'old_forward'):
    #             child.forward = child.old_forward
    #             child.old_forward = None
    #         else:
    #             restore_forward(child)

    def modify_forward(model):
        for name,mod in model.named_modules():
            if should_measure(mod):
                # 新增一个old_forward属性保存默认的forward函数
                # 便于计算flops结束后forward函数的恢复
                mod.old_forward = mod.forward
                mod.forward = new_forward(name,mod)

    def restore_forward(model):
        for name,mod in model.named_modules():
            # 对修改后的forward函数进行恢复
            if hasattr(mod, 'old_forward'):
                mod.forward = mod.old_forward
                mod.old_forward = None

    modify_forward(net)
    # forward过程中对全局的变量count_ops进行更新
    net.eval()
    net.forward(data)
    restore_forward(net)
    if print_flop:
        print('flop_num:{}'.format(count_ops))
    count_ops_tmp=int(count_ops)
    count_ops=0
    return count_ops_tmp

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
    # network=checkpoint['net']

    print(measure_model(net,dataset_name='cifar10'))

    #print(measure_model_mine(network,dataset_name='cifar10'))
    # ≈1.8G，和原文一致
