import matplotlib.pyplot as plt
from network.modules import conv2d_with_mask
import numpy as np
from network import net_with_predicted_mask
import torch
from framework import config as conf
import os
def draw_masked_net_pruned(net):
    num_layer = 0  # num of total conv layers
    layer_mask = []
    max_out_channels = 0  # max number of filters in one layer
    for name, mod in net.named_modules():
        if isinstance(mod, conv2d_with_mask):
            layer_mask += [mod.mask.detach().cpu().numpy()]
            max_out_channels = max(mod.out_channels, max_out_channels)
            num_layer += 1

    for i, mask in enumerate(layer_mask):
        if len(mask) < max_out_channels:
            if (max_out_channels - len(mask)) % 2 != 0:
                raise Exception('number of filters in a conv is not even?')
            pad_len = int((max_out_channels - len(mask)) / 2)
            layer_mask[i] = np.pad(mask, (pad_len, pad_len), 'constant',
                                   constant_values=-2)  # pad the mask with -2 indicating a placeholder

    margin = 0.02  # margin of the figure
    # draw in 0.1~0.9
    h_delta = (1 - 2 * margin) / num_layer  # space of each row
    w_delta = (1 - 2 * margin) / max_out_channels  # space of each col
    square_h = 0.5 * h_delta  # height of a square
    square_w = 0.5 * w_delta  # width of a square

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('width', fontsize=20)
    plt.ylabel('layer/height', fontsize=30)

    y = 1 - margin  # start painting from the top of the graph
    for i in range(num_layer):
        y -= h_delta
        x = margin  # start painting from the left of the graph
        for j in range(max_out_channels):
            if layer_mask[i][j] == -2:  # placeholder
                pass
            #todo:这里是不是有点问题
            elif layer_mask[i][j] == 0:  # pruned filter
                # (left bottom), width,height,color,transparency?, fill the square# (left bottom), width,height,color,transparency?, fill the square
                rect = plt.Rectangle((x, y), square_w, square_h, color=(1, 0, 0), alpha=0.3,fill=False)
                ax.add_patch(rect)
            else:  # remaining filter
                rect = plt.Rectangle((x, y), square_w, square_h, color='b', alpha=0.3, fill=True)
                ax.add_patch(rect)

            x += w_delta

    return fig

def draw_masked_net(net):
    num_layer = 0  # num of total conv layers
    layer_mask = []
    max_out_channels = 0  # max number of filters in one layer
    for name, mod in net.named_modules():
        if isinstance(mod, conv2d_with_mask):
            layer_mask += [mod.mask.detach().cpu().numpy()]
            max_out_channels = max(mod.out_channels, max_out_channels)
            num_layer += 1

    for i, mask in enumerate(layer_mask):
        if len(mask) < max_out_channels:
            if (max_out_channels - len(mask)) % 2 != 0:
                raise Exception('number of filters in a conv is not even?')
            pad_len = int((max_out_channels - len(mask)) / 2)
            layer_mask[i] = np.pad(mask, (pad_len, pad_len), 'constant',
                                   constant_values=-2)  # pad the mask with -2 indicating a placeholder

    margin = 0.02  # margin of the figure
    # draw in 0.1~0.9
    h_delta = (1 - 2 * margin) / num_layer  # space of each row
    w_delta = (1 - 2 * margin) / max_out_channels  # space of each col
    square_h = 0.5 * h_delta  # height of a square
    square_w = 0.5 * w_delta  # width of a square

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)

    plt.imshow(layer_mask,cmap=plt.cm.RdBu,interpolation='nearest',vmin=0, vmax=1)
    cmap_custom=plt.get_cmap('RdBu')
    cmap_custom.set_under('white')

    # heatmap=ax.pcolor(layer_mask,cmap=plt.cm.hot,vmin=0, vmax=1,interpolation='nearest')
    # heatmap.cmap.set_under('black')
    # bar = fig.colorbar(heatmap, extend='both')

    plt.colorbar()



    plt.xticks([])
    plt.yticks([])
    plt.xlabel('width', fontsize=20)
    plt.ylabel('layer/height', fontsize=30)

    plt.show()
    return fig



if __name__ == "__main__":
    from network import resnet_cifar
    import torch.nn as nn
    # 网络参数
    add_shortcut_ratio = 0.9  # 不是这儿！！！
    mask_update_freq = 1000
    mask_update_epochs = 900
    mask_training_start_epoch = 1
    mask_training_stop_epoch = 80

    total_flop = 125485706
    prune_ratio = 0
    flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
    gradient_clip_value = None
    learning_rate_decay_epoch = [mask_training_stop_epoch + 1 * i for i in [80, 120]]
    num_epochs = 160 * 1 + mask_training_stop_epoch
    #
    net = resnet_cifar.resnet56(num_classes=10).cuda()
    net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                           net_name='resnet56',
                                                                           dataset_name='cifar10',
                                                                           mask_update_epochs=mask_update_epochs,
                                                                           mask_update_freq=mask_update_freq,
                                                                           flop_expected=flop_expected,
                                                                           gcn_rounds=2,
                                                                           mask_training_start_epoch=mask_training_start_epoch,
                                                                           mask_training_stop_epoch=mask_training_stop_epoch,
                                                                           batch_size=128,
                                                                           add_shortcut_ratio=add_shortcut_ratio
                                                                           )
    net = net.cuda()
    i = 1

    checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet56', str(i) + '.tar'),map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])
    mask = net.extractor(net, net.net_name, net.dataset_name)  # predict mask using extractor
    mask=mask.abs()
    # mask=(mask-mask.min())/(mask.max()-mask.min())
    lo = hi = 0
    last_conv_mask = None
    for name, mod in net.net.named_modules():
        if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
            hi += mod.out_channels
            mod.set_mask(mask[lo:hi].view(-1))  # update mask for each conv
            lo = hi
            last_conv_mask = mod.mask

    fig=draw_masked_net(net)
    print()


