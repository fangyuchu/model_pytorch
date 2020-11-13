import os,sys
sys.path.append('../')
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
            elif layer_mask[i][j] == 0:  # pruned filter
                # (left bottom), width,height,color,transparency?, fill the square# (left bottom), width,height,color,transparency?, fill the square
                rect = plt.Rectangle((x, y), square_w, square_h, color=(1, 0, 0), alpha=0.3,fill=False)
                ax.add_patch(rect)
            else:  # remaining filter
                rect = plt.Rectangle((x, y), square_w, square_h, color='b', alpha=0.3, fill=True)
                ax.add_patch(rect)

            x += w_delta

    return fig

def draw_masked_net(net,pic_name,path):
    num_layer = 0  # num of total conv layers
    layer_mask = []
    max_out_channels = 0  # max number of filters in one layer
    for name, mod in net.named_modules():
        if isinstance(mod, conv2d_with_mask):
            layer_mask += [mod.mask.detach().cpu().numpy()]
            max_out_channels = max(mod.out_channels, max_out_channels)
            num_layer += 1

    layer_width=set()

    for i, mask in enumerate(layer_mask):
        layer_width.add(len(mask))
        if len(mask) < max_out_channels:
            if (max_out_channels - len(mask)) % 2 != 0:
                raise Exception('number of filters in a conv is not even?')
            # pad_len = int((max_out_channels - len(mask)) )
            # layer_mask[i] = np.pad(mask,  (0,pad_len), 'constant',
            #                        constant_values=-2)  # pad the mask with -2 indicating a placeholder
            pad_len = int((max_out_channels - len(mask)) / 2)
            layer_mask[i] = np.pad(mask, (pad_len, pad_len), 'constant',
                                   constant_values=-2)  # pad the mask with -2 indicating a placeholder


    layer_mask=np.array(layer_mask)
    layer_mask=layer_mask.T
    for i,l in enumerate(layer_mask):
        layer_mask[i]=l[::-1]
    # layer_mask=layer_mask[::-1]
    margin = 0.02  # margin of the figure
    # draw in 0.1~0.9
    h_delta = (1 - 2 * margin) / num_layer  # space of each row
    w_delta = (1 - 2 * margin) / max_out_channels  # space of each col
    square_h = 0.5 * h_delta  # height of a square
    square_w = 0.5 * w_delta  # width of a square
    # plt.style.use('fivethirtyeight')

    fig,ax=plt.subplots(figsize=(8,5))

    im=ax.imshow(layer_mask,cmap=plt.cm.YlOrRd,interpolation='nearest',vmin=0, vmax=1,aspect='auto')
    cmap_custom=plt.get_cmap('YlOrRd')
    cmap_custom.set_under('#A9A9A9')

    # heatmap=ax.pcolor(layer_mask,cmap=plt.cm.hot,vmin=0, vmax=1,interpolation='nearest')
    # heatmap.cmap.set_under('black')
    # bar = fig.colorbar(heatmap, extend='both')

    fontsize = 20

    cb=ax.figure.colorbar(im,ax=ax)
    cb.ax.tick_params(labelsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    # ax.set_xticks(np.array(list(layer_width))-1)
    # ax.set_xticklabels(list(layer_width))

    xtick_num=8
    gap=int(layer_mask.shape[1]/xtick_num)
    #todo:这里不对，横纵坐标搞错
    xticks=[i for i in range(0,layer_mask.shape[1],gap)]
    # xticks+=[layer_mask.shape[1]-1]
    xticks=np.array(xticks)
    ax.set_xticks(xticks)
    xticks=xticks[::-1]
    ax.set_xticklabels(xticks+1)
    ax.set_yticks([])

    # for i in range(0,14,1):
    #     ax.text(i, 1, str(i), fontsize=12)
    # for i in range(0,560,20):
    #     ax.text(1, i, str(i), fontsize=12)


    # #resnet56
    ax.text(18,3,'1',fontsize=fontsize)
    ax.text(18,63,'64',fontsize=fontsize)
    ax.text(36,18,'1',fontsize=fontsize)
    ax.text(36,47,'32',fontsize=fontsize)
    ax.text(52,23,'1',fontsize=fontsize)
    ax.text(51,43,'16',fontsize=fontsize)
    # #vgg16
    # ax.text(5.5,25,'1',fontsize=fontsize)
    # ax.text(5.5,505,'512',fontsize=fontsize)
    # ax.text(8.5,140,'1',fontsize=fontsize)
    # ax.text(8.5,380,'256',fontsize=fontsize)
    # ax.text(10.5,205,'1',fontsize=fontsize)
    # ax.text(10.5,335,'128',fontsize=fontsize)
    # ax.text(11.9,215,'1',fontsize=fontsize)
    # ax.text(11.7,315,'64',fontsize=fontsize)


    bwith = 2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    ax.set_xlabel('Layer', fontsize=fontsize)
    ax.set_ylabel('Width', fontsize=fontsize)
    fig.tight_layout()

    plt.savefig(os.path.join(path,pic_name+'.png'))
    plt.show()
    return fig



if __name__ == "__main__":
    from network import resnet_cifar,vgg
    import torch.nn as nn
    # resnet56
    add_shortcut_ratio = 0.9  # 不是这儿！！！
    mask_update_freq = 1000
    mask_update_epochs = 900
    mask_training_start_epoch = 1
    mask_training_stop_epoch = 80

    total_flop = 126550666
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
    i = 10086

    # checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet56', str(i) + '.tar'),map_location='cpu')

    checkpoint=torch.load('/home/disk_new/model_saved/resnet56_predicted_mask_and_variable_shortcut_net_mask_newinner_80epoch_std_8/checkpoint/masked_net.pth')

    net.load_state_dict(checkpoint['state_dict'])
    mask = net.extractor(net, net.net_name, net.dataset_name)  # predict mask using extractor
    mask=mask.abs()
    lo = hi = 0
    last_conv_mask = None
    for name, mod in net.net.named_modules():
        if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
            hi += mod.out_channels
            mod.set_mask(mask[lo:hi].view(-1))  # update mask for each conv
            lo = hi
            last_conv_mask = mod.mask

    fig=draw_masked_net(net,pic_name='resnet56_'+str(i),path='/home/victorfang/')
    print()


    # # vgg16
    # add_shortcut_ratio = 0.9  # 不是这儿！！！
    # mask_update_freq = 1000
    # mask_update_epochs = 900
    # mask_training_start_epoch = 1
    # mask_training_stop_epoch = 80
    #
    # total_flop = 314017290
    # prune_ratio = 0
    # flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
    # gradient_clip_value = None
    # learning_rate_decay_epoch = [mask_training_stop_epoch + 1 * i for i in [80, 120]]
    # num_epochs = 160 * 1 + mask_training_stop_epoch
    # #
    # net = vgg.vgg16_bn(dataset_name='cifar10').cuda()
    # net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
    #                                                                        net_name='vgg16_bn',
    #                                                                        dataset_name='cifar10',
    #                                                                        mask_update_epochs=mask_update_epochs,
    #                                                                        mask_update_freq=mask_update_freq,
    #                                                                        flop_expected=flop_expected,
    #                                                                        gcn_rounds=2,
    #                                                                        mask_training_start_epoch=mask_training_start_epoch,
    #                                                                        mask_training_stop_epoch=mask_training_stop_epoch,
    #                                                                        batch_size=128,
    #                                                                        add_shortcut_ratio=add_shortcut_ratio
    #                                                                        )
    # net = net.cuda()
    # i = 2
    #
    # checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net','vgg16', str(i) + '.tar'),map_location='cpu')
    # net.load_state_dict(checkpoint['state_dict'])
    # mask = net.extractor(net, net.net_name, net.dataset_name)  # predict mask using extractor
    # mask=mask.abs()
    # lo = hi = 0
    # last_conv_mask = None
    # for name, mod in net.net.named_modules():
    #     if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
    #         hi += mod.out_channels
    #         mod.set_mask(mask[lo:hi].view(-1))  # update mask for each conv
    #         lo = hi
    #         last_conv_mask = mod.mask
    #
    # fig=draw_masked_net(net,pic_name='vgg16_'+str(i),path='/home/victorfang/')
    # print()

