import matplotlib.pyplot as plt
from network.modules import conv2d_with_mask
import numpy as np

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
                rect = plt.Rectangle((x, y), square_w, square_h, color='b', alpha=0.3,fill=False)
                ax.add_patch(rect)
            else:  # remaining filter
                rect = plt.Rectangle((x, y), square_w, square_h, color='b', alpha=0.3, fill=True)
                ax.add_patch(rect)

            x += w_delta

    return fig





# fig = plt.figure(figsize=(13,13))
#
# ax = fig.add_subplot(111)
#
# rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='r', alpha=0.3)  # 左下起点，长，宽，颜色，α
#
# circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.5)  # 圆心，半径，颜色，α
#
# pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]], color='g', alpha=0.5)
#
# ax.add_patch(rect)
#
# ax.add_patch(circ)
#
# ax.add_patch(pgon)
#
# plt.savefig('/home/victorfang/test.png')
# plt.show()