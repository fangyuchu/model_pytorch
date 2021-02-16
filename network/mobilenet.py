import torch.nn as nn
import math


def expanded_cfg(c):
    v1_cfg = [c, 2*c, 4*c, 4*c, 8*c, 8*c, 16*c, 16*c, 16*c, 16*c, 16*c, 16*c, 32*c, 32*c]
    v2_cfg = [32, 96, 144, 144, 192, 192, 192, 384, 384, 384, 384, 576, 576, 576, 960, 960, 960, 1280]
    multiplier = c / 32
    if c != 32:
        v2_cfg = [int(v * multiplier) for v in v2_cfg]

    cfg = {
        'mobilenet_v1': v1_cfg,
        'mobilenet_v2': v2_cfg
    }
    return cfg

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

        return x

class ConvDepthWise(nn.Module):

    def __init__(self, inp, oup, stride):
        super(ConvDepthWise, self).__init__()
        self.conv1 = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.bn_1 = nn.BatchNorm2d(inp)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x



class MobileNet(nn.Module):
    def __init__(self, n_class, in_channel=32, multiplier=1.0, cfg=None):
        super(MobileNet, self).__init__()
        # original
        if cfg is None:
            cfg = expanded_cfg(in_channel)['mobilenet_v1']

        self.conv1 = ConvBNReLU(3, cfg[0], 3, 2, 1)
        self.features = self._make_layers(cfg[0], cfg[1:], ConvDepthWise)
        self.pool = nn.AvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1], n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.pool(x)  # global average pooling
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x

    def _make_layers(self, in_planes, cfg, layer):
        layers = []
        for i, x in enumerate(cfg):
            out_planes = x
            stride = 2 if i in [1, 3, 5, 11] else 1
            layers.append(layer(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v1(num_class, in_channel=32, multiplier=1.0, cfg=None):
    return MobileNet(num_class, in_channel, multiplier, cfg)


class InvertedBlock(nn.Module):
    def __init__(self, inp, oup, hid, stride):
        super(InvertedBlock, self).__init__()
        self.hid = hid
        # pw
        self.conv1 = nn.Conv2d(inp, hid, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hid)
        self.relu1 = nn.ReLU6(inplace=True)
        # dw
        self.conv2 = nn.Conv2d(hid, hid, 3, stride, 1, groups=hid, bias=False)
        self.bn2 = nn.BatchNorm2d(hid)
        self.relu2 = nn.ReLU6(inplace=True)
        # pw-linear
        self.conv3 = nn.Conv2d(hid, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, hid, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        if hid == inp:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hid, hid, 3, stride, 1, groups=hid, bias=False),
                nn.BatchNorm2d(hid),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hid, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = InvertedBlock(inp, oup, hid, stride)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, in_channel=32, multiplier=1.0, cfg=None):
        super(MobileNetV2, self).__init__()
        output_channels = [
            16, 24, 24, 32, 32, 32, 64, 64, 64, 64,
            96, 96, 96, 160, 160, 160, 320
        ]
        for i in range(len(output_channels)):
            output_channels[i] = int(multiplier * output_channels[i])

        if cfg is None:
            cfg = expanded_cfg(in_channel)['mobilenet_v2']

        self.features = [ConvBNReLU(3, cfg[0], kernel_size=3, stride=2, padding=1)]
        # building inverted residual blocks
        inp = cfg[0]
        for j, (hid, oup) in enumerate(zip(cfg[:-1], output_channels)):
            if j in [1, 3, 6, 13]:
                stride = 2
            else:
                stride = 1
            self.features.append(InvertedResidual(inp, oup, hid, stride))
            inp = oup

        # building last several layers
        self.features.append(ConvBNReLU(inp, cfg[-1], kernel_size=1, stride=1, padding=0))
        self.features.append(nn.AvgPool2d(7))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1], n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenet_v2(num_class, in_channel=32, multiplier=1.0, cfg=None):
    return MobileNetV2(num_class, in_channel, multiplier, cfg)



# #pytorch official implementation
# from torch import nn
# from torch.hub import load_state_dict_from_url
#
#
# __all__ = ['MobileNetV2', 'mobilenet_v2']
#
#
# model_urls = {
#     'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
# }
#
#
# def _make_divisible(v, divisor, min_value=None):
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     :param v:
#     :param divisor:
#     :param min_value:
#     :return:
#     """
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v
#
#
# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
#         padding = (kernel_size - 1) // 2
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
#             nn.BatchNorm2d(out_planes),
#             nn.ReLU6(inplace=True)
#         )
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
#
#         hidden_dim = int(round(inp * expand_ratio))
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#         layers = []
#         if expand_ratio != 1:
#             # pw
#             layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
#         layers.extend([
#             # dw
#             ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
#             # pw-linear
#             nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(oup),
#         ])
#         self.conv = nn.Sequential(*layers)
#
#     def forward(self, x):
#         if self.use_res_connect:
#             # print('yes')
#             return x + self.conv(x)
#         else:
#             # print('no')
#             return self.conv(x)
#
#
# class MobileNetV2(nn.Module):
#     def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
#         """
#         MobileNet V2 main class
#
#         Args:
#             num_classes (int): Number of classes
#             width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
#             inverted_residual_setting: Network structure
#             round_nearest (int): Round the number of channels in each layer to be a multiple of this number
#             Set to 1 to turn off rounding
#         """
#         super(MobileNetV2, self).__init__()
#         block = InvertedResidual
#         input_channel = 32
#         last_channel = 1280
#
#         if inverted_residual_setting is None:
#             inverted_residual_setting = [
#                 # t, c, n, s
#                 [1, 16, 1, 1],
#                 [6, 24, 2, 2],
#                 [6, 32, 3, 2],
#                 [6, 64, 4, 2],
#                 [6, 96, 3, 1],
#                 [6, 160, 3, 2],
#                 [6, 320, 1, 1],
#             ]
#
#         # only check the first element, assuming user knows t,c,n,s are required
#         if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
#             raise ValueError("inverted_residual_setting should be non-empty "
#                              "or a 4-element list, got {}".format(inverted_residual_setting))
#
#         # building first layer
#         input_channel = _make_divisible(input_channel * width_mult, round_nearest)
#         self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
#         features = [ConvBNReLU(3, input_channel, stride=2)]
#         # building inverted residual blocks
#         for t, c, n, s in inverted_residual_setting:
#             output_channel = _make_divisible(c * width_mult, round_nearest)
#             for i in range(n):
#                 stride = s if i == 0 else 1
#                 features.append(block(input_channel, output_channel, stride, expand_ratio=t))
#                 input_channel = output_channel
#         # building last several layers
#         features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*features)
#
#         # building classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(self.last_channel, num_classes),
#         )
#
#         # weight initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.mean([2, 3])
#         x = self.classifier(x)
#         return x
#
#
# def mobilenet_v2(pretrained=False, progress=True, **kwargs):
#     """
#     Constructs a MobileNetV2 architecture from
#     `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     model = MobileNetV2(**kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
