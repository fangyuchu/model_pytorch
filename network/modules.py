from torch import nn
import torch
import torch.nn.functional as F


class conv2d_with_mask(nn.modules.Conv2d):
    def __init__(self, conv):
        super(conv2d_with_mask, self).__init__(
            conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
            padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=(conv.bias is not None))
        self.weight = conv.weight
        if self.bias is not None:
            self.bias = conv.bias
        # self.mask=nn.Parameter(torch.randn((conv.out_channels,1)))
        # self.mask=torch.randn((conv.out_channels,1))
        mask = torch.ones(conv.out_channels)
        self.register_buffer('mask', mask)  # register self.mask as buffer in pytorch module

    def forward(self, input):
        masked_weight = self.weight * self.mask.view(-1, 1, 1, 1)
        if self.bias is None:
            masked_bias = None
        else:
            masked_bias = self.bias * self.mask.view(-1)
        # 考虑conv减了之后bn怎么办
        # 暂时通过bn不track running stats解决
        try:
            out = nn.functional.conv2d(input, masked_weight, masked_bias, self.stride,
                                       self.padding, self.dilation, self.groups)
        except RuntimeError:
            print()

        return out

class BasicBlock_with_mask(nn.Module):
    def __init__(self, basic_block):#in_planes, planes, stride=1, option='A'):
        super(BasicBlock_with_mask, self).__init__()
        self.conv1 = basic_block.conv1#nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = basic_block.bn1#nn.BatchNorm2d(planes)
        self.relu1 = basic_block.relu1#nn.ReLU(True)
        self.conv2 = basic_block.conv2#nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = basic_block.bn2#nn.BatchNorm2d(planes)
        self.planes = basic_block.planes#planes  # changed
        self.relu2 = basic_block.relu2#nn.ReLU(True)
        self.downsample = basic_block.downsample#nn.Sequential()
        self.mask=nn.Parameter(data=torch.zeros(1),requires_grad=True)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def tmp_func(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes // 4, self.planes // 4), "constant", 0)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * self.mask
        out += self.downsample(x)
        out = self.relu2(out)
        return out

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class identity_map(nn.Module):
    def __init__(self):
        super(identity_map,self).__init__()

    def forward(self, input):
        return input

