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
        mask = torch.ones(conv.out_channels)
        self.register_buffer('mask', mask)  # register self.mask as buffer in pytorch module

    def forward(self, input):
        if self.weight.device.type!='cuda' or self.mask.device.type !='cuda':
            print(self,'!!!')
        masked_weight = self.weight * self.mask.view(-1, 1, 1, 1)
        if self.bias is None:
            masked_bias = None
        else:
            masked_bias = self.bias * self.mask.view(-1)

        out = nn.functional.conv2d(input, masked_weight, masked_bias, self.stride,
                                   self.padding, self.dilation, self.groups)


        return out


class block_with_mask_shortcut(conv2d_with_mask):
    def __init__(self, conv, w_in):
        super(block_with_mask_shortcut, self).__init__(conv)
        w_out = int((w_in + 2 * conv.padding[0] - conv.kernel_size[0]) / conv.stride[0]) + 1
        if w_in != w_out or conv.in_channels != conv.out_channels:
            # add a shortcut with 1x1 conv
            # (w_in+2p-k)/(w_out-1) <= stride <= (w_in+2p-k)/(w_out)
            stride = int((w_in + 2 * conv.padding[0] - conv.kernel_size[0]) / (w_out - 1))
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=conv.in_channels,
                          out_channels=conv.out_channels,
                          stride=stride,
                          kernel_size=1),
                nn.BatchNorm2d(conv.out_channels)
            )
        else:
            self.downsample = nn.Sequential()  # idendity function

        self.bn=nn.BatchNorm2d(conv.out_channels)  # need to be updated in the net rather than in module



    def forward(self, input):
        x = super().forward(input)
        downsample=self.downsample(input)
        x = x + downsample
        x=self.bn(x)
        return x


class conv2d_with_mask_and_variable_shortcut(conv2d_with_mask):
    def __init__(self, conv, w_in,add_shortcut_ratio=0):
        super(conv2d_with_mask_and_variable_shortcut, self).__init__(conv)
        w_out = int((w_in + 2 * conv.padding[0] - conv.kernel_size[0]) / conv.stride[0]) + 1
        if w_in != w_out or conv.in_channels > conv.out_channels:
            # add a shortcut with 1x1 conv
            # (w_in+2p-k)/(w_out-1) <= stride <= (w_in+2p-k)/(w_out)
            stride = int((w_in + 2 * conv.padding[0] - conv.kernel_size[0]) / (w_out - 1))
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=conv.in_channels,
                          out_channels=conv.out_channels,
                          stride=stride,
                          kernel_size=1),
                nn.BatchNorm2d(conv.out_channels)
            )
        else:
            self.downsample = nn.Sequential()  # idendity function
        self.add_shortcut_ratio=add_shortcut_ratio
        self.w_in=w_in
        self.w_out=int((self.w_in + 2 * self.padding[0] - self.kernel_size[0]) //
                    self.stride[0] + 1)
        self.flops=None

    def compute_flops(self, in_channels, out_channels, multi_add=1):
        out_h = int((self.w_in + 2 * self.padding[0] - self.kernel_size[0]) //
                    self.stride[0] + 1)
        out_w = int((self.w_in + 2 * self.padding[1] - self.kernel_size[1]) //
                    self.stride[1] + 1)
        self.flops = in_channels * out_channels * self.kernel_size[0] * \
                     self.kernel_size[1] * out_h * out_w // self.groups * multi_add
        return self.flops

    def compute_downsample_flops(self, multi_add=1):
        if len(self.downsample) == 0:
            return 0
        else:
            # flops of conv
            conv = self.downsample[0]
            out_h = int((self.w_in + 2 * conv.padding[0] - conv.kernel_size[0]) //
                        conv.stride[0] + 1)
            out_w = int((self.w_in + 2 * conv.padding[1] - conv.kernel_size[1]) //
                        conv.stride[1] + 1)
            flops = conv.in_channels * conv.out_channels * conv.kernel_size[0] * \
                    conv.kernel_size[1] * out_h * out_w // conv.groups * multi_add
            # flops of bn
            normalize_ops = out_h * out_w * conv.out_channels
            scale_shift = normalize_ops
            flops += normalize_ops + scale_shift
            return flops

    def forward(self, input):
        x = super().forward(input)
        # zero_maps=float(((x[0, :, :, :] != 0).sum(axis=(1, 2))==0).sum()/float(x.shape[1]))
        #print('after conv:',zero_maps)
        pruned = self.out_channels != self.mask.size()[0]  # shortcut will not be added if conv has been pruned
        if pruned is False:  # the mask and shortcut is still working
            if torch.sum(self.mask == 0) / float(len(self.mask)) >= self.add_shortcut_ratio:  # ratio of unmasked conv is large
                #print('has shortcut')
                downsample = self.downsample(input)  # add shortcut
                if downsample.size() != x.size():  # if number of input feature maps is less than that of output's
                    #print('has add_zero shortcut')
                    add_zeros = torch.zeros(x.shape[0], x.shape[1] - downsample.shape[1], x.shape[2], x.shape[3])
                    add_zeros = add_zeros.to(input.device)
                    downsample = torch.cat((downsample, add_zeros), 1)
                x = x + downsample
                # zero_maps = float(((x[0, :, :, :] != 0).sum(axis=(1, 2)) == 0).sum() / float(x.shape[1]))
                #print('after shortcut:', zero_maps)
            else:  # shortcut will not be added if only few filters are pruned
                pass

        return x




class block_with_mask_weighted_shortcut(block_with_mask_shortcut):
    def __init__(self, conv, w_in):
        super(block_with_mask_weighted_shortcut, self).__init__(conv, w_in)
        # self.bn = nn.BatchNorm2d(conv.out_channels)  # need to be updated in the net rather than in module
        shortcut_mask = torch.ones(1)
        self.register_buffer('shortcut_mask', shortcut_mask)  # register self.mask as buffer in pytorch module

    def forward(self, input):
        out = super(block_with_mask_shortcut, self).forward(input)


        out =out * self.shortcut_mask
        out = out +self.downsample(input)
        out = self.bn(out)
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

