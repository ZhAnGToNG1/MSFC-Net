import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SeparableConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels=None, norm=True, activation=False):
        super(SeparableConvBlock,self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,
                                        groups=in_channels,bias=False , padding=1)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,padding=0)

        self.norm = norm

        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels,momentum=0.01,eps=1e-3)

        self.activation = activation

        if self.activation:
            self.swish = Swish()

        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.kaiming_normal_(self.pointwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0)

    def forward(self, x):

        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.swish(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.pool = nn.MaxPool2d(*args, **kwargs)

        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2



    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left , right , top , bottom])

        x = self.pool(x)
        return x
