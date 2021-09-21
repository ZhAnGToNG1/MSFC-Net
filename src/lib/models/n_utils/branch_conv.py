from models.networks.DCNv2.DCN.dcn_v2 import dcn_v2_conv
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math

class branch_conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation=1,
                 groups = 1,
                 deformable_groups = 1,
                 bias = False,
                 part_deform = False):
        super(branch_conv,self).__init__()

        assert not bias

        assert in_channels % groups == 0, \
        'in channels {} connot be divisible by groups{}'.format(
            in_channels,groups
        )

        assert out_channels % groups == 0, \
        'out channels {} connot be divisible by groups{}'.format(
            out_channels,groups
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.part_deform = part_deform

        if self.part_deform:
            self.conv_offset = nn.Conv2d(
                self.in_channels,
                self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
                kernel_size = self.kernel_size,
                stride = _pair(self.stride),
                padding = _pair(self.padding),
                bias= True)
            self.init_offset()

        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        self.level = 1

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels//self.groups,
                         *self.kernel_size)
        )

        self.reset_parameters()


    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        self.weight.data.uniform_(-stdv,stdv)

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()


    def forward(self, i  ,x ):
        if i < self.level or not self.part_deform:
            return F.conv2d(x, self.weight, bias = self.bias, stride=self.stride, padding=self.padding,
                                              dilation=self.dilation, groups=self.groups)

        out = self.conv_offset(x)

        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(x, offset, mask, self.weight, self.bias ,self.stride, self.padding,
                           self.dilation, self.deformable_groups)